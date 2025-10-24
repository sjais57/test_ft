import os
import sys
import json
import subprocess
import hashlib
import secrets
from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator, model_serializer
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uvicorn
import re
from config import SUPERUSER_SECRET_HASH, SUPERUSER_SALT
import asyncio
from secret_manager import get_secret_manager, SecretManager
from vault_client import MultiVaultManager, VaultError
from encryption_utils import get_encryption_manager
import logging
import pathlib

# ==================== LOGGING CONFIGURATION ====================

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logs_dir = pathlib.Path("/test/logs/fd")
logs_dir.mkdir(parents=True, exist_ok=True)  # parents=True creates intermediate directories

# Create log filename with current date
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename = logs_dir / f"control_tower_{current_date}.log"

# Set umask to create files with 755 permissions
original_umask = os.umask(0o022)  # 777 - 022 = 755

try:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_filename), mode="a")
        ]
    )
finally:
    # Restore original umask
    os.umask(original_umask)

logger = logging.getLogger("control_tower")

# ==================== FASTAPI APP INITIALIZATION ====================

app = FastAPI(title="DSPAI - Control Tower", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow cross-origin (optional, production should restrict this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting DSP AI Control Tower application.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down DSP AI Control Tower application.")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code} for {request.url}")
    return response

@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    logger.info("Swagger UI accessed")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="DSPAI - Control Tower",
        swagger_favicon_url="/static/control-tower.ico"
    )

# ==================== DATA MODELS ====================

class PolicyEvaluationRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="Input data to evaluate against the policy")

class PolicyEvaluationResponse(BaseModel):
    result: Dict[str, Any]
    allow: bool
    policy_path: str

class ClientSecretRequest(BaseModel):
    client_id: str = Field(..., description="Client ID for which to generate a secret")
    plain_secret: str = Field(..., description="Plain text secret to hash")

class ClientSecretResponse(BaseModel):
    client_id: str
    hashed_secret: str
    salt: str

class UserPoliciesRequest(BaseModel):
    user_id: str = Field(..., description="User ID to find applicable policies for")
    group_ids: List[str] = Field(default=[], description="Optional list of group IDs the user belongs to")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "nbkABCD",
                    "group_ids": ["unix_grp1", "unix_grp2"]
                }
            ]
        }
    }

class JupyterLabRequest(BaseModel):
    aihpc_lane: str = Field(..., description="Environment type to deploy to (e.g., 'training_dev', 'training_prod')")
    username: str = Field(..., description="Username for the Jupyter Lab")
    conda_env: str = Field(..., description="Conda environment to use")
    port: int = Field(8888, description="Port to run Jupyter Lab on")
    aihpc_env: str = Field("dev", description="AIHPC environment to use (e.g., 'dev', 'prod')")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "aihpc_lane": "training_dev",
                    "username": "user123",
                    "conda_env": "pytorch",
                    "port": 8888,
                    "aihpc_env": "dev"
                }
            ]
        }
    }

class ModelDeploymentRequest(BaseModel):
    aihpc_lane: str = Field(..., description="Environment type to deploy to (e.g., 'inference_dev', 'inference_prod')")
    username: str = Field(..., description="Username for the model deployment")
    model_name: str = Field(..., description="Name of the model to deploy")
    conda_env: str = Field(..., description="Conda environment to use")
    script_path: str = Field(..., description="Path to the deployment script")
    model_dir: str = Field(..., description="Directory containing the model files")
    port: int = Field(8000, description="Port to run the model server on")
    workers: int = Field(2, description="Number of workers for the model server")
    aihpc_env: str = Field("dev", description="AIHPC environment to use (e.g., 'dev', 'prod')")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "aihpc_lane": "inference_dev",
                    "username": "user123",
                    "model_name": "sentiment_analysis",
                    "conda_env": "pytorch",
                    "script_path": "app.server",
                    "model_dir": "/models/sentiment",
                    "port": 8000,
                    "workers": 2,
                    "aihpc_env": "dev"
                }
            ]
        }
    }

class HpcTemplateResponse(BaseModel):
    template: Dict[str, Any]
    message: str

class PolicyRequest(BaseModel):
    client_id: str = Field(..., description="Client ID (policy file name without .rego extension)")
    policy_content: str = Field(..., description="Full content of the Rego policy file")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "client_id": "example_client",
                    "policy_content": "package dspai.policy\nimport rego.v1\n\n# Client authentication\nclient_secret := \"hashed_secret\"\nclient_salt := \"random_salt\"\n\n# Default deny\ndefault allow := false\n\nallow := true {\n    input.action == \"read\"\n}"
                }
            ]
        }
    }

class PolicyStatusRequest(BaseModel):
    enabled: bool = Field(..., description="Whether the policy should be enabled or disabled")

# ==================== PROJECT MANIFEST MODELS ====================

class ModuleType(str, Enum):
    JWT_CONFIG = "jwt_config"
    RAG_CONFIG = "rag_config"
    API_GATEWAY = "api_gateway"
    INFERENCE_ENDPOINT = "inference_endpoint"
    SECURITY = "security"
    MONITORING = "monitoring"
    MODEL_REGISTRY = "model_registry"
    DATA_PIPELINE = "data_pipeline"
    DEPLOYMENT = "deployment"
    RESOURCE_MANAGEMENT = "resource_management"
    NOTIFICATIONS = "notifications"
    BACKUP_RECOVERY = "backup_recovery"
    VAULT = "vault"

class ModuleStatus(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"
    DEVELOPMENT = "development"

class JWTConfigModule(BaseModel):
    # Core JWT fields
    secret_key: Optional[str] = Field(None, description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    expiration_minutes: int = Field(default=30, description="Token expiration time in minutes")
    issuer: Optional[str] = Field(None, description="JWT issuer")
    audience: Optional[str] = Field(None, description="JWT audience")
    refresh_token_enabled: bool = Field(default=True, description="Enable refresh tokens")
    
    # Extended fields for DSP AI JWT service integration
    id: Optional[str] = Field(None, description="JWT config identifier")
    owner: Optional[str] = Field(None, description="Config owner/team")
    service_url: Optional[str] = Field(None, description="JWT service URL for token generation/validation")
    claims: Optional[Dict[str, Any]] = Field(None, description="Static and dynamic claims configuration")
    
    # JWE (JSON Web Encryption) configuration - nested format only
    jwe_config: Optional[Dict[str, Any]] = Field(None, description="JWE encryption configuration")
    
class RAGConfigModule(BaseModel):
    vector_store_type: str = Field(..., description="Type of vector store (faiss, pinecone, etc.)")
    embedding_model: str = Field(..., description="Embedding model name")
    chunk_size: int = Field(default=512, description="Document chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    reranker_enabled: bool = Field(default=False, description="Enable reranking")
    query_expansion_enabled: bool = Field(default=False, description="Enable query expansion")
    
class APIGatewayModule(BaseModel):
    gateway_type: str = Field(default="generic", description="Gateway type")
    rate_limiting: Dict[str, int] = Field(default_factory=dict, description="Rate limiting rules")
    cors_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")
    authentication_required: bool = Field(default=True, description="Require authentication")
    api_versioning: str = Field(default="v1", description="API version")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    load_balancing_strategy: str = Field(default="round_robin", description="Load balancing strategy")
    
    @model_validator(mode='before')
    @classmethod
    def reject_apisix(cls, data: Any) -> Any:
        """Reject if gateway_type is apisix - should use APISIXGatewayModule instead"""
        if isinstance(data, dict) and data.get('gateway_type') == 'apisix':
            raise ValueError("Use APISIXGatewayModule for gateway_type='apisix'")
        return data
    
class APISIXPlugin(BaseModel):
    """APISIX plugin configuration"""
    name: str = Field(..., description="Plugin name (e.g., jwt-auth, limit-req, prometheus)")
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific configuration")
    priority: Optional[int] = Field(None, description="Plugin execution priority (higher runs first)")
    
class APISIXRoute(BaseModel):
    """APISIX route configuration"""
    name: str = Field(..., description="Route name")
    uri: str = Field(..., description="URI pattern for matching requests")
    methods: List[str] = Field(default_factory=lambda: ["GET", "POST"], description="HTTP methods")
    upstream_id: Optional[str] = Field(None, description="Reference to upstream service")
    upstream: Optional[Dict[str, Any]] = Field(None, description="Inline upstream configuration")
    service_id: Optional[str] = Field(None, description="Reference to service configuration")
    plugins: Union[List[APISIXPlugin], Dict[str, Any]] = Field(default_factory=dict, description="Plugins (list or dict format)")
    host: Optional[str] = Field(None, description="Host header for routing")
    priority: int = Field(default=0, description="Route priority (higher matches first)")
    vars: Optional[List[List[str]]] = Field(None, description="Advanced routing conditions")
    
class APISIXUpstream(BaseModel):
    """APISIX upstream configuration for load balancing"""
    name: str = Field(..., description="Upstream name")
    type: str = Field(default="roundrobin", description="Load balancing type (roundrobin, chash, least_conn)")
    nodes: Dict[str, int] = Field(..., description="Backend nodes {host:port: weight}")
    timeout: Dict[str, int] = Field(
        default_factory=lambda: {"connect": 30, "send": 30, "read": 30},
        description="Timeout settings in seconds"
    )
    retries: int = Field(default=1, description="Number of retries")
    health_check: Optional[Dict[str, Any]] = Field(None, description="Health check configuration")
    
class APISIXGatewayModule(BaseModel):
    """APISIX API Gateway configuration for AI services"""
    gateway_type: str = Field(default="apisix", description="Gateway type - must be 'apisix'")
    admin_api_url: str = Field(default="http://localhost:9080", description="APISIX Admin API URL")
    admin_key: str = Field(default="${APISIX_ADMIN_KEY}", description="Admin API key")
    gateway_url: str = Field(default="http://localhost:9080", description="Gateway URL for clients")
    dashboard_url: Optional[str] = Field(default="http://localhost:9000", description="APISIX Dashboard URL")
    
    # Core configurations
    routes: List[APISIXRoute] = Field(default_factory=list, description="Route configurations")
    upstreams: List[APISIXUpstream] = Field(default_factory=list, description="Upstream service configurations")
    consumer: Optional[Dict[str, Any]] = Field(None, description="APISIX consumer configuration")
    service: Optional[Dict[str, Any]] = Field(None, description="APISIX service configuration")
    
    # Global plugins that apply to all routes
    global_plugins: List[APISIXPlugin] = Field(default_factory=list, description="Global plugins")
    
    # Default plugin configurations
    jwt_auth_enabled: bool = Field(default=True, description="Enable JWT authentication globally")
    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting globally")
    logging_enabled: bool = Field(default=True, description="Enable logging globally")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    
    # Security configurations
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_cert: Optional[str] = Field(None, description="SSL certificate path")
    ssl_key: Optional[str] = Field(None, description="SSL private key path")
    
    # CORS configuration
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    
    # Default timeout and retry settings
    default_timeout: int = Field(default=60, description="Default request timeout in seconds")
    default_retries: int = Field(default=2, description="Default number of retries")
    
    # AI-specific configurations
    streaming_enabled: bool = Field(default=True, description="Enable streaming for LLM responses")
    response_buffering: bool = Field(default=False, description="Buffer responses before sending")
    request_buffering: bool = Field(default=True, description="Buffer requests before forwarding")
    
class InferenceEndpointModule(BaseModel):
    # APISIX route reference (if using APISIX gateway)
    apisix_route: Optional[str] = Field(None, description="APISIX route name to use for this endpoint")
    apisix_gateway_module: Optional[str] = Field(None, description="Name of the APISIX gateway module providing the route")
    
    # Direct endpoint configuration (optional if using APISIX route)
    model_name: Optional[str] = Field(None, description="Model name for inference")
    model_version: Optional[str] = Field(default="latest", description="Model version")
    endpoint_url: Optional[str] = Field(None, description="Inference endpoint URL")
    system_prompt: Optional[str] = Field(None, description="System prompt for LLM")
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens per response")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter")
    batch_size: Optional[int] = Field(default=1, description="Batch size for inference")
    
class SecurityModule(BaseModel):
    encryption_at_rest: bool = Field(default=True, description="Enable encryption at rest")
    encryption_in_transit: bool = Field(default=True, description="Enable encryption in transit")
    vulnerability_scanning: bool = Field(default=True, description="Enable vulnerability scanning")
    access_control_type: str = Field(default="rbac", description="Access control type (rbac, abac)")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    compliance_standards: List[str] = Field(default_factory=list, description="Compliance standards")
    
class MonitoringModule(BaseModel):
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    logging_level: str = Field(default="INFO", description="Logging level")
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    alerting_enabled: bool = Field(default=True, description="Enable alerting")
    dashboard_url: Optional[str] = Field(None, description="Monitoring dashboard URL")
    
class ModelRegistryModule(BaseModel):
    registry_type: str = Field(..., description="Model registry type (mlflow, wandb, etc.)")
    registry_url: str = Field(..., description="Model registry URL")
    auto_versioning: bool = Field(default=True, description="Enable automatic versioning")
    model_validation: bool = Field(default=True, description="Enable model validation")
    metadata_tracking: bool = Field(default=True, description="Enable metadata tracking")
    experiment_tracking: bool = Field(default=True, description="Enable experiment tracking")
    
class DataPipelineModule(BaseModel):
    pipeline_type: str = Field(..., description="Pipeline type (batch, streaming, hybrid)")
    data_sources: List[str] = Field(..., description="List of data sources")
    data_sinks: List[str] = Field(..., description="List of data sinks")
    processing_engine: str = Field(..., description="Processing engine (spark, airflow, etc.)")
    schedule: Optional[str] = Field(None, description="Pipeline schedule (cron format)")
    data_quality_checks: bool = Field(default=True, description="Enable data quality checks")
    
class DeploymentModule(BaseModel):
    deployment_strategy: str = Field(default="blue_green", description="Deployment strategy")
    container_registry: str = Field(..., description="Container registry URL")
    orchestration_platform: str = Field(..., description="Orchestration platform (k8s, docker-compose)")
    auto_scaling: bool = Field(default=True, description="Enable auto-scaling")
    rollback_enabled: bool = Field(default=True, description="Enable rollback")
    environment_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Environment-specific configs")
    
class ResourceManagementModule(BaseModel):
    compute_resources: Dict[str, Any] = Field(default_factory=dict, description="Compute resource allocations")
    storage_resources: Dict[str, Any] = Field(default_factory=dict, description="Storage resource allocations")
    network_resources: Dict[str, Any] = Field(default_factory=dict, description="Network resource allocations")
    auto_scaling_policies: Dict[str, Any] = Field(default_factory=dict, description="Auto-scaling policies")
    cost_optimization: bool = Field(default=True, description="Enable cost optimization")
    resource_quotas: Dict[str, Any] = Field(default_factory=dict, description="Resource quotas")
    
class NotificationModule(BaseModel):
    email_enabled: bool = Field(default=True, description="Enable email notifications")
    slack_enabled: bool = Field(default=False, description="Enable Slack notifications")
    webhook_enabled: bool = Field(default=False, description="Enable webhook notifications")
    notification_channels: Dict[str, Any] = Field(default_factory=dict, description="Notification channel configs")
    alert_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Alert rules")
    escalation_policies: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation policies")
    
class BackupRecoveryModule(BaseModel):
    backup_enabled: bool = Field(default=True, description="Enable automated backups")
    backup_frequency: str = Field(default="daily", description="Backup frequency")
    retention_policy: str = Field(default="30d", description="Backup retention policy")
    disaster_recovery_enabled: bool = Field(default=True, description="Enable disaster recovery")
    backup_storage_type: str = Field(default="cloud", description="Backup storage type")
    restore_testing: bool = Field(default=True, description="Enable restore testing")

class VaultInstanceConfig(BaseModel):
    """Configuration for a single Vault instance"""
    instance_name: str = Field(..., description="Unique name for this Vault instance")
    vault_url: str = Field(..., description="Vault server URL")
    vault_namespace: Optional[str] = Field(None, description="Vault namespace (Enterprise)")
    auth_method: str = Field(default="token", description="Authentication method (token, approle)")
    
    # Token authentication
    vault_token: Optional[str] = Field(None, description="Vault token (supports env:, config:, encrypted: prefixes)")
    
    # AppRole authentication
    role_id: Optional[str] = Field(None, description="AppRole Role ID (supports env:, config:, encrypted: prefixes)")
    secret_id: Optional[str] = Field(None, description="AppRole Secret ID (supports env:, config:, encrypted: prefixes)")
    
    # KV configuration
    kv_mount_point: str = Field(default="secret", description="KV secrets engine mount point")
    kv_version: int = Field(default=2, description="KV secrets engine version (1 or 2)")
    
    # Connection settings
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    timeout: int = Field(default=30, description="Request timeout in seconds")

class VaultModule(BaseModel):
    """HashiCorp Vault configuration module for multi-instance secret management"""
    # Multiple Vault instances
    vault_instances: List[VaultInstanceConfig] = Field(
        default_factory=list,
        description="List of Vault instances to connect to"
    )
    
    # External configuration
    vault_config_file: Optional[str] = Field(
        None,
        description="Path to external vault configuration JSON file"
    )
    
    secrets_config_file: Optional[str] = Field(
        None,
        description="Path to secrets configuration JSON file (for config: references)"
    )
    
    # Encryption settings
    encryption_enabled: bool = Field(default=True, description="Enable encryption for stored secrets")
    encryption_key_source: str = Field(
        default="env:ENCRYPTION_KEY",
        description="Source of encryption key (env:VAR_NAME, config:path, vault:instance:path#key)"
    )
    
    # Secret resolution settings
    cache_secrets: bool = Field(default=True, description="Cache resolved secrets in memory")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    
    # Fallback behavior
    allow_env_fallback: bool = Field(default=True, description="Allow fallback to environment variables")
    fail_on_missing_secret: bool = Field(default=True, description="Fail if secret cannot be resolved")

class ModuleCrossReference(BaseModel):
    """Cross-reference to another module for specific functionality"""
    module_name: str = Field(..., description="Name of the referenced module")
    module_type: Optional[str] = Field(None, description="Expected type of the referenced module")
    purpose: str = Field(..., description="Purpose of this cross-reference")
    required: bool = Field(default=True, description="Whether this reference is required")
    fallback: Optional[str] = Field(None, description="Fallback module if primary is not available")

class ModuleConfig(BaseModel):
    module_type: ModuleType = Field(..., description="Type of the module")
    name: str = Field(..., description="Module name")
    version: Optional[str] = Field(default="1.0.0", description="Module version")
    status: Optional[ModuleStatus] = Field(default=ModuleStatus.ENABLED, description="Module status")
    description: Optional[str] = Field(None, description="Module description")
    config: Union[
        JWTConfigModule,
        RAGConfigModule,
        APISIXGatewayModule,
        APIGatewayModule,
        InferenceEndpointModule,
        SecurityModule,
        MonitoringModule,
        ModelRegistryModule,
        DataPipelineModule,
        DeploymentModule,
        ResourceManagementModule,
        NotificationModule,
        BackupRecoveryModule,
        VaultModule,
        Dict[str, Any]
    ] = Field(..., description="Module-specific configuration")
    
    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data: Any) -> Any:
        """Custom validator to ensure config matches module_type"""
        if not isinstance(data, dict):
            return data
        
        module_type = data.get('module_type')
        config = data.get('config')
        
        if not module_type or not config:
            return data
        
        # Map module types to their config classes
        type_to_config = {
            'jwt_config': JWTConfigModule,
            'rag_config': RAGConfigModule,
            'api_gateway': APIGatewayModule,
            'inference_endpoint': InferenceEndpointModule,
            'security': SecurityModule,
            'monitoring': MonitoringModule,
            'model_registry': ModelRegistryModule,
            'data_pipeline': DataPipelineModule,
            'deployment': DeploymentModule,
            'resource_management': ResourceManagementModule,
            'notifications': NotificationModule,
            'backup_recovery': BackupRecoveryModule,
            'vault': VaultModule,
        }
        
        # Special handling for api_gateway - check if it's APISIX
        if module_type == 'api_gateway' and isinstance(config, dict):
            if config.get('gateway_type') == 'apisix':
                # Validate as APISIX config
                try:
                    data['config'] = APISIXGatewayModule.model_validate(config)
                except Exception:
                    pass  # Fall back to Union matching
            else:
                # Validate as generic API gateway
                try:
                    data['config'] = APIGatewayModule.model_validate(config)
                except Exception:
                    pass  # Fall back to Union matching
        elif module_type in type_to_config:
            # Validate config against the expected type
            config_class = type_to_config[module_type]
            try:
                data['config'] = config_class.model_validate(config)
            except Exception:
                # If validation fails, keep original and let Union matching handle it
                pass
        
        return data
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "module_type": "inference_endpoint",
                    "name": "my-llm",
                    "config": {
                        "model_name": "gpt-4",
                        "endpoint_url": "https://api.openai.com/v1/chat/completions"
                    }
                }
            ]
        }
    }
    
class ProjectManifest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier")
    project_name: str = Field(..., description="Human-readable project name")
    version: Optional[str] = Field(default="1.0.0", description="Manifest version")
    description: Optional[str] = Field(None, description="Project description")
    owner: str = Field(..., description="Project owner")
    tags: Optional[List[str]] = Field(default_factory=list, description="Project tags")
    environment: Optional[str] = Field(default="development", description="Target environment")
    environments: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Environment-specific configurations")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update timestamp")
    modules: List[ModuleConfig] = Field(..., description="List of module configurations")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
class ManifestRequest(BaseModel):
    manifest: ProjectManifest = Field(..., description="Project manifest")
    
class ManifestResponse(BaseModel):
    message: str
    manifest_id: str
    manifest_path: str
    
class ManifestListResponse(BaseModel):
    manifests: List[Dict[str, Any]]
    count: int
    
class ManifestValidationRequest(BaseModel):
    manifest: ProjectManifest = Field(..., description="Manifest to validate")
    
class ManifestValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# ==================== MANIFEST UTILITY FUNCTIONS ====================

def validate_manifest_dependencies(modules: List[ModuleConfig]) -> List[str]:
    """Validate manifest modules"""
    errors = []
    # Basic validation - check for duplicate module names
    module_names = [module.name for module in modules]
    duplicates = [name for name in set(module_names) if module_names.count(name) > 1]
    if duplicates:
        errors.append(f"Duplicate module names found: {', '.join(duplicates)}")
    
    return errors

def get_manifest_path(project_id: str) -> str:
    """Get the file path for a manifest"""
    return f"manifests/{project_id}.json"

def load_manifest(project_id: str) -> Optional[ProjectManifest]:
    """Load a manifest from file"""
    manifest_path = get_manifest_path(project_id)
    
    if not os.path.exists(manifest_path):
        return None
    
    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        return ProjectManifest.model_validate(manifest_data)
    except Exception as e:
        logger.error(f"Failed to load manifest {project_id}: {str(e)}")
        return None

def save_manifest(manifest: ProjectManifest) -> str:
    """Save a manifest to file"""
    manifest_path = get_manifest_path(manifest.project_id)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    # Update timestamp
    manifest.updated_at = datetime.now()
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)
    
    logger.info(f"Manifest saved: {manifest.project_id} at {manifest_path}")
    return manifest_path

def resolve_environment_variables(data: Any, manifest: ProjectManifest, secret_manager: Optional[SecretManager] = None) -> Any:
    """Recursively resolve environment variable placeholders, Vault references, and other secret sources"""
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            # Resolve both keys and values (important for upstream nodes)
            resolved_key = resolve_environment_variables(key, manifest, secret_manager) if isinstance(key, str) else key
            resolved_value = resolve_environment_variables(value, manifest, secret_manager)
            resolved[resolved_key] = resolved_value
        return resolved
    elif isinstance(data, list):
        return [resolve_environment_variables(item, manifest, secret_manager) for item in data]
    elif isinstance(data, str):
        # Handle secret manager resolution (vault:, config:, env:, encrypted:)
        if secret_manager and any(prefix in data for prefix in ["vault:", "config:", "env:", "encrypted:"]):
            try:
                return secret_manager.resolve_secret(data)
            except Exception as e:
                logger.warning(f"Failed to resolve secret reference '{data}': {str(e)}")
                return data
        
        # Handle environment variable substitution
        if "${" in data:
            resolved_value = data
            
            # Handle ${environments.STATIC_NAME.key} pattern (e.g., ${environments.common.secrets.key})
            static_env_pattern = r'\$\{environments\.([a-zA-Z0-9_-]+)\.([^}]+)\}'
            static_matches = re.findall(static_env_pattern, resolved_value)
            for env_name, key_path in static_matches:
                # Skip if this is the ${environment} variable itself
                if env_name == "${environment}":
                    continue
                    
                placeholder = f"${{environments.{env_name}.{key_path}}}"
                
                # Navigate to the environment value
                try:
                    if hasattr(manifest, 'environments') and manifest.environments:
                        env_data = manifest.environments.get(env_name, {})
                        
                        # Split the key path (e.g., "secrets.jwt_secret_key")
                        key_parts = key_path.split('.')
                        value = env_data
                        for part in key_parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        
                        if value is not None:
                            # If value is a secret reference, resolve it
                            if secret_manager and isinstance(value, str):
                                value = secret_manager.resolve_secret(str(value))
                            resolved_value = resolved_value.replace(placeholder, str(value))
                except (AttributeError, KeyError):
                    # Keep original placeholder if resolution fails
                    pass
            
            # Handle ${environments.${environment}.key} pattern (dynamic environment)
            env_pattern = r'\$\{environments\.\$\{environment\}\.([^}]+)\}'
            matches = re.findall(env_pattern, resolved_value)
            for match in matches:
                placeholder = f"${{environments.${{environment}}.{match}}}"
                
                # Navigate to the environment value
                try:
                    current_env = manifest.environment
                    if hasattr(manifest, 'environments') and manifest.environments:
                        env_data = manifest.environments.get(current_env, {})
                        
                        # Split the key path (e.g., "secrets.jwt_secret_key")
                        key_parts = match.split('.')
                        value = env_data
                        for part in key_parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        
                        if value is not None:
                            # If value is a secret reference, resolve it
                            if secret_manager and isinstance(value, str):
                                value = secret_manager.resolve_secret(str(value))
                            resolved_value = resolved_value.replace(placeholder, str(value))
                except (AttributeError, KeyError):
                    # Keep original placeholder if resolution fails
                    pass
            
            # Handle ${environment} pattern
            env_var_pattern = r'\$\{environment\}'
            resolved_value = re.sub(env_var_pattern, manifest.environment, resolved_value)
            
            # Handle other ${VARIABLE} patterns (environment variables)
            var_pattern = r'\$\{([A-Z_][A-Z0-9_]*)\}'
            matches = re.findall(var_pattern, resolved_value)
            for match in matches:
                placeholder = f"${{{match}}}"
                env_value = os.getenv(match)
                if env_value is not None:
                    resolved_value = resolved_value.replace(placeholder, env_value)
            
            return resolved_value
        return data
    else:
        return data


def get_resolved_manifest(project_id: str, resolve_env: bool = False) -> Optional[ProjectManifest]:
    """Load a manifest and optionally resolve environment variables and secrets"""
    manifest = load_manifest(project_id)
    if not manifest or not resolve_env:
        return manifest
    
    # Initialize secret manager if Vault module exists
    secret_manager = None
    for module in manifest.modules:
        if module.module_type == ModuleType.VAULT:
            try:
                # Access config as Pydantic model attributes
                vault_config = module.config
                
                # Get secrets config file path
                secrets_config_file = getattr(vault_config, 'secrets_config_file', None)
                
                secret_manager = get_secret_manager(
                    config_file_path=secrets_config_file,
                    force_new=True
                )
                
                # Add Vault instances (skip failed ones)
                vault_instances = getattr(vault_config, 'vault_instances', [])
                for vault_inst_config in vault_instances:
                    try:
                        vault_dict = vault_inst_config.dict() if hasattr(vault_inst_config, 'dict') else vault_inst_config
                        secret_manager.add_vault_instance_from_config(vault_dict)
                        logger.info(f"Initialized Vault instance: {vault_dict.get('instance_name', 'unknown')}")
                    except Exception as vault_error:
                        instance_name = vault_dict.get('instance_name', 'unknown') if 'vault_dict' in locals() else 'unknown'
                        logger.warning(f"Skipping Vault instance '{instance_name}': {str(vault_error)}")
                
                # Load external config if specified
                vault_config_file = getattr(vault_config, 'vault_config_file', None)
                if vault_config_file and os.path.exists(vault_config_file):
                    with open(vault_config_file, 'r') as f:
                        external_config = json.load(f)
                    for vault_config_data in external_config.get("vault_instances", []):
                        secret_manager.add_vault_instance_from_config(vault_config_data)
            except Exception as e:
                logger.error(f"Failed to initialize secret manager: {str(e)}")
                # Don't fail completely - secret manager might still work for config: references
                if secret_manager is None:
                    # Try to create a basic secret manager for config file resolution
                    try:
                        secret_manager = get_secret_manager(
                            config_file_path=secrets_config_file if 'secrets_config_file' in locals() else None,
                            force_new=True
                        )
                        logger.info("Created basic secret manager for config file resolution")
                    except Exception as ex:
                        logger.error(f"Failed to create basic secret manager: {str(ex)}")
            break
    
    # Resolve environment variables in-place on the manifest dict
    manifest_dict = manifest.model_dump()
    resolved_dict = resolve_environment_variables(manifest_dict, manifest, secret_manager)
    
    # Reconstruct modules preserving their types
    # We need to manually reconstruct each module to avoid Union type confusion
    if 'modules' in resolved_dict:
        resolved_modules = []
        for i, module_data in enumerate(resolved_dict['modules']):
            # Get the original module to preserve its type
            original_module = manifest.modules[i]
            # Update only the config with resolved values
            original_dict = original_module.model_dump()
            original_dict['config'] = module_data.get('config', original_dict['config'])
            # Reconstruct the module with the same type
            resolved_modules.append(ModuleConfig.model_validate(original_dict))
        resolved_dict['modules'] = resolved_modules
    
    # Now validate the full manifest
    return ProjectManifest.model_validate(resolved_dict)

def get_resolved_module(project_id: str, module_name: str, resolve_env: bool = False) -> Optional[ModuleConfig]:
    """Get a specific module with optional environment resolution"""
    manifest = load_manifest(project_id)
    if not manifest:
        return None
    
    # Find the module
    target_module = None
    for module in manifest.modules:
        if module.name == module_name:
            target_module = module
            break
    
    if not target_module:
        return None
    
    if not resolve_env:
        return target_module
    
    # Resolve environment variables
    module_dict = target_module.model_dump()
    resolved_dict = resolve_environment_variables(module_dict, manifest)
    
    # Preserve the original module structure, only update config
    original_dict = target_module.model_dump()
    original_dict['config'] = resolved_dict.get('config', original_dict['config'])
    
    return ModuleConfig.model_validate(original_dict)

def analyze_cross_references(modules: List[ModuleConfig]) -> Dict[str, Any]:
    """Analyze module capabilities (deprecated - cross_references removed)"""
    module_graph = {}
    
    # Build module capability map based on type
    for module in modules:
        services = []
        if module.module_type == "jwt_config":
            services.extend(["authentication", "authorization", "token_validation"])
        elif module.module_type == "monitoring":
            services.extend(["logging", "metrics", "tracing", "health_checks"])
        elif module.module_type == "security":
            services.extend(["encryption", "access_control", "audit_logging"])
        elif module.module_type == "notifications":
            services.extend(["alerting", "notifications", "escalation"])
        elif module.module_type == "backup_recovery":
            services.extend(["backup", "disaster_recovery", "data_protection"])
        elif module.module_type == "resource_management":
            services.extend(["scaling", "resource_allocation", "cost_optimization"])
        elif module.module_type == "model_registry":
            services.extend(["model_versioning", "experiment_tracking", "model_validation"])
        elif module.module_type == "api_gateway":
            services.extend(["routing", "rate_limiting", "cors", "load_balancing"])
        elif module.module_type == "rag_config":
            services.extend(["knowledge_retrieval", "document_search", "semantic_search"])
        elif module.module_type == "inference_endpoint":
            services.extend(["llm_inference", "text_generation", "model_serving"])
        elif module.module_type == "data_pipeline":
            services.extend(["data_processing", "etl", "data_quality"])
        elif module.module_type == "deployment":
            services.extend(["container_deployment", "orchestration", "scaling"])
            
        module_graph[module.name] = {
            "provides": services,
            "module_type": module.module_type
        }
    
    return module_graph

def get_cross_reference_suggestions(modules: List[ModuleConfig]) -> Dict[str, List[str]]:
    """Get suggestions for module relationships (deprecated - cross_references removed)"""
    # Return empty suggestions since cross_references field has been removed
    return {}

def list_manifests() -> List[Dict[str, Any]]:
    """List all available manifests"""
    manifests = []
    manifest_dir = "manifests"
    
    if not os.path.exists(manifest_dir):
        return manifests
    
    for file in os.listdir(manifest_dir):
        if file.endswith(".json"):
            project_id = file.replace(".json", "")
            manifest = load_manifest(project_id)
            if manifest:
                manifests.append({
                    "project_id": manifest.project_id,
                    "project_name": manifest.project_name,
                    "version": manifest.version,
                    "environment": manifest.environment,
                    "owner": manifest.owner,
                    "module_count": len(manifest.modules),
                    "created_at": manifest.created_at,
                    "updated_at": manifest.updated_at
                })
    
    logger.info(f"Listed {len(manifests)} manifests")
    return manifests

def extract_aihpc_config(policy_content: str, aihpc_env: str, aihpc_lane: str) -> Dict[str, str]:
    """Extract AIHPC configuration from policy content for a specific environment and lane"""
    # Extract aihpc configuration for the specified environment
    aihpc_match = re.search(r'aihpc\.' + aihpc_env + r'\s*:=\s*({.*?})\s*$', policy_content, re.DOTALL | re.MULTILINE)
    if not aihpc_match:
        logger.error(f"AIHPC configuration not defined for environment: {aihpc_env}")
        raise HTTPException(status_code=400, detail=f"AIHPC configuration not defined for environment: {aihpc_env}")
    
    # Extract the specific environment configuration
    # Modified regex to properly handle nested braces and find the specific lane
    env_config_str = aihpc_match.group(1)
    lane_pattern = rf'"{aihpc_lane}"\s*:\s*({{\s*"[^"]+"\s*:\s*"[^"]+"\s*,\s*"[^"]+"\s*:\s*"[^"]+"\s*,\s*"[^"]+"\s*:\s*\d+\s*,\s*.*?}})'
    env_match = re.search(lane_pattern, env_config_str, re.DOTALL)
    
    if not env_match:
        logger.error(f"Environment type not defined: {aihpc_lane}")
        raise HTTPException(status_code=400, detail=f"Environment type not defined: {aihpc_lane}")
    
    env_config = env_match.group(1)
    
    # Define the configuration fields to extract with their default values
    config_fields = {
        "account": None,
        "partition": None,
        "num_gpu": "1"  # Default value
    }
    
    # Extract each field from the environment configuration
    for field, default in config_fields.items():
        if field == "num_gpu":
            # Special case for num_gpu which is a number
            match = re.search(rf'"{field}"\s*:\s*(\d+)', env_config)
            if match:
                config_fields[field] = match.group(1)
        else:
            # For string fields
            match = re.search(rf'"{field}"\s*:\s*"([^"]+)"', env_config)
            if match:
                config_fields[field] = match.group(1)
            elif default is None:
                # Required field is missing
                logger.error(f"{field.capitalize()} not defined for environment: {aihpc_lane}")
                raise HTTPException(status_code=400, detail=f"{field.capitalize()} not defined for environment: {aihpc_lane}")
    
    logger.info(f"Extracted AIHPC config for {aihpc_lane}: {config_fields}")
    return config_fields

def hash_secret(secret: str, salt: str = None):
    """Hash a secret with a salt using SHA-256"""
    if salt is None:
        # Generate a random salt if none is provided
        salt = secrets.token_hex(16)
    
    # Combine the secret and salt and hash
    combined = secret + salt
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    
    return hashed, salt

async def authenticate_superuser(
    x_dspai_client_secret: str = Header(..., description="Superuser secret for authentication", alias="X-DSPAI-Client-Secret")
):
    """Authenticate superuser using the superuser secret"""
    # Check if superuser secret is provided
    superuser_hashed_secret, _ = hash_secret(x_dspai_client_secret, SUPERUSER_SALT)
    if superuser_hashed_secret != SUPERUSER_SECRET_HASH:
        # Add a delay to prevent timing attacks
        await asyncio.sleep(1)
        logger.warning("Invalid superuser credentials provided")
        raise HTTPException(status_code=401, detail="Invalid superuser credentials")
    
    logger.info("Superuser authenticated successfully")
    return True

async def authenticate_client(
    x_dspai_client_id: str = Header(..., description="Client ID (policy file name)", alias="X-DSPAI-Client-ID"),
    x_dspai_client_secret: str = Header(..., description="Client secret for authentication", alias="X-DSPAI-Client-Secret")
):
    """Authenticate client using client_id and client_secret from headers"""
    # Check if superuser secret is provided
    superuser_hashed_secret, _ = hash_secret(x_dspai_client_secret, SUPERUSER_SALT)
    if superuser_hashed_secret == SUPERUSER_SECRET_HASH:
        # Superuser authentication successful, construct policy path
        policy_path = f"policies/clients/{x_dspai_client_id}.rego"
        
        # Check if policy file exists
        if not os.path.exists(policy_path):
            logger.warning(f"Client ID not found: {x_dspai_client_id}")
            raise HTTPException(status_code=404, detail=f"Client ID not found: {x_dspai_client_id}")
        
        logger.info(f"Superuser access granted for client: {x_dspai_client_id}")
        return policy_path
    
    # Construct the policy path
    policy_path = f"policies/clients/{x_dspai_client_id}.rego"
    
    # Check if policy file exists
    if not os.path.exists(policy_path):
        logger.warning(f"Client ID not found: {x_dspai_client_id}")
        raise HTTPException(status_code=404, detail=f"Client ID not found: {x_dspai_client_id}")
    
    # Read the policy file to extract the client secret
    with open(policy_path, 'r') as f:
        policy_content = f.read()
    
    # Look for client_secret and salt in the policy file
    import re
    hashed_secret_match = re.search(r'client_secret\s*:=\s*"([^"]+)"', policy_content)
    salt_match = re.search(r'client_salt\s*:=\s*"([^"]+)"', policy_content)
    
    if not hashed_secret_match or not salt_match:
        logger.warning(f"Client secret or salt not defined in policy for: {x_dspai_client_id}")
        raise HTTPException(status_code=401, detail="Client secret or salt not defined in policy")
    
    stored_hashed_secret = hashed_secret_match.group(1)
    stored_salt = salt_match.group(1)
    
    # Hash the provided secret with the stored salt
    provided_hashed_secret, _ = hash_secret(x_dspai_client_secret, stored_salt)
    
    # Verify the client secret
    if provided_hashed_secret != stored_hashed_secret:
        logger.warning(f"Invalid client secret for: {x_dspai_client_id}")
        raise HTTPException(status_code=401, detail="Invalid client secret")
    
    logger.info(f"Client authenticated successfully: {x_dspai_client_id}")
    return policy_path

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "DSP AI Control Tower - OPA Policy Evaluator API. Swagger: /docs"}

@app.get("/policies")
async def list_policies():
    """List all available Rego policies"""
    logger.info("Listing all policies")
    policies = []
    client_dir = "policies/clients"
    
    if os.path.exists(client_dir):
        for file in os.listdir(client_dir):
            if file.endswith(".rego") and not file.endswith("_test.rego"):
                policy_path = os.path.join(client_dir, file)
                # Convert Windows path to Unix-style for consistency
                policy_path = policy_path.replace("\\", "/")
                
                # Read the policy file to check if it's enabled
                with open(policy_path, 'r') as f:
                    policy_content = f.read()
                
                # Check if policy is enabled (if enabled flag exists)
                enabled_match = re.search(r'policy_enabled\s*:=\s*(true|false)', policy_content)
                
                policy_info = {
                    "policy_path": policy_path,
                    "policy_name": os.path.basename(policy_path)
                }
                
                # Add enabled status if it exists
                if enabled_match:
                    policy_info["enabled"] = enabled_match.group(1) == "true"
                else:
                    policy_info["enabled"] = True  # Default to enabled if flag doesn't exist
                
                policies.append(policy_info)
    
    logger.info(f"Found {len(policies)} policies")
    return {"policies": policies}

@app.post("/generate-client-secret", response_model=ClientSecretResponse)
async def generate_client_secret(request: ClientSecretRequest):
    """Generate a hashed client secret with salt"""
    logger.info(f"Generating client secret for: {request.client_id}")
    hashed_secret, salt = hash_secret(request.plain_secret)
    
    logger.info(f"Client secret generated successfully for: {request.client_id}")
    return {
        "client_id": request.client_id,
        "hashed_secret": hashed_secret,
        "salt": salt
    }

@app.post("/evaluate", response_model=PolicyEvaluationResponse)
async def evaluate_policy(
    request: PolicyEvaluationRequest,
    policy_path: str = Depends(authenticate_client)
):
    """Evaluate input data against a specified Rego policy with client authentication via headers"""
    logger.info(f"Policy evaluation requested for: {policy_path}")
    
    # Create temporary input file
    input_file = "temp_input.json"
    with open(input_file, "w") as f:
        json.dump(request.input_data, f)
    
    try:
        # Run OPA evaluation using the opa.exe executable
        cmd = [
            "opa", "eval", 
            "--data", policy_path, 
            "--input", input_file, 
            "--format", "json",
            "data.dspai.policy"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"OPA evaluation failed for {policy_path}: {result.stderr}")
            raise HTTPException(
                status_code=500, 
                detail=f"OPA evaluation failed: {result.stderr}"
            )
        
        # Parse the OPA result
        opa_result = json.loads(result.stdout)
        
        # Extract the allow decision
        allow = False
        if "result" in opa_result and len(opa_result["result"]) > 0:
            if "allow" in opa_result["result"][0]["expressions"][0]["value"]:
                allow = opa_result["result"][0]["expressions"][0]["value"]["allow"]
        
        logger.info(f"Policy evaluation completed for {policy_path}: allow={allow}")
        return {
            "result": opa_result,
            "allow": allow,
            "policy_path": policy_path
        }
    
    except Exception as e:
        logger.error(f"Error evaluating policy {policy_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating policy: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.remove(input_file)

@app.post("/batch-evaluate")
async def batch_evaluate_policies(
    input_data: Dict[str, Any] = Body(..., description="Input data to evaluate against the policy"),
    policy_path: str = Depends(authenticate_client)
):
    """Evaluate input data against policy with client authentication via headers"""
    logger.info(f"Batch policy evaluation requested for: {policy_path}")
    results = []
    
    # Create temporary input file
    input_file = "temp_input.json"
    with open(input_file, "w") as f:
        json.dump(input_data, f)
    
    try:
        # Run OPA evaluation
        cmd = [
            "opa.exe", "eval", 
            "--data", policy_path, 
            "--input", input_file, 
            "--format", "json",
            "data.dspai.policy"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            results.append({
                "policy_path": policy_path,
                "error": f"OPA evaluation failed: {result.stderr}",
                "allow": False
            })
            logger.error(f"Batch OPA evaluation failed for {policy_path}: {result.stderr}")
        else:
            # Parse the OPA result
            opa_result = json.loads(result.stdout)
            
            # Extract the allow decision
            allow = False
            if "result" in opa_result and len(opa_result["result"]) > 0:
                if "allow" in opa_result["result"][0]["expressions"][0]["value"]:
                    allow = opa_result["result"][0]["expressions"][0]["value"]["allow"]
            
            results.append({
                "policy_path": policy_path,
                "result": opa_result,
                "allow": allow
            })
            logger.info(f"Batch policy evaluation completed for {policy_path}: allow={allow}")
    
    except Exception as e:
        results.append({
            "policy_path": policy_path,
            "error": f"Error evaluating policy: {str(e)}",
            "allow": False
        })
        logger.error(f"Error in batch policy evaluation for {policy_path}: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.remove(input_file)
    
    return {"results": results}

@app.post("/user-policies")
async def list_user_policies(request: UserPoliciesRequest):
    """List all policies applicable to a specific user and their groups"""
    logger.info(f"User policies requested for user: {request.user_id}, groups: {request.group_ids}")
    applicable_policies = []
    client_dir = "policies/clients"
    
    if not os.path.exists(client_dir):
        logger.info("No policies directory found")
        return {"policies": []}
    
    for file in os.listdir(client_dir):
        if file.endswith(".rego") and not file.endswith("_test.rego"):
            policy_path = os.path.join(client_dir, file)
            # Convert Windows path to Unix-style for consistency
            policy_path = policy_path.replace("\\", "/")
            
            # Read the policy file to extract user and group roles
            with open(policy_path, 'r') as f:
                policy_content = f.read()
            
            # Check if policy is enabled (if enabled flag exists)
            enabled_match = re.search(r'policy_enabled\s*:=\s*(true|false)', policy_content)
            # If enabled flag exists and is set to false, skip this policy
            if enabled_match and enabled_match.group(1) == "false":
                continue
            
            # Check if user is directly mentioned in user_roles
            user_match = re.search(rf'"{request.user_id}":\s*"([^"]+)"', policy_content)
            
            # Check if any of the user's groups are mentioned in group_roles
            group_matches = []
            for group_id in request.group_ids:
                group_match = re.search(rf'"{group_id}":\s*"([^"]+)"', policy_content)
                if group_match:
                    group_matches.append({
                        "group_id": group_id,
                        "role": group_match.group(1)
                    })
            
            # If either user or any group is found, add to applicable policies
            if user_match or group_matches:
                policy_info = {
                    "policy_path": policy_path,
                    "policy_name": os.path.basename(policy_path),
                }
                
                # Add enabled status if it exists
                if enabled_match:
                    policy_info["enabled"] = enabled_match.group(1) == "true"
                else:
                    policy_info["enabled"] = True  # Default to enabled if flag doesn't exist
                
                if user_match:
                    policy_info["user_role"] = user_match.group(1)
                
                if group_matches:
                    policy_info["group_roles"] = group_matches
                
                # Extract allowed actions based on roles
                roles_match = re.search(r'roles\s*:=\s*{([^}]+)}', policy_content, re.DOTALL)
                if roles_match:
                    roles_content = roles_match.group(1)
                    policy_info["available_actions"] = {}
                    
                    # Extract user's direct role actions if available
                    if user_match:
                        user_role = user_match.group(1)
                        role_actions_match = re.search(rf'"{user_role}":\s*\[(.*?)\]', roles_content)
                        if role_actions_match:
                            actions = re.findall(r'"([^"]+)"', role_actions_match.group(1))
                            policy_info["available_actions"]["user"] = actions
                    
                    # Extract group role actions
                    for group_match in group_matches:
                        group_role = group_match["role"]
                        role_actions_match = re.search(rf'"{group_role}":\s*\[(.*?)\]', roles_content)
                        if role_actions_match:
                            actions = re.findall(r'"([^"]+)"', role_actions_match.group(1))
                            if "groups" not in policy_info["available_actions"]:
                                policy_info["available_actions"]["groups"] = {}
                            policy_info["available_actions"]["groups"][group_match["group_id"]] = actions
                
                applicable_policies.append(policy_info)
    
    logger.info(f"Found {len(applicable_policies)} applicable policies for user {request.user_id}")
    return {"policies": applicable_policies}

def load_template(template_name: str) -> Dict[str, Any]:
    """Load a template from the templates directory"""
    template_path = os.path.join("templates", f"{template_name}.json")
    
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_name}")
        raise HTTPException(status_code=404, detail=f"Template not found: {template_name}")
    
    try:
        with open(template_path, 'r') as f:
            template = json.load(f)
        return template
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading template: {str(e)}")

@app.post("/templates/jupyter-lab", response_model=HpcTemplateResponse)
async def generate_jupyter_lab_template(
    request: JupyterLabRequest,
    policy_path: str = Depends(authenticate_client)
):
    """Generate a Jupyter Lab job template for HPC Slurm cluster"""
    logger.info(f"Generating Jupyter Lab template for user: {request.username}, lane: {request.aihpc_lane}")
    
    # Load the template
    template = load_template("jupyter_lab")
    
    # Extract policy information
    with open(policy_path, 'r') as f:
        policy_content = f.read()
    
    # Extract project name from policy
    project_match = re.search(r'project\s*:=\s*"([^"]+)"', policy_content)
    if project_match:
        project = project_match.group(1)
    else:
        # Use policy filename as project if not explicitly defined
        project = os.path.basename(policy_path).replace(".rego", "")
    
    # Extract allowed models if available
    allowed_models = []
    allowed_models_match = re.search(r'allowed_models\s*:=\s*\[(.*?)\]', policy_content, re.DOTALL)
    if allowed_models_match:
        models_str = allowed_models_match.group(1)
        allowed_models = [m.strip('"') for m in re.findall(r'"([^"]+)"', models_str)]
    
    # Extract AIHPC configuration
    aihpc_config = extract_aihpc_config(policy_content, request.aihpc_env, request.aihpc_lane)
    
    # Replace placeholders in the template
    template_str = json.dumps(template)
    template_str = template_str.replace("{project}", project)
    template_str = template_str.replace("{aihpc.account}", aihpc_config["account"])
    template_str = template_str.replace("{aihpc.partition}", aihpc_config["partition"])
    template_str = template_str.replace("{aihpc.num_gpu}", aihpc_config["num_gpu"])
    
    # Replace allowed_models if available
    if allowed_models:
        template_str = template_str.replace("{allowed_models}", ", ".join(allowed_models))
    else:
        template_str = template_str.replace("{allowed_models}", "")
    
    # Replace user-specific values
    template_str = template_str.replace("{username}", request.username)
    template_str = template_str.replace("{conda_env}", request.conda_env)
    template_str = template_str.replace("{port}", str(request.port))
    
    # Convert back to dictionary
    filled_template = json.loads(template_str)
    
    logger.info(f"Jupyter Lab template generated successfully for user: {request.username}")
    return {
        "template": filled_template,
        "message": "Jupyter Lab template generated successfully"
    }

@app.post("/templates/model-deployment", response_model=HpcTemplateResponse)
async def generate_model_deployment_template(
    request: ModelDeploymentRequest,
    policy_path: str = Depends(authenticate_client)
):
    """Generate a Model Deployment job template for HPC Slurm cluster"""
    logger.info(f"Generating Model Deployment template for user: {request.username}, model: {request.model_name}")
    
    # Load the template
    template = load_template("model_deployment")
    
    # Extract policy information
    with open(policy_path, 'r') as f:
        policy_content = f.read()
    
    # Extract project name from policy
    project_match = re.search(r'project\s*:=\s*"([^"]+)"', policy_content)
    if project_match:
        project = project_match.group(1)
    else:
        # Use policy filename as project if not explicitly defined
        project = os.path.basename(policy_path).replace(".rego", "")
    
    # Extract AIHPC configuration
    aihpc_config = extract_aihpc_config(policy_content, request.aihpc_env, request.aihpc_lane)
    
    # Replace placeholders in the template
    template_str = json.dumps(template)
    template_str = template_str.replace("{project}", project)
    template_str = template_str.replace("{aihpc.account}", aihpc_config["account"])
    template_str = template_str.replace("{aihpc.partition}", aihpc_config["partition"])
    template_str = template_str.replace("{aihpc.num_gpu}", aihpc_config["num_gpu"])
    
    # Replace user-specific values
    template_str = template_str.replace("model_name", request.model_name)
    template_str = template_str.replace("/home", f"/home/{request.username}/models/{request.model_name}")
    template_str = template_str.replace("/home/models/", f"/home/{request.username}/models/{request.model_name}")
    template_str = template_str.replace("/home/models/logs", f"/home/{request.username}/models/{request.model_name}/logs")
    template_str = template_str.replace("source activate", f"source activate {request.conda_env}; python -m {request.script_path} --model-dir={request.model_dir} --port={request.port} --workers={request.workers}")
    
    # Convert back to dictionary
    filled_template = json.loads(template_str)
    
    logger.info(f"Model Deployment template generated successfully for user: {request.username}, model: {request.model_name}")
    return {
        "template": filled_template,
        "message": "Model Deployment template generated successfully"
    }

@app.post("/policies/add", status_code=201)
async def add_policy(
    request: PolicyRequest,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Add a new Rego policy (superuser only)"""
    logger.info(f"Adding new policy: {request.client_id}")
    
    # Ensure the client_id is valid
    if not re.match(r'^[a-zA-Z0-9_]+$', request.client_id):
        logger.error(f"Invalid client_id format: {request.client_id}")
        raise HTTPException(status_code=400, detail="Invalid client_id format. Use only alphanumeric characters and underscores.")
    
    # Construct the policy path
    policy_path = f"policies/clients/{request.client_id}.rego"
    
    # Check if policy already exists
    if os.path.exists(policy_path):
        logger.error(f"Policy already exists: {request.client_id}")
        raise HTTPException(status_code=409, detail=f"Policy already exists: {request.client_id}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    
    # Add enabled flag if not already present in the policy content
    if "policy_enabled" not in request.policy_content:
        # Find the first line after package declaration to insert the enabled flag
        lines = request.policy_content.split('\n')
        package_index = -1
        for i, line in enumerate(lines):
            if line.startswith('package '):
                package_index = i
                break
        
        if package_index >= 0:
            # Insert after package and import statements
            insert_index = package_index + 1
            while insert_index < len(lines) and lines[insert_index].startswith('import '):
                insert_index += 1
            
            lines.insert(insert_index, "\n# Policy status - controls whether this policy is active")
            lines.insert(insert_index + 1, "policy_enabled := true")
            
            # Reassemble the policy content
            request.policy_content = '\n'.join(lines)
    
    # Write the policy file
    try:
        with open(policy_path, 'w') as f:
            f.write(request.policy_content)
        logger.info(f"Policy added successfully: {request.client_id}")
    except Exception as e:
        logger.error(f"Failed to write policy file {request.client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to write policy file: {str(e)}")
    
    return {"message": f"Policy added successfully: {request.client_id}", "policy_path": policy_path}

@app.put("/policies/update/{client_id}")
async def update_policy(
    client_id: str,
    request: PolicyRequest,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Update an existing Rego policy (superuser only)"""
    logger.info(f"Updating policy: {client_id}")
    
    # Ensure the client_id in path matches the one in request
    if client_id != request.client_id:
        logger.error(f"Client ID mismatch: path={client_id}, body={request.client_id}")
        raise HTTPException(status_code=400, detail="client_id in path must match client_id in request body")
    
    # Ensure the client_id is valid
    if not re.match(r'^[a-zA-Z0-9_]+$', request.client_id):
        logger.error(f"Invalid client_id format: {request.client_id}")
        raise HTTPException(status_code=400, detail="Invalid client_id format. Use only alphanumeric characters and underscores.")
    
    # Construct the policy path
    policy_path = f"policies/clients/{request.client_id}.rego"
    
    # Check if policy exists
    if not os.path.exists(policy_path):
        logger.error(f"Policy not found: {request.client_id}")
        raise HTTPException(status_code=404, detail=f"Policy not found: {request.client_id}")
    
    # Write the updated policy file
    try:
        with open(policy_path, 'w') as f:
            f.write(request.policy_content)
        logger.info(f"Policy updated successfully: {request.client_id}")
    except Exception as e:
        logger.error(f"Failed to update policy file {request.client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update policy file: {str(e)}")
    
    return {"message": f"Policy updated successfully: {request.client_id}", "policy_path": policy_path}

@app.delete("/policies/delete/{client_id}")
async def delete_policy(
    client_id: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Delete an existing Rego policy (superuser only)"""
    logger.info(f"Deleting policy: {client_id}")
    
    # Ensure the client_id is valid
    if not re.match(r'^[a-zA-Z0-9_]+$', client_id):
        logger.error(f"Invalid client_id format: {client_id}")
        raise HTTPException(status_code=400, detail="Invalid client_id format. Use only alphanumeric characters and underscores.")
    
    # Construct the policy path
    policy_path = f"policies/clients/{client_id}.rego"
    
    # Check if policy exists
    if not os.path.exists(policy_path):
        logger.error(f"Policy not found: {client_id}")
        raise HTTPException(status_code=404, detail=f"Policy not found: {client_id}")
    
    # Delete the policy file
    try:
        os.remove(policy_path)
        logger.info(f"Policy deleted successfully: {client_id}")
    except Exception as e:
        logger.error(f"Failed to delete policy file {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete policy file: {str(e)}")
    
    return {"message": f"Policy deleted successfully: {client_id}"}

@app.get("/policies/{client_id}")
async def get_policy(
    client_id: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Get the content of a specific Rego policy (superuser only)"""
    logger.info(f"Getting policy: {client_id}")
    
    # Ensure the client_id is valid
    if not re.match(r'^[a-zA-Z0-9_]+$', client_id):
        logger.error(f"Invalid client_id format: {client_id}")
        raise HTTPException(status_code=400, detail="Invalid client_id format. Use only alphanumeric characters and underscores.")
    
    # Construct the policy path
    policy_path = f"policies/clients/{client_id}.rego"
    
    # Check if policy exists
    if not os.path.exists(policy_path):
        logger.error(f"Policy not found: {client_id}")
        raise HTTPException(status_code=404, detail=f"Policy not found: {client_id}")
    
    # Read the policy file
    try:
        with open(policy_path, 'r') as f:
            policy_content = f.read()
        logger.info(f"Policy retrieved successfully: {client_id}")
    except Exception as e:
        logger.error(f"Failed to read policy file {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read policy file: {str(e)}")
    
    return {"client_id": client_id, "policy_content": policy_content, "policy_path": policy_path}

@app.patch("/policies/{client_id}/status")
async def update_policy_status(
    client_id: str,
    request: PolicyStatusRequest,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Enable or disable a policy (superuser only)"""
    logger.info(f"Updating policy status for {client_id}: enabled={request.enabled}")
    
    # Ensure the client_id is valid
    if not re.match(r'^[a-zA-Z0-9_]+$', client_id):
        logger.error(f"Invalid client_id format: {client_id}")
        raise HTTPException(status_code=400, detail="Invalid client_id format. Use only alphanumeric characters and underscores.")
    
    # Construct the policy path
    policy_path = f"policies/clients/{client_id}.rego"
    
    # Check if policy exists
    if not os.path.exists(policy_path):
        logger.error(f"Policy not found: {client_id}")
        raise HTTPException(status_code=404, detail=f"Policy not found: {client_id}")
    
    # Read the policy file
    try:
        with open(policy_path, 'r') as f:
            policy_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read policy file {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read policy file: {str(e)}")
    
    # Check if policy_enabled flag exists
    enabled_match = re.search(r'policy_enabled\s*:=\s*(true|false)', policy_content)
    
    if enabled_match:
        # Update the existing flag
        new_status = "true" if request.enabled else "false"
        updated_content = re.sub(
            r'policy_enabled\s*:=\s*(true|false)',
            f'policy_enabled := {new_status}',
            policy_content
        )
    else:
        # Add the flag if it doesn't exist
        # Find the first line after package declaration to insert the enabled flag
        lines = policy_content.split('\n')
        package_index = -1
        for i, line in enumerate(lines):
            if line.startswith('package '):
                package_index = i
                break
        
        if package_index >= 0:
            # Insert after package and import statements
            insert_index = package_index + 1
            while insert_index < len(lines) and lines[insert_index].startswith('import '):
                insert_index += 1
            
            new_status = "true" if request.enabled else "false"
            lines.insert(insert_index, "\n# Policy status - controls whether this policy is active")
            lines.insert(insert_index + 1, f"policy_enabled := {new_status}")
            
            # Reassemble the policy content
            updated_content = '\n'.join(lines)
        else:
            # If package declaration not found, just prepend the flag
            new_status = "true" if request.enabled else "false"
            updated_content = f"# Policy status - controls whether this policy is active\npolicy_enabled := {new_status}\n\n{policy_content}"
    
    # Write the updated policy file
    try:
        with open(policy_path, 'w') as f:
            f.write(updated_content)
        logger.info(f"Policy status updated for {client_id}: enabled={request.enabled}")
    except Exception as e:
        logger.error(f"Failed to update policy file {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update policy file: {str(e)}")
    
    status_text = "enabled" if request.enabled else "disabled"
    return {"message": f"Policy {status_text} successfully: {client_id}", "policy_path": policy_path, "enabled": request.enabled}

# ==================== MANIFEST API ENDPOINTS ====================

@app.get("/manifests", response_model=ManifestListResponse)
async def list_project_manifests():
    """List all project manifests"""
    logger.info("Listing all project manifests")
    manifests = list_manifests()
    return {"manifests": manifests, "count": len(manifests)}

@app.post("/manifests", response_model=ManifestResponse, status_code=201)
async def create_project_manifest(
    request: ManifestRequest,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Create a new project manifest (superuser only)"""
    logger.info(f"Creating new project manifest: {request.manifest.project_id}")
    
    # Validate project_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', request.manifest.project_id):
        logger.error(f"Invalid project_id format: {request.manifest.project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    # Check if manifest already exists
    if load_manifest(request.manifest.project_id):
        logger.error(f"Manifest already exists: {request.manifest.project_id}")
        raise HTTPException(
            status_code=409, 
            detail=f"Manifest already exists: {request.manifest.project_id}"
        )
    
    # Validate module dependencies
    dependency_errors = validate_manifest_dependencies(request.manifest.modules)
    if dependency_errors:
        logger.error(f"Dependency validation failed for {request.manifest.project_id}: {dependency_errors}")
        raise HTTPException(status_code=400, detail="Dependency validation failed: " + "; ".join(dependency_errors))
    
    try:
        manifest_path = save_manifest(request.manifest)
        logger.info(f"Manifest created successfully: {request.manifest.project_id}")
        return {
            "message": f"Manifest created successfully: {request.manifest.project_id}",
            "manifest_id": request.manifest.project_id,
            "manifest_path": manifest_path
        }
    except Exception as e:
        logger.error(f"Failed to create manifest {request.manifest.project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create manifest: {str(e)}")

@app.get("/manifests/{project_id}")
async def get_project_manifest(
    project_id: str,
    resolve_env: bool = Query(False, description="Resolve environment variables and apply overrides")
):
    """Get a specific project manifest with optional environment variable resolution"""
    logger.info(f"Getting project manifest: {project_id}, resolve_env={resolve_env}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest = get_resolved_manifest(project_id, resolve_env)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    logger.info(f"Manifest retrieved successfully: {project_id}")
    return manifest

@app.put("/manifests/{project_id}", response_model=ManifestResponse)
async def update_project_manifest(
    project_id: str,
    request: ManifestRequest,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Update an existing project manifest (superuser only)"""
    logger.info(f"Updating project manifest: {project_id}")
    
    if project_id != request.manifest.project_id:
        logger.error(f"Project ID mismatch: path={project_id}, body={request.manifest.project_id}")
        raise HTTPException(
            status_code=400, 
            detail="project_id in path must match project_id in manifest"
        )
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    # Check if manifest exists
    existing_manifest = load_manifest(project_id)
    if not existing_manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    # Validate module dependencies
    dependency_errors = validate_manifest_dependencies(request.manifest.modules)
    if dependency_errors:
        logger.error(f"Dependency validation failed for {project_id}: {dependency_errors}")
        raise HTTPException(status_code=400, detail="Dependency validation failed: " + "; ".join(dependency_errors))
    
    # Preserve creation timestamp
    request.manifest.created_at = existing_manifest.created_at
    
    try:
        manifest_path = save_manifest(request.manifest)
        logger.info(f"Manifest updated successfully: {project_id}")
        return {
            "message": f"Manifest updated successfully: {project_id}",
            "manifest_id": project_id,
            "manifest_path": manifest_path
        }
    except Exception as e:
        logger.error(f"Failed to update manifest {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update manifest: {str(e)}")

@app.delete("/manifests/{project_id}")
async def delete_project_manifest(
    project_id: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Delete a project manifest (superuser only)"""
    logger.info(f"Deleting project manifest: {project_id}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest_path = get_manifest_path(project_id)
    
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    try:
        os.remove(manifest_path)
        logger.info(f"Manifest deleted successfully: {project_id}")
        return {"message": f"Manifest deleted successfully: {project_id}"}
    except Exception as e:
        logger.error(f"Failed to delete manifest {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete manifest: {str(e)}")

@app.post("/manifests/validate", response_model=ManifestValidationResponse)
async def validate_project_manifest(request: ManifestValidationRequest):
    """Validate a project manifest without saving it"""
    logger.info(f"Validating project manifest: {request.manifest.project_id}")
    
    errors = []
    warnings = []
    
    # Validate project_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', request.manifest.project_id):
        errors.append("Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens.")
    
    # Validate module dependencies
    dependency_errors = validate_manifest_dependencies(request.manifest.modules)
    errors.extend(dependency_errors)
    
    # Check for duplicate module names
    module_names = [module.name for module in request.manifest.modules]
    duplicate_names = [name for name in set(module_names) if module_names.count(name) > 1]
    if duplicate_names:
        errors.append(f"Duplicate module names found: {', '.join(duplicate_names)}")
    
    # Warning for disabled modules
    disabled_modules = [module.name for module in request.manifest.modules if module.status == ModuleStatus.DISABLED]
    if disabled_modules:
        warnings.append(f"Disabled modules found: {', '.join(disabled_modules)}")
    
    # Warning for deprecated modules
    deprecated_modules = [module.name for module in request.manifest.modules if module.status == ModuleStatus.DEPRECATED]
    if deprecated_modules:
        warnings.append(f"Deprecated modules found: {', '.join(deprecated_modules)}")
    
    logger.info(f"Manifest validation completed for {request.manifest.project_id}: valid={len(errors) == 0}, errors={len(errors)}, warnings={len(warnings)}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

@app.get("/manifests/{project_id}/modules")
async def get_project_modules(
    project_id: str,
    resolve_env: bool = Query(False, description="Resolve environment variables and apply overrides")
):
    """Get all modules for a specific project with optional environment variable resolution"""
    logger.info(f"Getting modules for project: {project_id}, resolve_env={resolve_env}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest = get_resolved_manifest(project_id, resolve_env)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    logger.info(f"Retrieved {len(manifest.modules)} modules for project: {project_id}")
    return {"modules": manifest.modules, "count": len(manifest.modules)}

@app.get("/manifests/{project_id}/modules/{module_name}")
async def get_project_module(
    project_id: str, 
    module_name: str,
    resolve_env: bool = Query(False, description="Resolve environment variables and apply overrides")
):
    """Get a specific module configuration from a project with optional environment variable resolution"""
    logger.info(f"Getting module {module_name} from project: {project_id}, resolve_env={resolve_env}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    module = get_resolved_module(project_id, module_name, resolve_env)
    if not module:
        # Check if project exists first
        manifest = load_manifest(project_id)
        if not manifest:
            logger.error(f"Manifest not found: {project_id}")
            raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
        else:
            logger.error(f"Module not found: {module_name} in project {project_id}")
            raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
    
    logger.info(f"Module retrieved successfully: {module_name} from project {project_id}")
    return module

@app.get("/manifests/{project_id}/cross-references")
async def get_project_cross_references(project_id: str):
    """Get cross-reference analysis for a project manifest"""
    logger.info(f"Getting cross-references for project: {project_id}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest = load_manifest(project_id)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    cross_ref_analysis = analyze_cross_references(manifest.modules)
    suggestions = get_cross_reference_suggestions(manifest.modules)
    
    logger.info(f"Cross-reference analysis completed for project: {project_id}")
    return {
        "project_id": project_id,
        "module_capabilities": cross_ref_analysis,
        "summary": {
            "total_modules": len(manifest.modules)
        }
    }

@app.get("/manifests/{project_id}/cross-references/suggestions")
async def get_cross_reference_suggestions_for_project(project_id: str):
    """Get cross-reference suggestions for a project"""
    logger.info(f"Getting cross-reference suggestions for project: {project_id}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest = load_manifest(project_id)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    suggestions = get_cross_reference_suggestions(manifest.modules)
    
    logger.info(f"Cross-reference suggestions retrieved for project: {project_id}")
    return {
        "project_id": project_id,
        "suggestions": suggestions,
        "summary": {
            "modules_with_suggestions": len(suggestions),
            "total_suggestions": sum(len(s) for s in suggestions.values())
        }
    }

@app.get("/manifests/{project_id}/modules/{module_name}/references")
async def get_module_references(
    project_id: str, 
    module_name: str
):
    """Get all cross-references for a specific module"""
    logger.info(f"Getting references for module {module_name} in project: {project_id}")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        logger.error(f"Invalid project_id format: {project_id}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid project_id format. Use only alphanumeric characters, underscores, and hyphens."
        )
    
    manifest = load_manifest(project_id)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    target_module = None
    for module in manifest.modules:
        if module.name == module_name:
            target_module = module
            break
    
    if not target_module:
        logger.error(f"Module not found: {module_name} in project {project_id}")
        raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
    
    cross_ref_analysis = analyze_cross_references(manifest.modules)
    module_analysis = cross_ref_analysis.get(module_name, {})
    
    logger.info(f"Module references retrieved for {module_name} in project {project_id}")
    return {
        "project_id": project_id,
        "module_name": module_name,
        "module_type": target_module.module_type,
        "provides_services": module_analysis.get("provides", []),
        "references": module_analysis.get("references", {}),
        "referenced_by": module_analysis.get("referenced_by", []),
        "cross_references_raw": target_module.cross_references
    }

@app.get("/module-types")
async def get_available_module_types():
    """Get all available module types and their descriptions"""
    logger.info("Getting available module types")
    return {
        "module_types": [
            {"type": ModuleType.JWT_CONFIG, "description": "JWT authentication and authorization configuration"},
            {"type": ModuleType.RAG_CONFIG, "description": "Retrieval Augmented Generation system configuration"},
            {"type": ModuleType.API_GATEWAY, "description": "API gateway and routing configuration"},
            {"type": ModuleType.INFERENCE_ENDPOINT, "description": "LLM inference endpoint configuration with prompts"},
            {"type": ModuleType.SECURITY, "description": "Security policies and compliance configuration"},
            {"type": ModuleType.MONITORING, "description": "Monitoring, logging, and observability configuration"},
            {"type": ModuleType.MODEL_REGISTRY, "description": "Model registry and versioning configuration"},
            {"type": ModuleType.DATA_PIPELINE, "description": "Data processing pipeline configuration"},
            {"type": ModuleType.DEPLOYMENT, "description": "Deployment strategy and environment configuration"},
            {"type": ModuleType.RESOURCE_MANAGEMENT, "description": "Resource allocation and scaling configuration"},
            {"type": ModuleType.NOTIFICATIONS, "description": "Notification and alerting configuration"},
            {"type": ModuleType.BACKUP_RECOVERY, "description": "Backup and disaster recovery configuration"},
            {"type": ModuleType.VAULT, "description": "HashiCorp Vault multi-instance secret management configuration"}
        ]
    }

# ==================== VAULT & SECRET MANAGEMENT ENDPOINTS ====================

@app.post("/vault/initialize")
async def initialize_vault_system(
    project_id: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Initialize Vault system for a project (superuser only)"""
    logger.info(f"Initializing Vault system for project: {project_id}")
    
    manifest = load_manifest(project_id)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    # Find Vault module
    vault_module = None
    for module in manifest.modules:
        if module.module_type == ModuleType.VAULT:
            vault_module = module
            break
    
    if not vault_module:
        logger.error(f"No Vault module configured in manifest: {project_id}")
        raise HTTPException(status_code=404, detail="No Vault module configured in manifest")
    
    # Initialize secret manager
    secret_manager = get_secret_manager(
        config_file_path=vault_module.config.get("secrets_config_file"),
        force_new=True
    )
    
    # Add all Vault instances
    initialized_instances = []
    for vault_config in vault_module.config.get("vault_instances", []):
        try:
            vault_dict = vault_config.dict() if hasattr(vault_config, 'dict') else vault_config
            instance_name = secret_manager.add_vault_instance_from_config(vault_dict)
            initialized_instances.append(instance_name)
            logger.info(f"Vault instance initialized: {instance_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vault instance '{vault_config.get('instance_name')}': {str(e)}")
            return {
                "success": False,
                "error": f"Failed to initialize Vault instance '{vault_config.get('instance_name')}': {str(e)}"
            }
    
    # Load external vault config if specified
    if vault_module.config.get("vault_config_file"):
        try:
            vault_config_path = vault_module.config["vault_config_file"]
            if os.path.exists(vault_config_path):
                with open(vault_config_path, 'r') as f:
                    external_config = json.load(f)
                
                for vault_config in external_config.get("vault_instances", []):
                    instance_name = secret_manager.add_vault_instance_from_config(vault_config)
                    initialized_instances.append(instance_name)
                    logger.info(f"External Vault instance initialized: {instance_name}")
        except Exception as e:
            logger.error(f"Failed to load external vault config: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to load external vault config: {str(e)}"
            }
    
    logger.info(f"Vault system initialized successfully for project {project_id}: {len(initialized_instances)} instances")
    return {
        "success": True,
        "project_id": project_id,
        "initialized_instances": initialized_instances,
        "total_instances": len(initialized_instances)
    }

@app.get("/vault/health")
async def check_vault_health(
    project_id: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Check health of all Vault instances for a project (superuser only)"""
    logger.info(f"Checking Vault health for project: {project_id}")
    
    manifest = load_manifest(project_id)
    if not manifest:
        logger.error(f"Manifest not found: {project_id}")
        raise HTTPException(status_code=404, detail=f"Manifest not found: {project_id}")
    
    # Initialize secret manager and Vault instances
    init_response = await initialize_vault_system(project_id, is_superuser=True)
    if not init_response.get("success"):
        logger.error(f"Failed to initialize Vault system: {init_response.get('error')}")
        raise HTTPException(status_code=500, detail=init_response.get("error"))
    
    secret_manager = get_secret_manager()
    health_results = secret_manager.vault_manager.health_check_all()
    
    all_healthy = all(v.get("healthy", False) for v in health_results.values())
    logger.info(f"Vault health check completed for project {project_id}: all_healthy={all_healthy}")
    
    return {
        "project_id": project_id,
        "vault_instances": health_results,
        "total_instances": len(health_results),
        "all_healthy": all_healthy
    }

@app.post("/vault/read-secret")
async def read_vault_secret(
    project_id: str,
    instance_name: str,
    secret_path: str,
    key: Optional[str] = None,
    version: Optional[int] = None,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Read a secret from a specific Vault instance (superuser only)"""
    logger.info(f"Reading Vault secret from {instance_name}: {secret_path}")
    
    # Initialize Vault system
    init_response = await initialize_vault_system(project_id, is_superuser=True)
    if not init_response.get("success"):
        logger.error(f"Failed to initialize Vault system: {init_response.get('error')}")
        raise HTTPException(status_code=500, detail=init_response.get("error"))
    
    secret_manager = get_secret_manager()
    
    try:
        secret_value = secret_manager.vault_manager.read_secret(
            instance_name=instance_name,
            path=secret_path,
            key=key,
            version=version
        )
        
        logger.info(f"Vault secret read successfully from {instance_name}: {secret_path}")
        return {
            "project_id": project_id,
            "instance_name": instance_name,
            "secret_path": secret_path,
            "key": key,
            "value": secret_value
        }
    except Exception as e:
        logger.error(f"Failed to read Vault secret from {instance_name}: {secret_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read secret: {str(e)}")

@app.post("/encryption/generate-key")
async def generate_encryption_key(
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Generate a new encryption key (superuser only)"""
    logger.info("Generating new encryption key")
    
    from encryption_utils import EncryptionManager
    key = EncryptionManager.generate_key()
    
    logger.info("Encryption key generated successfully")
    return {
        "encryption_key": key,
        "message": "Store this key securely as ENCRYPTION_KEY environment variable"
    }

@app.post("/encryption/encrypt")
async def encrypt_value(
    plaintext: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Encrypt a plaintext value (superuser only)"""
    logger.info("Encrypting value")
    
    try:
        encryption_manager = get_encryption_manager()
        encrypted = encryption_manager.encrypt(plaintext)
        
        logger.info("Value encrypted successfully")
        return {
            "plaintext": plaintext,
            "encrypted": encrypted
        }
    except ValueError as e:
        logger.error(f"Encryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/encryption/decrypt")
async def decrypt_value(
    ciphertext: str,
    is_superuser: bool = Depends(authenticate_superuser)
):
    """Decrypt an encrypted value (superuser only)"""
    logger.info("Decrypting value")
    
    try:
        encryption_manager = get_encryption_manager()
        decrypted = encryption_manager.decrypt(ciphertext)
        
        logger.info("Value decrypted successfully")
        return {
            "ciphertext": ciphertext,
            "decrypted": decrypted
        }
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DSP AI Control Tower')
    parser.add_argument('--host', default=os.getenv('HOST', '0.0.0.0'), help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8000')), help='Port to bind to')
    parser.add_argument('--https-port', type=int, default=int(os.getenv('HTTPS_PORT', '8443')), help='HTTPS port to bind to')
    parser.add_argument('--reload', action='store_true', default=os.getenv('RELOAD', 'true').lower() == 'true', help='Enable auto-reload')
    parser.add_argument('--ssl', action='store_true', default=os.getenv('SSL_ENABLED', 'false').lower() == 'true', help='Enable HTTPS')
    parser.add_argument('--ssl-cert', default=os.getenv('SSL_CERT_FILE', 'certs/server.crt'), help='SSL certificate file')
    parser.add_argument('--ssl-key', default=os.getenv('SSL_KEY_FILE', 'certs/server.key'), help='SSL key file')
    args = parser.parse_args()
    
    # Determine port and SSL settings
    if args.ssl:
        port = args.https_port
        ssl_keyfile = args.ssl_key
        ssl_certfile = args.ssl_cert
        
        # Verify SSL files exist
        if not os.path.exists(ssl_certfile):
            logger.error(f"SSL certificate not found: {ssl_certfile}")
            print(f"Error: SSL certificate not found: {ssl_certfile}")
            print("Run: python generate_ssl_certs.py")
            sys.exit(1)
        if not os.path.exists(ssl_keyfile):
            logger.error(f"SSL key not found: {ssl_keyfile}")
            print(f"Error: SSL key not found: {ssl_keyfile}")
            print("Run: python generate_ssl_certs.py")
            sys.exit(1)
        
        logger.info(f"Starting Control Tower with HTTPS on {args.host}:{port}")
        print(f"Starting Control Tower with HTTPS on {args.host}:{port}")
        print(f"  Certificate: {ssl_certfile}")
        print(f"  Key: {ssl_keyfile}")
        
        uvicorn.run(
            "app:app",
            host=args.host,
            port=port,
            reload=args.reload,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile
        )
    else:
        port = args.port
        logger.info(f"Starting Control Tower with HTTP on {args.host}:{port}")
        print(f"Starting Control Tower with HTTP on {args.host}:{port}")
        print("⚠ Warning: Running without HTTPS. Use --ssl for production.")
        
        uvicorn.run(
            "app:app",
            host=args.host,
            port=port,
            reload=args.reload
        )
