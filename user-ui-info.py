import json

@app.post("/user_ui_info")
async def user_ui_info(request: UserPoliciesRequest):
    client_dir = "policies/clients"
    policies_ui = []

    # Load the deployment models JSON file
    deployment_models_file = "deployment_models.json"  # Adjust path as needed
    try:
        with open(deployment_models_file, 'r') as f:
            deployment_data = json.load(f)
    except FileNotFoundError:
        return {"error": f"Deployment models file {deployment_models_file} not found"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in {deployment_models_file}"}

    if not os.path.exists(client_dir):
        return {"policies": []}

    for file in os.listdir(client_dir):
        if file.endswith(".rego") and not file.endswith("_test.rego"):
            policy_path = os.path.join(client_dir, file).replace("\\", "/")
            with open(policy_path, "r") as f:
                policy_content = f.read()

            enabled_match = re.search(r"policy_enabled\s*:\s*(true|false)", policy_content)
            if enabled_match and enabled_match.group(1) == "false":
                continue

            policy_name = os.path.basename(policy_path)
            project_name = policy_name

            # Extract project details
            project_details = {}
            project_details_fields = {
                "project_owner": r'project_owner\s*=\s*"([^"]+)"',
                "project_code": r'project_code\s*=\s*"([^"]+)"',
                "project_label": r'project_label\s*=\s*"([^"]+)"',
                "project_team": r'project_team\s*=\s*"([^"]+)"',
            }

            for key, pattern in project_details_fields.items():
                match = re.search(pattern, policy_content)
                if match:
                    project_details[key] = match.group(1)
                else:
                    project_details[key] = ""

            # Get deployment models for this project from JSON
            deploy_models = {}
            jl_models_set = set()
            
            # Find project in deployment data (case-insensitive match)
            project_key = None
            for key in deployment_data.get("project", {}).keys():
                if key.lower() == project_name.lower():
                    project_key = key
                    break
            
            if project_key and project_key in deployment_data.get("project", {}):
                deployment_model_list = deployment_data["project"][project_key]
                
                # Search for each model in model_yaml_data
                for model_unique_name in deployment_model_list:
                    # Search for the model in model_yaml_data by unique_model_name
                    matching_configs = find_model_configs_by_unique_name(model_yaml_data, model_unique_name)
                    
                    for cfg in matching_configs:
                        hpc_model_name = cfg.get("hpc_model_name", "")
                        model_name = cfg.get("unique_model_name", "")
                        if not model_name:
                            continue

                        jl_models_set.add(hpc_model_name)
                        if model_name not in deploy_models:
                            deploy_models[model_name] = []
                        deploy_models[model_name].append({
                            "inference_type": cfg.get("inference_type", ""),
                            "preselected_gpu": str(cfg.get("task_gpu", ""))
                        })

            jl_models = list(jl_models_set)
            roles = list(set(actions))

            # Lane access logic with env-level GPU (unchanged)
            lane_map = {
                "train_pvt": "Training (Non-Public Data)",
                "train_pub": "Training (Public Data)",
                "deploy_prod": "Inference (Prod)",
                "deploy_dev": "Inference (Dev)",
            }

            lane_access = []
            for action_key, label in lane_map.items():
                gpu_dict = {}
                for env in ["dev", "prod"]:
                    try:
                        aihpc_config = extract_aihpc_config(policy_content, env, label)
                        account = aihpc_config["account"]
                        gpu_dict[env] = aihpc_config.get("gpu", "0")
                    except Exception:
                        gpu_dict[env] = "0"
                lane_access.append({
                    "label": label,
                    "value": action_key,
                    "account": account,
                    "gpu": gpu_dict
                })

            policy_ui = {
                "project_name": project_name,
                "project_details": project_details,
                "lane_access": lane_access,
                "role": roles,
                "jl_models": jl_models,
                "deploy_models": deploy_models,
            }

            policies_ui.append(policy_ui)

    return {"policies": policies_ui}


def find_model_configs_by_unique_name(model_yaml_data, unique_model_name):
    """
    Search for model configurations by unique_model_name in model_yaml_data.
    Returns a list of matching configs.
    """
    matching_configs = []
    
    def search_recursive(data):
        if isinstance(data, dict):
            # Check if this is a model config with the matching unique_model_name
            if data.get("unique_model_name") == unique_model_name:
                matching_configs.append(data)
            # Continue searching in nested structures
            for value in data.values():
                search_recursive(value)
        elif isinstance(data, list):
            for item in data:
                search_recursive(item)
    
    search_recursive(model_yaml_data)
    return matching_configs
