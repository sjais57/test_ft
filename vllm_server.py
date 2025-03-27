from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai import api_server

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the engine with your desired model
engine_args = AsyncEngineArgs(
    model="mistralai/Mistral-7B-Instruct-v0.1",  # Replace with your preferred model
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

# Add OpenAI-compatible endpoints
app = api_server.create_app(engine=engine)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
