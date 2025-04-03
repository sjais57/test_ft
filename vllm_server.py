from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel
import uvicorn
import time
from typing import List, Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 50  # Reduced for CPU
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

# Initialize the engine with CPU-specific settings
engine_args = AsyncEngineArgs(
    model="facebook/opt-125m",  # Smaller model
    tensor_parallel_size=1,
    gpu_memory_utilization=0,  # Not using GPU
    disable_custom_all_reduce=True,  # CPU optimization
    enforce_eager=True,  # Better for CPU
    max_model_len=512,  # Reduced for CPU
    device="cpu",  # Force CPU usage
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Generate completion for the request following OpenAI's format."""
    # Simple prompt formatting for OPT model
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    prompt += "\nassistant:"
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )
    
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        return {"error": "No output generated"}
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_output.outputs[0].text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # Single worker for CPU
