from openai import OpenAI
import time

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key-required"
)

def chat_completion():
    print("Sending request to OPT-125M model on CPU...")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain artificial intelligence in simple terms."}
        ],
        temperature=0.7,
        max_tokens=50  # Keep responses short for CPU
    )
    
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f} seconds")
    return response

if __name__ == "__main__":
    response = chat_completion()
    print("\nAssistant response:")
    print(response.choices[0].message.content)
