from openai import OpenAI
from vllm.transformers_utils.tokenizer import get_tokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Initialize vLLM's tokenizer for accurate Qwen2.5 token counting
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
try:
    vllm_tokenizer = get_tokenizer(model_name)
    print(f"Using vLLM tokenizer for {model_name}")
except Exception as e:
    print(f"Failed to load vLLM tokenizer: {e}")
    vllm_tokenizer = None

def count_tokens_via_vllm(text):
    """Count tokens using vLLM's native tokenizer for Qwen2.5"""
    if vllm_tokenizer:
        try:
            tokens = vllm_tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"vLLM tokenization failed: {e}")
    return None

def count_tokens(text):
    """Count tokens in the given text using the most accurate method available"""
    # First priority: vLLM's native tokenizer (most accurate for Qwen2.5)
    vllm_count = count_tokens_via_vllm(text)
    if vllm_count is not None:
        return vllm_count
    else:
        exit("Failed to count tokens")

# Test tokenizer with a simple string to verify it's working
test_text = "Hello, world!"
test_tokens = count_tokens(test_text)
print(f"Tokenizer test: '{test_text}' -> {test_tokens} tokens")

# Initialize conversation with system message
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# First round
print("=== Round 1 ===")
messages.append({"role": "user", "content": "this is a one day project!"})

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=messages,
    max_completion_tokens=50,
    extra_body={
        "ignore_eos": True,
        "job_id": "123",
        "last_func_call": "mv",
        "is_last_step": False,
        "this_func_call": "mv",
    }
)

print(chat_response.choices[0].message.content)
print(count_tokens(chat_response.choices[0].message.content))