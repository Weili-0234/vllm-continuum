# Explanation
This repo contains the minimal version of the paper Continuum based on vLLM v0.10.2 for simplicity of use. It is currently under update.
Next update:
Example script running mini-swe-agent + SWE-Bench

## Installation

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode
uv pip install .

# Install dependencies
uv pip install lmcache
uv pip install hf_transfer
uv pip install -U "huggingface_hub[cli]"

# Log in to Hugging Face
hf auth login
# Use your Hugging Face access token when prompted
```

## Starting the Server

### Original vLLM

```bash
# Without CPU offload
vllm serve <your_model_here>

# With CPU offload
LMCACHE_MAX_LOCAL_CPU_SIZE=<cpu_size> \
vllm serve <your_model_here> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Continuum Scheduling

```bash
# Without CPU offload
vllm serve <your_model_here> --scheduling_policy=continuum

# With CPU offload
LMCACHE_MAX_LOCAL_CPU_SIZE=<cpu_size> \
vllm serve <your_model_here> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --scheduling_policy=continuum
```
