# llama.cpp

**Purpose:** C/C++ implementation of LLM inference optimized for CPU and Apple Silicon. Supports GGUF quantization format.

## Installation

```powershell
pip install llama-cpp-python
```

## Usage

```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    model_path="./model.gguf",
    # Context window size (tokens)
    n_ctx=8192,
    # Number of GPU layers (-1 = all)
    n_gpu_layers=-1,
    # Batch size for prompt processing
    n_batch=512,
    # Thread count for CPU
    n_threads=8,
    # Use Flash Attention
    use_flash_attn=True,
    # Memory settings
    verbose=True,
)

# Generate
output = llm(
    "Hello, how are you?",
    # Maximum tokens to generate
    max_tokens=256,
    # Sampling temperature
    temperature=0.7,
    # Top-p nucleus sampling
    top_p=0.9,
    # Echo the prompt in output
    echo=False,
)

print(output["choices"][0]["text"])

# Chat format
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ],
)
```

## GGUF Quantization Levels

| Type | Bits | Size (7B) | Quality |
|------|------|-----------|---------|
| Q2_K | 2-bit | 2.8 GB | Poor |
| Q3_K_M | 3-bit | 3.3 GB | Low |
| Q4_K_M | 4-bit | 4.1 GB | Good (recommended) |
| Q5_K_M | 5-bit | 4.8 GB | Very good |
| Q6_K | 6-bit | 5.4 GB | Excellent |
| Q8_0 | 8-bit | 6.7 GB | Near lossless |
| F16 | 16-bit | 13 GB | Full precision |

## Documentation

- https://github.com/ggml-org/llama.cpp
