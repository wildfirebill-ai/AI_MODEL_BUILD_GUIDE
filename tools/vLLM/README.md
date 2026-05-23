# vLLM

**Version:** 0.6+

**Purpose:** High-throughput, low-latency LLM inference server. Supports continuous batching, PagedAttention, and large context windows.

## Installation

```powershell
pip install vllm
```

## Usage

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./checkpoints/final",
    tokenizer="./tokenizer.json",
    tensor_parallel_size=1,   # Number of GPUs
    max_model_len=200000,     # 200K context
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=4096,
)

outputs = llm.generate(["Explain transformer attention"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible Server

```powershell
python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/final \
    --tokenizer ./tokenizer.json \
    --max-model-len 200000 \
    --tensor-parallel-size 1 \
    --port 8000
```

Then use like OpenAI API:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.completions.create(model="default", prompt="Hello", max_tokens=100)
```

## Key Features

- **PagedAttention** — efficient memory management for KV cache
- **Continuous batching** — dynamic request batching
- **Prefix caching** — reuse common prefixes (system prompts)
- **Quantization** — FP16, GPTQ, AWQ, FP8
- **Streaming** — token-by-token output

## Documentation

- https://docs.vllm.ai/
