# TGI (Text Generation Inference)

**Version:** 3.0+

**Purpose:** HuggingFace's optimized inference server for large language models. TGI provides continuous batching, tensor parallelism, Flash Attention, quantization (AWQ, GPTQ, FP8), and OpenAI-compatible API endpoints. It is designed as a production-ready alternative to vLLM with tight HuggingFace ecosystem integration. In the WFB model pipeline, TGI serves as the primary inference backend for deployed models.

## Installation

```powershell
docker pull ghcr.io/huggingface/text-generation-inference:3.0.0
```

## Basic Usage

### Docker Deployment

```powershell
docker run --gpus all -p 8080:80 `
    -v G:\zed\wfb_model\checkpoints\final:/model `
    ghcr.io/huggingface/text-generation-inference:3.0.0 `
    --model-id /model `
    --max-total-tokens 131072 `
    --max-batch-prefill-tokens 32768 `
    --num-shard 1
```

### Client Request

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "Explain attention mechanisms"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Advanced Usage

### Quantized Deployment

```powershell
docker run --gpus all -p 8080:80 `
    ghcr.io/huggingface/text-generation-inference:3.0.0 `
    --model-id /model `
    --quantize awq `
    --dtype float16
```

### Custom Tokenizer and Streaming

```python
import requests

response = requests.post(
    "http://localhost:8080/generate_stream",
    json={
        "inputs": "The future of AI is",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "stop": ["."],
        },
    },
    stream=True,
)
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

### Configuration via Environment Variables

```yaml
# docker-compose.yml
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:3.0.0
    ports:
      - "8080:80"
    volumes:
      - ./checkpoints/final:/model
    environment:
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - MAX_INPUT_LENGTH=8192
      - MAX_TOTAL_TOKENS=131072
    command: --model-id /model --num-shard 2 --max-batch-total-tokens 4194304
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

## Integration with WFB Model

TGI is the production inference option for **Part 8** (Production & Deployment) of the WFB pipeline. After quantization with AWQ or GPTQ (Part 7), deploy the quantized model via TGI for low-latency serving. TGI's native HuggingFace integration makes it the preferred choice when the model uses custom `transformers` components (e.g., custom attention, MoE routing). For models exported to standard architectures, vLLM offers higher throughput; TGI excels when compatibility with the HuggingFace generation pipeline is critical.

## Common Pitfalls

- **Docker GPU passthrough**: Requires `nvidia-container-toolkit`. Run `docker run --gpus all` to verify.
- **Shared memory**: TGI uses NCCL. Add `--shm-size 2g` to docker args for multi-shard.
- **Model ID resolution**: When using local paths, ensure all model files (safetensors, config.json, tokenizer.json) are present. Missing tokenizer causes silent fallback to GPT-2 tokenizer.
- **OOM on long sequences**: Reduce `--max-batch-prefill-tokens` or `--max-total-tokens`.
- **Quantization mismatch**: AWQ/GPTQ quantized models require matching `--quantize` flag. Wrong flag yields garbage output.

## Documentation

- https://huggingface.co/docs/text-generation-inference
- https://github.com/huggingface/text-generation-inference
