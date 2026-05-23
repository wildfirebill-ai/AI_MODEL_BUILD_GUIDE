# BentoML

**Version:** 1.4+

**Purpose:** Model serving and deployment framework that packages ML models into production-ready services. BentoML provides a unified API for defining services, running inference on GPUs with runners, building deployable artifacts (Bentos), and containerizing for Kubernetes. It includes native integrations for PyTorch, Transformers, and vLLM. In the WFB pipeline, BentoML wraps quantized models behind REST endpoints with auto-scaling and canary deployments.

## Installation

```powershell
pip install "bentoml==1.4.0" "torch" "transformers"
```

## Basic Usage

### Define and Run a Service

```python
import bentoml
import torch

@bentoml.service(resources={"gpu": 1}, traffic={"timeout": 60})
class WFBModelService:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda:0"
        self.model = AutoModelForCausalLM.from_pretrained(
            "./checkpoints/final", device_map=self.device, torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/final")

    @bentoml.api
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Client:
import requests
r = requests.post("http://localhost:3000/generate", json={"prompt": "BentoML deploys"})
print(r.json())
```

Run with: `bentoml serve service.py:svc --port 3000 --reload`

## Advanced Usage

### vLLM Integration with Batching

```python
from vllm import LLM, SamplingParams

@bentoml.service(resources={"gpu": 1}, traffic={"concurrency": 8})
class VLLMService:
    def __init__(self):
        self.llm = LLM(model="./checkpoints/awq", quantization="AWQ")

    @bentoml.api(batchable=True, max_batch_size=32)
    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=256))
        return [o.outputs[0].text for o in outputs]
```

### Containerization

```yaml
# bentofile.yaml
service: "service.py:svc"
name: "wfb-model-service"
version: "1.0.0"
python:
  packages:
    - torch==2.5.0
    - transformers==4.47.0
    - bentoml==1.4.0
docker:
  cuda_version: "12.4"
```

```powershell
bentoml build
bentoml containerize wfb-model-service:1.0.0 -t registry/wfb-model:latest
```

### Adaptive Batching

```python
@bentoml.api(batchable=True, batch_dim=0, max_batch_size=64, max_latency_ms=100)
async def generate(self, prompts: list[str]) -> list[str]:
    return [o.outputs[0].text for o in self.llm.generate(prompts, SamplingParams(temperature=0.7))]
```

## Integration with WFB Model

BentoML is used in **Part 8** (Production & Deployment) as the final serving layer. After quantization (Part 7), BentoML packages the model into a reproducible Bento artifact with pinned dependencies. The Bento is containerized into a Docker image and deployed to Kubernetes with auto-scaling, canary rollouts, and Prometheus monitoring. Adaptive batching maximizes GPU utilization when routing to vLLM or TGI backends.

## Common Pitfalls

- **Model loading on import**: Initialize in `__init__`, not at module level.
- **GPU allocation**: `resources={"gpu": 1}` allocates one GPU. Configure model parallelism for multi-GPU.
- **Bentofile paths**: Relative to bentofile location. Use absolute paths for external model dirs.
- **Secrets**: Use environment variables. BentoML supports `.env` with `envfile: .env`.
- **Graceful shutdown**: Long generations are interrupted during scaling. Set `traffic={"timeout": 300}`.

## Documentation

- https://docs.bentoml.com/
- https://github.com/bentoml/BentoML
