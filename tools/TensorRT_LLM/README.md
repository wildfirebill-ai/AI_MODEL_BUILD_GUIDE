# TensorRT-LLM

**Version:** 0.10+

**Purpose:** NVIDIA's optimized inference engine for LLMs. Maximum performance on A100/H100/B200 GPUs.

## Installation

```powershell
# Requires TensorRT installed first
pip install tensorrt-llm
```

## Usage

```python
# Build engine from checkpoint
from tensorrt_llm.builder import build
from tensorrt_llm.models import LLaMAForCausalLM

# Build TensorRT engine
engine = build(
    model=LLaMAForCausalLM.from_checkpoint("./checkpoints/final"),
    batch_size=8,
    max_input_len=8192,
    max_output_len=2048,
    use_fp8=True,         # FP8 quantization (H100/Blackwell)
    use_fused_mlp=True,   # Fused MLP kernels
    world_size=4,          # Tensor parallelism
)

engine.save("model.engine")

# Inference
from tensorrt_llm.runtime import GenerationSession
session = GenerationSession(engine)
output = session.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    temperature=0.7,
)
```

## Performance

| GPU | FP16 (ms/token) | FP8 (ms/token) | Speedup |
|-----|----------------|----------------|---------|
| A100 80GB | 12 ms | - | 1x |
| H100 80GB | 8 ms | 4 ms | 2-3x |
| B200 | 5 ms | 2.5 ms | 4-5x |

## Documentation

- https://github.com/NVIDIA/TensorRT-LLM
