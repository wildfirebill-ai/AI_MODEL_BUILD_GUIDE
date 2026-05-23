# MLX

**Version:** 0.22+

**Purpose:** Apple's machine learning framework optimized for Apple Silicon (M-series) via Metal GPU acceleration. MLX provides a NumPy-compatible array library (`mlx.core`), neural network layers (`mlx.nn`), and optimizers (`mlx.optimizers`) with a unified memory architecture — arrays live on the same memory as the CPU, eliminating PCIe transfers. In the WFB model pipeline, MLX enables on-device inference, fast prototyping on MacBooks, and efficient local evaluation during development.

## Installation

```powershell
pip install "mlx==0.22.0" "mlx-lm==0.22.0"
```

## Basic Usage

### Array Operations

```python
import mlx.core as mx

x = mx.random.normal(shape=(4, 1024))
w = mx.random.normal(shape=(1024, 4096))
y = x @ w  # Metal-accelerated matmul
print(y.shape)  # (4, 4096)
```

### Neural Network

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dim, n_heads)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def __call__(self, x):
        x = self.attention(x, x, x)
        return self.linear2(nn.relu(self.linear1(x)))

model = TransformerBlock(512, 8)
mx.eval(model.parameters())
```

### Model Inference with mlx-lm

```python
from mlx_lm import load, generate

model_path = "./checkpoints/final"
tokenizer_path = "./tokenizer"
model, tokenizer = load(model_path, tokenizer=tokenizer_path)

prompt = "Explain self-attention:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

## Advanced Usage

### LoRA Fine-Tuning on Mac

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import lora

model, tokenizer = load("./checkpoints/final")
lora_config = lora.LoraConfig(
    lora_layers=16,
    lora_rank=8,
    lora_scale=16.0,
)
lora_model = lora.LoraModel(model, lora_config)

optimizer = optim.Adam(learning_rate=1e-5)
loss_fn = nn.losses.cross_entropy
```

### Custom Training Loop

```python
def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

optimizer = optim.Adam(learning_rate=1e-4)
state = [model.state(), optimizer.state()]

def step(x, y):
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(state)
    return loss
```

## Integration with WFB Model

MLX is used during **Part 2** (Environment & Tooling) and **Part 7** (Optimization & Quantization) of the WFB pipeline. On Apple Silicon Macs, MLX serves as the primary development environment for prototyping architecture changes, testing tokenizers, and running small-scale ablations before moving to GPU clusters. After training on NVIDIA hardware, MLX can load the exported model for local testing. During quantization, MLX's FP16/FP32 unified memory model enables rapid iteration on calibration datasets. For team members without GPU access, MLX provides a viable local development path.

## Common Pitfalls

- **Apple Silicon only**: MLX does not run on Intel Macs or non-Apple hardware. Check `platform.processor()`.
- **Metal memory limit**: Unified memory is shared with the system. Monitor with `mx.metal.get_active_memory()`. Large models (>13B) may cause swap thrashing.
- **No CUDA interop**: MLX and PyTorch models cannot share weights in memory. Export via safetensors.
- **Graph compilation**: `mx.compile()` is experimental. For reliable performance, rely on eager-mode Metal acceleration.
- **Tokenizer mismatch**: mlx-lm expects `tokenizer.json`. Convert HF tokenizers with `mlx_lm.tokenizer_utils`.

## Documentation

- https://ml-explore.github.io/mlx/
- https://github.com/ml-explore/mlx
- https://github.com/ml-explore/mlx-examples
