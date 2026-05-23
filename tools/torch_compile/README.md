# torch.compile (PyTorch 2.0+)

**Version:** Built into PyTorch 2.x

**Purpose:** JIT-compile PyTorch models for 10-30% training speedup and 10-50% inference speedup.

## Usage

```python
import torch

# Basic usage
model = TransformerModel(config)
model = torch.compile(model)

# With options
model = torch.compile(
    model,
    backend="inductor",          # or "cudagraphs", "triton"
    mode="reduce-overhead",      # or "max-autotune", "default"
    fullgraph=False,             # Set True if no graph breaks
    dynamic=True,                # For dynamic shapes
)

# Verify
print(model)  # Should show OptimizedModule

# Training loop stays identical
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## Modes

| Mode | Speedup | Compile Time | Best For |
|------|---------|--------------|----------|
| `default` | 1.1-1.3x | Fast | General |
| `reduce-overhead` | 1.2-1.5x | Medium | Training with fixed shapes |
| `max-autotune` | 1.3-2x | Slow | Production inference |

## Backends

| Backend | GPU | CPU | Notes |
|---------|-----|-----|-------|
| `inductor` | Yes | Yes | Default, uses Triton |
| `cudagraphs` | Yes | No | Static shapes only |
| `triton` | Yes | No | Custom Triton kernels |

## Troubleshooting

```python
# Check if compilation succeeded
print(f"Compiled: {isinstance(model, torch._dynamo.eval_frame.OptimizedModule)}")

# Export compiled graph
torch._dynamo.config.output_code = True

# Disable for specific functions
@torch.compiler.disable
def preprocessing(x):
    return x.normalize()
```

## Documentation

- https://pytorch.org/docs/stable/generated/torch.compile.html
