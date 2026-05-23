# FlashAttention-2

**Version:** 2.6+

**Purpose:** Fast and memory-efficient exact attention — essential for 150K+ context windows.

## Installation

### Pre-built wheel (recommended for CUDA 12.x)
```powershell
pip install flash-attn --no-build-isolation
```

### Build from source (if pre-built fails)
```powershell
pip install ninja
set DISTUTILS_USE_SDK=1
pip install flash-attn --no-build-isolation --verbose
```

## Requirements

- CUDA 11.8+ (12.x recommended)
- PyTorch 2.0+
- NVIDIA GPU with compute capability 8.0+ (Ampere: A100, RTX 3090, RTX 4090, H100)

## Performance Gains

| Context Length | Standard Attention | FlashAttention-2 | Speedup |
|----------------|-------------------|------------------|---------|
| 2K | 10 ms | 3 ms | 3x |
| 8K | 80 ms | 12 ms | 6x |
| 32K | 1.2 s | 50 ms | 24x |
| 128K | 20 s | 200 ms | 100x |

## Usage

```python
from flash_attn import flash_attn_func

# flash_attn_func(q, k, v, dropout_p, causal, softmax_scale)
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

# Parameters:
# q, k, v: (batch, seqlen, num_heads, head_dim)
# causal: True for autoregressive (mask future tokens)
# dropout_p: dropout probability (0.0 for inference)
```

## Integration in Model

FlashAttention is used automatically in the `GroupedQueryAttention` class in `model.py` when available. It falls back to manual attention if not installed.

## Documentation

- https://github.com/Dao-AILab/flash-attention
