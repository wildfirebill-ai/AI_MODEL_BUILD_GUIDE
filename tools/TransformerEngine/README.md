# Transformer Engine (NVIDIA)

**Version:** 1.6+

**Purpose:** NVIDIA's library for FP8 training and high-performance transformer layers on Hopper (H100) and Blackwell (B100/B200) GPUs.

## Installation

```powershell
pip install transformer-engine
```

## Usage

```python
import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_autocast

# FP8 transformer layer (replaces standard nn.Linear)
class FP8TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = te.LayerNormLinear(hidden_size, hidden_size, bias=False)
        self.mlp = te.LayerNormMLP(hidden_size, hidden_size * 4, bias=False)

    def forward(self, x):
        with fp8_autocast(enabled=True):
            x = self.attention(x)
            x = self.mlp(x)
        return x

# FP8 training loop
with fp8_autocast(enabled=True):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))

loss.backward()
```

## Key Features

- **FP8 training** (E4M3, E5M2) -- 2x faster, 2x less memory
- **MXFP4 (Blackwell)** -- 4-bit block-sparse training
- **Fused kernels** -- LayerNorm + Linear, QKV projection
- **Delayed scaling** -- automatic scaling factor management

## Hardware Support

| GPU | FP8 | MXFP4 | TF32 |
|-----|-----|-------|------|
| H100 (Hopper) | ok | No | ok |
| B100 (Blackwell) | ++ | ok | ok |
| B200 (Blackwell) | ++ | ok | ok |

## Documentation

- https://github.com/NVIDIA/TransformerEngine
