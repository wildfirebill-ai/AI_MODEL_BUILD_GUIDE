# Mamba (State Space Models)

**Version:** 2.0+

**Purpose:** Linear-time sequence modeling as an alternative to Transformers. Mamba uses selective state space models (SSMs) to achieve O(L) inference complexity instead of O(L²) attention, enabling unlimited context length with constant memory growth. Mamba-2 improves throughput with the SSD (State Space Dual) formulation. The WFB model may use Mamba blocks for long-context layers.

## Installation

```powershell
pip install mamba-ssm causal-conv1d
```

**Requirements:**
- CUDA 12.1+ (required for the fused CUDA kernels)
- PyTorch 2.1+
- `ninja` (build system — install with `pip install ninja` first)
- Linux recommended — Windows support is experimental

### Build from source (if pip fails)
```powershell
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
```

## Basic Usage

```python
import torch
from mamba_ssm import Mamba

batch = 2
seq_len = 2048
dim = 4096

x = torch.randn(batch, seq_len, dim, device="cuda")

model = Mamba(
    d_model=dim,
    d_state=16,      # SSM state dimension
    d_conv=4,        # Local convolution width
    expand_factor=2, # Expansion factor for inner dimension
).to("cuda")

output = model(x)  # (batch, seq_len, dim) — same shape, linear in seq_len
```

## Advanced Usage / Configuration

### Mamba-2 (SSD)
```python
from mamba_ssm import Mamba2

model = Mamba2(
    d_model=dim,
    d_state=128,     # Mamba-2 supports larger state
    d_conv=4,
    expand_factor=2,
    headdim=64,       # Head dimension for SSD
)
```

### Hybrid Mamba + Attention
```python
# Many performant models interleave Mamba blocks with attention layers
# Example alternating pattern:
layers = [
    MambaBlock(d_model=dim),      # Layer 0: Mamba
    AttentionBlock(d_model=dim),  # Layer 1: Attention
    MambaBlock(d_model=dim),      # Layer 2: Mamba
    # ...
]
```

### Loading pretrained models
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-slimpj",
    trust_remote_code=True,
    device="cuda",
)
```

## Integration with WFB Model
Mamba blocks are integrated in the WFB model as an optional long-context layer type (configurable in `configs/model/mamba_layers.yaml`). The model supports hybrid architectures that use Mamba for lower layers and attention for upper layers, balancing efficiency with recall:

```python
# wfb_model/models/hybrid_mamba.py
# Alternate Mamba and attention layers for efficient long-context processing
```

## Common Pitfalls / Troubleshooting
- **`CUDA error: no kernel image available`:** Mamba kernels are compiled for specific GPU architectures — set `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` before building for Ampere/Ada Lovelace/Blackwell support
- **Build fails on Windows:** Mamba has limited Windows support — use WSL2 or Linux; the CUDA kernels use `__align__` and `__half2` intrinsics that may not compile on MSVC
- **Model outputs NaNs:** Mamba has known numerical stability issues at high learning rates — reduce LR or increase `dt_min` and `dt_max` parameters
- **Slow training compared to attention:** Mamba is faster at inference but may be slower to train due to the sequential SSM recurrence — use `Mamba2` for better training throughput
- **Memory grows with sequence length for training:** The SSM stores intermediate states for backpropagation — training memory still grows with sequence length, just not quadratically

## Documentation
- https://github.com/state-spaces/mamba
- https://arxiv.org/abs/2312.00752 (Mamba paper)
- https://arxiv.org/abs/2405.21060 (Mamba-2 paper)
