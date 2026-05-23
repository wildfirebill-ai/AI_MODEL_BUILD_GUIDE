# FSDP (Fully Sharded Data Parallel)

**Version:** Built into PyTorch 2.x

**Purpose:** Shard model parameters, gradients, and optimizer states across GPUs (ZeRO-3 equivalent) for training large models.

## Installation

Built into PyTorch -- no separate install needed.

```powershell
# Ensure PyTorch is installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
import torch

# FSDP configuration
fsdp_config = dict(
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # ZeRO-3 within node, ZeRO-1 across
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    cpu_offload=CPUOffload(offload_params=False),
    device_id=torch.cuda.current_device(),
)

# Wrap model
model = FSDP(model, **fsdp_config)

# Training (same as usual)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Sharding Strategies

| Strategy | Description | Memory Savings |
|----------|-------------|---------------|
| `NO_SHARD` | DDP (no sharding) | 1x |
| `SHARD_GRAD_OP` | ZeRO-2 (shard grads + optim) | 2-4x |
| `FULL_SHARD` | ZeRO-3 (shard params + grads + optim) | 4-8x |
| `HYBRID_SHARD` | FULL_SHARD within node, SHARD_GRAD_OP across | 2-4x (lower communication) |

## When to Use

| Model Size | GPUs per Node | Strategy |
|-----------|---------------|----------|
| <1B | 1-8 | DDP (NO_SHARD) |
| 1B-7B | 4-8 | FULL_SHARD |
| 7B-70B | 8 | HYBRID_SHARD |
| 70B+ | 8+ nodes | HYBRID_SHARD + Tensor Parallel |

## Documentation

- https://pytorch.org/docs/stable/fsdp.html
