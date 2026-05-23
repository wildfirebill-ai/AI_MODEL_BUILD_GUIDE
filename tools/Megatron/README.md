# Megatron-LM / NeMo

**Purpose:** NVIDIA's framework for large-scale distributed training with tensor, pipeline, and data parallelism.

## Installation

```powershell
pip install nemo-toolkit
# Or clone Megatron-LM:
git clone https://github.com/NVIDIA/Megatron-LM
```

## Key Features

- **3D Parallelism**: Tensor + Pipeline + Data Parallelism
- **Sequence Parallelism**: Split sequence along GPUs
- **Distributed Optimizer**: ZeRO-style optimizer state sharding
- **BF16/FP8 Training**: Native mixed precision
- **Activation Recomputation**: Gradient checkpointing
- **Flash Attention**: Integrated with TE (Transformer Engine)

## Usage (simplified)

```python
from megatron.core import parallel_state
from megatron.training import pretrain

# Config:
# --tensor-model-parallel-size 4
# --pipeline-model-parallel-size 4
# --sequence-parallel
# --use-flash-attn
# --bf16
# --lr 1e-4

pretrain(model_provider, train_dataset, model_parallel_size=16)
```

## Best For

- **70B-1T+ parameter models**
- **Multi-node DGX clusters** (A100/H100/B200)
- **Production training** at scale

## Documentation

- https://github.com/NVIDIA/Megatron-LM
- https://docs.nvidia.com/deeplearning/nemo/
