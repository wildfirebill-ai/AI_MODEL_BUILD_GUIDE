# DeepSpeed

**Version:** 0.14+

**Purpose:** Microsoft's deep learning optimization library for training large models (ZeRO optimization, offloading, pipeline parallelism).

## Installation

```powershell
pip install deepspeed
```

## ZeRO Optimization Stages

| Stage | Description | Memory Savings | Use Case |
|-------|-------------|---------------|----------|
| ZeRO-1 | Partition optimizer states | 2x | Small models, few GPUs |
| ZeRO-2 | Partition optimizer + gradients | 4x | Medium models (1B-7B) |
| ZeRO-3 | Partition optimizer + gradients + params | 8x+ | Large models (7B+) |

## Offloading

Offload to CPU when GPU memory is insufficient:
- `offload_optimizer.device: "cpu"` — optimizer states to CPU
- `offload_param.device: "cpu"` — model parameters to CPU

## Configuration (ds_config.json)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true },
    "overlap_comm": true
  },
  "bf16": { "enabled": true },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 4
}
```

## Run

```powershell
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

## Key Features

- **ZeRO** — shard optimizer, gradients, parameters across GPUs
- **CPU/NVMe Offloading** — train large models on limited GPU memory
- **Gradient checkpointing** — trade compute for memory
- **Pipeline parallelism** — split layers across GPUs
- **Sparse attention** — efficient long sequences

## Documentation

- https://www.deepspeed.ai/
- https://deepspeed.readthedocs.io/
