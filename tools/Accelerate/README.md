# HuggingFace Accelerate

**Version:** 0.33+

**Purpose:** Simplifies running PyTorch training across GPUs, TPUs, and mixed precision without boilerplate.

## Installation

```powershell
pip install accelerate
```

## Setup

```powershell
# Interactive config (answers saved to ~/.cache/huggingface/accelerate/default_config.yaml)
accelerate config

# Or create config manually:
accelerate config --config_file accelerate_config.yaml
```

## Key Features

- **Mixed precision** (fp16/bf16) — 2x speed, half memory
- **Multi-GPU** (DDP, FSDP, DeepSpeed)
- **Gradient accumulation** — simulate larger batch sizes
- **Device placement** — automatic `.to(device)`
- **Model offloading** — CPU offload for large models

## Usage

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
)

model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

with accelerator.accumulate(model):
    loss = model(batch)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## Run

```powershell
# Single GPU
python train.py

# Multi-GPU
accelerate launch --num_processes=4 train.py

# With DeepSpeed
accelerate launch --use_deepspeed --num_processes=4 train.py
```

## Documentation

- https://huggingface.co/docs/accelerate
