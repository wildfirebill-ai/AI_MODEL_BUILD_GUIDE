# HuggingFace Datasets

**Version:** 2.20+

**Purpose:** Efficient dataset loading, processing, and streaming for ML training.

## Installation

```powershell
pip install datasets
```

## Key Features

- **Streaming** — load datasets without downloading fully (important for large datasets)
- **Map/Filter** — transform data efficiently
- **Sharding** — split datasets for distributed training
- **Integration** — works directly with PyTorch DataLoader

## Quick Example

```python
from datasets import load_dataset

# Stream from HuggingFace (no full download)
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Take a sample
for i, example in enumerate(dataset):
    if i < 5:
        print(example["text"][:200])
```

## Common Datasets for LLM Training

| Dataset | Size | Type |
|---------|------|------|
| `HuggingFaceFW/fineweb` | 15T tokens | Web text |
| `codeparrot/github-code` | 200GB | Code |
| `bigcode/the-stack` | 6TB | Code |
| `tiiuae/falcon-refinedweb` | 600B tokens | Web text |

## Saving to Disk

```python
dataset.save_to_disk("path/to/save", num_shards=64)
dataset = load_from_disk("path/to/save")
```

## Documentation

- https://huggingface.co/docs/datasets
