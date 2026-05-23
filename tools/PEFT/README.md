# PEFT (Parameter-Efficient Fine-Tuning)

**Version:** 0.12+

**Purpose:** Library for efficient fine-tuning methods — LoRA, Prefix Tuning, P-Tuning, Prompt Tuning, IA3.

## Installation

```powershell
pip install peft
```

## Supported Methods

| Method | Description | Trainable Params |
|--------|-------------|-----------------|
| **LoRA** | Low-rank matrices on attention layers | 0.1-1% |
| **QLoRA** | LoRA + 4-bit quantization | 0.1-1% |
| **Prefix Tuning** | Learnable prefix tokens | 0.1-0.5% |
| **P-Tuning** | Learnable continuous prompts | 0.01-0.1% |
| **Prompt Tuning** | Soft prompt embeddings | 0.01% |
| **IA3** | Learnable scaling vectors | 0.01% |

## LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,                           # Rank — higher = more capacity
    lora_alpha=32,                  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,              # Dropout for regularization
    bias="none",                    # Train bias terms? "none", "all", "lora_only"
    task_type=TaskType.CAUSAL_LM,   # Task type
)

model = get_peft_model(base_model, config)
print(f"Trainable params: {model.num_parameters(only_trainable=True) / 1e6:.2f}M")
```

## Saving & Loading

```python
# Save
model.save_pretrained("lora-adapter")

# Load
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "lora-adapter")

# Merge into base model
merged = model.merge_and_unload()
```

## Memory Comparison (7B Model)

| Method | VRAM Required | Relative to Full FT |
|--------|--------------|-------------------|
| Full Fine-tune | 56 GB | 100% |
| LoRA (fp16) | 18 GB | 32% |
| QLoRA (4-bit) | 6 GB | 11% |
| Prefix Tuning | 16 GB | 29% |

## Documentation

- https://huggingface.co/docs/peft
