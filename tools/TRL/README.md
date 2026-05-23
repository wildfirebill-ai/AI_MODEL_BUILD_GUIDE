# TRL (Transformer Reinforcement Learning)

**Version:** 0.10+

**Purpose:** Training library for language models using RLHF (PPO, DPO, ILQL), supervised fine-tuning (SFT), and preference tuning.

## Installation

```powershell
pip install trl
```

## Key Trainers

| Trainer | Purpose |
|---------|---------|
| `SFTTrainer` | Supervised fine-tuning on instruction data |
| `DPOTrainer` | Direct Preference Optimization — align with human preferences |
| `PPOTrainer` | Proximal Policy Optimization — RL from human feedback |
| `RewardTrainer` | Train a reward model for RLHF |

## SFTTrainer Example

```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="instructions.jsonl")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./sft-out",
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        bf16=True,
        max_steps=1000,
    ),
    max_seq_length=2048,
    formatting_func=lambda ex: f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}",
)

trainer.train()
```

## DPO Example (Alignment)

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=TrainingArguments(
        output_dir="./dpo-out",
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        max_steps=500,
    ),
    beta=0.1,  # KL penalty coefficient
)

dpo_trainer.train()
```

## Documentation

- https://huggingface.co/docs/trl
