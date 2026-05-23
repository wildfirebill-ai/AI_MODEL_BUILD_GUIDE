# QLoRA (Quantized Low-Rank Adaptation)

**Purpose:** Train large language models on consumer GPUs by combining 4-bit quantization with LoRA adapters.

## How It Works

1. **4-bit NormalFloat quantization** — compresses model to ~4GB for a 7B model
2. **LoRA adapters** — small trainable matrices (16-256 rank) added to attention layers
3. **Double quantization** — quantizes the quantization constants for extra savings
4. **Paged optimizers** — uses CPU memory for optimizer states when GPU runs out

## Installation

```powershell
pip install bitsandbytes peft trl
```

## Training Script (7B on RTX 4090)

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True).select(range(1000)),
    args=TrainingArguments(
        output_dir="./qlora-out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        max_steps=1000,
        save_steps=500,
        logging_steps=10,
    ),
    max_seq_length=2048,
)
trainer.train()
model.save_pretrained("qlora-final")
```

## Memory Requirements

| Model Size | Full Fine-tune | QLoRA (4-bit) |
|-----------|---------------|---------------|
| 7B | 56 GB | 6 GB |
| 13B | 104 GB | 10 GB |
| 34B | 272 GB | 20 GB |
| 70B | 560 GB | 35 GB |

## Merging LoRA Weights

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "qlora-final")
merged = model.merge_and_unload()
merged.save_pretrained("merged-model")
```
