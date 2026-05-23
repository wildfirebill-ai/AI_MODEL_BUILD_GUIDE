# HuggingFace Transformers

**Version:** 4.45+

**Purpose:** Provides pre-trained models, model architectures, tokenizers, and training utilities for NLP.

## Installation

```powershell
pip install transformers
```

## Key Components Used in This Project

- `transformers.AutoTokenizer` — load and use tokenizers
- `transformers.AutoModelForCausalLM` — load causal language models
- `transformers.TrainingArguments` — training configuration
- `transformers.get_cosine_schedule_with_warmup` — learning rate schedules
- `transformers.BitsAndBytesConfig` — quantization for QLoRA

## Quick Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `PreTrainedModel` | Base class for all models |
| `PreTrainedTokenizer` | Base class for all tokenizers |
| `Trainer` | Full-featured training loop |
| `DataCollator` | Batching and padding utilities |

## Documentation

- https://huggingface.co/docs/transformers
