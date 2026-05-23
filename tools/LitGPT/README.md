# LitGPT

**Purpose:** Command-line tool for training, fine-tuning, and deploying LLMs with support for 20+ model architectures.

## Installation

```powershell
pip install litgpt
```

## Usage

```powershell
# Download a model
litgpt download mistralai/Mistral-7B-v0.1

# Fine-tune with LoRA
litgpt finetune_lora \
    --checkpoint_dir checkpoints/mistral-7b \
    --data JSON \
    --data.json_path train.jsonl \
    --out_dir out/lora \
    --precision bf16-true \
    --learning_rate 3e-4 \
    --epochs 3

# Chat with fine-tuned model
litgpt chat --checkpoint_dir out/lora/final

# Convert to GGUF for llama.cpp
litgpt convert_lora --checkpoint_dir out/lora/final
# Then convert to GGUF using convert.py
```

## Python API

```python
from litgpt import LLM

llm = LLM.load("mistralai/Mistral-7B-v0.1")
llm.finetune(
    data=[{"instruction": "...", "output": "..."}],
    learning_rate=3e-4,
    epochs=3,
    lora_r=16,
)
llm.save("finetuned-model")
```

## Supported Models

LLaMA 2/3, Mistral, Mixtral, Falcon, Phi, Gemma, Qwen, DeepSeek, Yi, StableLM, and more.

## Documentation

- https://github.com/Lightning-AI/litgpt
