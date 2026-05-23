# axolotl

**Purpose:** Streamlined fine-tuning framework for LLMs with support for LoRA, QLoRA, Flash Attention, and multi-GPU.

## Installation

```powershell
pip install axolotl
```

## Usage

```yaml
# config.yml — annotated with key/value explanations
model_config:
  # Model to fine-tune from HuggingFace or local path
  base_model: mistralai/Mistral-7B-v0.1
  # Model type for loading
  model_type: MistralForCausalLM
  # Tokenizer to use
  tokenizer_config: mistralai/Mistral-7B-v0.1

# LoRA / QLoRA configuration
lora:
  # Whether to use LoRA
  lora: true
  # LoRA rank: higher = more capacity, more memory
  lora_r: 16
  # LoRA alpha: scaling factor (higher = stronger adaptation)
  lora_alpha: 32
  # Dropout for regularization
  lora_dropout: 0.05
  # Which modules to apply LoRA to
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# 4-bit quantization
load_in_4bit: true
# Quantization type: nf4 or fp4
bnb_4bit_quant_type: nf4
# Compute dtype for quantized layers
bnb_4bit_compute_dtype: bfloat16
# Double quantization (extra memory savings)
bnb_4bit_use_double_quant: true

# Training hyperparameters
training:
  # Micro batch size per GPU
  micro_batch_size: 2
  # Gradient accumulation steps (effective batch = micro_batch * grad_accum * num_gpus)
  gradient_accumulation_steps: 4
  # Number of training epochs
  num_epochs: 3
  # Learning rate
  learning_rate: 2e-4
  # Optimizer
  optimizer: adamw_bnb_8bit
  # Learning rate scheduler type
  lr_scheduler: cosine
  # Warmup ratio (fraction of total steps)
  warmup_ratio: 0.03
  # Maximum gradient norm for clipping
  max_grad_norm: 0.3
  # Sequence length for training
  sequence_len: 2048
  # Flash Attention 2
  flash_attention: true
  # Save checkpoints every N steps
  save_steps: 500
  # Log metrics every N steps
  logging_steps: 10

# Dataset configuration
datasets:
  # Path to dataset (HuggingFace or local JSONL)
  - path: my_dataset.jsonl
    # Dataset type for formatting
    ds_type: json
    # How to format the conversation
    conversation: sharegpt
    # Field names in the data
    field:
      messages: conversations
```

```powershell
# Run training
accelerate launch -m axolotl.cli.train config.yml
```

## Documentation

- https://github.com/axolotl-ai-cloud/axolotl
