# HuggingFace PEFT — Parameter-Efficient Fine-Tuning Library

**Version:** 0.13 / `peft>=0.13.0`

## Purpose

PEFT (Parameter-Efficient Fine-Tuning) provides implementations of adapter-based fine-tuning methods that update only a tiny fraction of model parameters while achieving full fine-tuning quality. Supports LoRA, QLoRA, IA3, AdaLoRA, Prefix Tuning, P-Tuning, LoHa, LoKr, OFT, and PiSSA. Works with Transformers, Diffusers, Triton, and vLLM models. Essential for fine-tuning large models on single GPUs.

## Installation

```bash
pip install "peft[torch]>=0.13.0"
# For quantization (QLoRA):
pip install bitsandbytes
```

PEFT is model-agnostic — it wraps any `transformers.PreTrainedModel`.

## Basic Usage Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # ~0.1% of params

# Training loop (standard transformers.Trainer)
from transformers import TrainingArguments, Trainer

train_args = TrainingArguments(
    output_dir="./mistral-lora",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
)
trainer = Trainer(model=peft_model, args=train_args, train_dataset=dataset)
trainer.train()

# Save adapter
peft_model.save_pretrained("./mistral-lora-adapter")
```

### Load adapter for inference

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
adapter = PeftModel.from_pretrained(base, "./mistral-lora-adapter")
adapter = adapter.merge_and_unload()  # Optional: fuse weights
```

## Advanced Usage / Configuration

- **IA3**: `IA3Config(target_modules=["k_proj", "v_proj", "ff"])` — learns rescaling vectors instead of low-rank matrices (even fewer params).
- **AdaLoRA**: `AdaLoraConfig(r=8, target_modules=["q_proj", "v_proj"])` — automatically allocates rank budget across layers.
- **Prefix Tuning**: `PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)` — prepends learnable prefix tokens.
- **P-Tuning**: `PromptEncoderConfig(num_virtual_tokens=20, encoder_hidden_size=128)` — similar to Prefix but with a prompt encoder.
- **LoHa/LoKr**: `LoHaConfig(r=8)` and `LoKrConfig(r=8)` — factorized adapters with even lower parameter counts.
- **PiSSA**: `PiSSAConfig(r=16)` — SVD-initialized adapters that converge faster than LoRA.
- **Multi-adapter stacking**: Load multiple adapters and switch: `peft_model.load_adapter("path2", "adapter2")` then `peft_model.set_adapter("adapter2")`.
- **Gradient checkpointing**: `peft_model.gradient_checkpointing_enable()` for large model training.

## Integration with the WFB Model Project

```python
# wfb_model/peft/wfb_lora.py
from peft import get_peft_model, LoraConfig, TaskType

def wrap_model_for_wfb(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)
```

WFB uses PEFT as the underlying adapter layer for all LLM fine-tuning experiments. LoRA adapters are stored in `wfb_model/checkpoints/peft/`. The training pipeline in `wfb_model/trainers/peft_trainer.py` applies the appropriate PEFT config based on experiment configuration. WFB benchmarks track trade-offs between adapter rank, memory usage, and downstream task accuracy.

## Common Pitfalls / Troubleshooting

- **`target_modules` not matching model**: Set `target_modules=["q_proj", "v_proj"]` for most LLaMA-style models. For GPT-2 use `["c_attn"]`. Inspect `model.named_modules()` if unsure.
- **4-bit training requires `prepare_model_for_kbit_training`**: Without this, gradients may not propagate through `nn.Linear4bit` layers correctly.
- **Adapter weights not loaded**: `PeftModel.from_pretrained` expects a directory containing `adapter_config.json` and `adapter_model.safetensors`. Both files are created by `save_pretrained`.
- **Merge produces NaN**: Ensure the base model uses `torch_dtype=torch.float16` or `bfloat16`. Merge in the same dtype used for training.
- **ModuleNotFoundError: bitsandbytes**: On Windows, use `pip install bitsandbytes --find-links https://jllllll.github.io/bitsandbytes-windows/`.
- **Tokenizer padding side**: For causal LM, set `tokenizer.padding_side = "left"`; for seq2seq, use `"right"`. Wrong padding side causes training collapse.

## Documentation Links

- PEFT docs: https://huggingface.co/docs/peft
- GitHub: https://github.com/huggingface/peft
- Adapter types: https://huggingface.co/docs/peft/main/en/developer_guides/adapters
- PEFT + quantization guide: https://huggingface.co/docs/peft/main/en/task_guides/quantized_lora
