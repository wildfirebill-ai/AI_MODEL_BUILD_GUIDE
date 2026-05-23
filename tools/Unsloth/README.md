# Unsloth — Optimized LLM Training (2x Faster, 50% Less Memory)

**Version:** 2024.11 / `unsloth>=2024.11`

## Purpose

Unsloth provides optimized kernel implementations for LLM fine-tuning that achieve up to 2x training speed and 50% memory reduction compared to standard Hugging Face + PEFT workflows. It achieves this through manual fused attention kernels, 4-bit QLoRA optimizations with nf4/double quantization, and reduced memory fragmentation. Supports Llama, Mistral, Gemma, Qwen, DeepSeek, Phi, Yi, and 20+ model families. Drop-in replacement for Hugging Face Transformers + PEFT with the same API.

## Installation

```bash
# For Llama/Mistral/Qwen etc. (CUDA 12.1+)
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"

# For Maxwell/Pascal GPUs (architecture-specific wheels)
# See https://github.com/unslothai/unsloth for the correct pip command
```

Requires PyTorch 2.3+, CUDA 12.1, and an NVIDIA GPU with compute capability 7.0+ (V100, RTX 20xx, A100, H100, etc.).

## Basic Usage Example

```python
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model with 2x faster kernels
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Add LoRA adapters using Unsloth's optimized implementation
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Fast gradient checkpointing
    random_state=42,
)

# Standard SFTTrainer — Unsloth handles the rest
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        logging_steps=10,
        save_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)
trainer.train()

# Save 4-bit model (not a merge, just adapter + config)
model.save_pretrained_merged("./lora_model", tokenizer, save_method="merged_16bit")
```

### Inference with trained model

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(["### Instruction: Explain quantum computing\n\n### Response:"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Advanced Usage / Configuration

- **`use_gradient_checkpointing="unsloth"`**: Uses the custom checkpointing kernel (2x faster than HF default). Alternatives: `"hf"` or `True`.
- **`max_seq_length`**: Set higher (4096–8192) for long-context fine-tuning; Unsloth handles RoPE scaling automatically.
- **`save_method` options**: `"merged_16bit"`, `"merged_4bit"`, `"lora"`, `"gguf"`, `"q4_k_m"` (for llama.cpp).
- **`load_in_4bit=True`**: Uses Unsloth's own 4-bit linear layer (not bitsandbytes). Falls back to bitsandbytes if unavailable.
- **Continuous batching**: Use `FastLanguageModel.for_training(model)` after training to reset dropout/grad norms for further training.
- **Convergence gains**: Unsloth reduces loss instability with a 0 lora_dropout default — empirically better for most tasks.
- **Model families**: `llama`, `mistral`, `gemma`, `qwen2`, `phi3`, `deepseek`, `yi`, `cohere`, `dbrx` — all with model-specific optimizations.

## Integration with the WFB Model Project

```python
# wfb_model/trainers/unsloth_trainer.py
from unsloth import FastLanguageModel

def train_wfb_with_unsloth(dataset, output_dir="./checkpoints/unsloth"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=8, target_modules=["q_proj", "v_proj"])
    # followed by SFTTrainer...
```

WFB uses Unsloth as the default training backend for LLM fine-tuning due to its memory efficiency (enables Mistral 7B fine-tuning on 12 GB GPUs). All training configs in `wfb_model/configs/unsloth/` use Unsloth's kernel-optimized settings. Output adapters are saved in both Unsloth format (`save_pretrained_merged`) and HF-compatible format for inference with vLLM.

## Common Pitfalls / Troubleshooting

- **Unsloth not found**: After install, the wheel pre-compiles for your CUDA arch. If it fails, install from source: `pip install ninja && pip install unsloth --no-binary unsloth`.
- **`max_seq_length` too high**: Each additional token costs O(n²) in attention. For 8K context ensure A100 80 GB or use `load_in_4bit=True` and reduce batch size to 1.
- **`save_pretrained_merged` OOM**: Merging requires loading base model in 16-bit. Try `save_method="lora"` to save only the adapter, or use `save_pretrained_gguf` for GGUF quantized output.
- **Training diverges**: Set `lora_dropout=0` (Unsloth default). Higher dropout with 4-bit can destabilize training. Use warmup steps (10% of total).
- **Tokenizer padding**: Unsloth automatically sets `pad_token = eos_token` for causal LMs; if you override, ensure left padding for generation.
- **Loss is nan**: Verify `load_in_4bit` is `True` (or `False` for 16-bit). Mixed precision enabled? Set `bf16=True` or `fp16=True` in `TrainingArguments`.

## Documentation Links

- Unsloth GitHub: https://github.com/unslothai/unsloth
- Unsloth docs: https://docs.unsloth.ai/
- Supported models: https://docs.unsloth.ai/get-started/all-our-models
- Performance benchmarks: https://github.com/unslothai/unsloth#benchmarks
