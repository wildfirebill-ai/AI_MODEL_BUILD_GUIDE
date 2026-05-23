# LLaMA Factory — Easy LLM Fine-Tuning Framework

**Version:** 0.9 / `llamafactory>=0.9.0`

## Purpose

LLaMA Factory provides a simplified, unified interface for fine-tuning large language models (LLMs) on consumer and enterprise GPUs. Supports LoRA, QLoRA (4-bit NF4), DoRA, and full-parameter tuning for Llama, Mistral, Qwen, DeepSeek, Baichuan, Yi, ChatGLM, and 100+ architectures. Includes built-in dataset management, chat templates, training monitoring (wandb/tensorboard), and model export. The CLI tool `llamafactory-cli` handles training, evaluation, and inference with single YAML configuration files.

## Installation

```bash
pip install "llamafactory>=0.9.0"
# For QLoRA (bitsandbytes on Windows requires manual install):
pip install bitsandbytes --find-links https://jllllll.github.io/bitsandbytes-windows/
```

Requires PyTorch 2.3+ and CUDA 12.1. For full-parameter tuning, at least 8×A100 80 GB recommended.

## Basic Usage Example

### 1. Configuration file: `train_wfb.yaml`

```yaml
# model
model_name_or_path: mistralai/Mistral-7B-Instruct-v0.3
template: mistral
trust_remote_code: true

# method
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target: all

# dataset
dataset: wfb_instructions
dataset_dir: data
cutoff_len: 512
max_samples: 10000

# training
output_dir: ./output/wfb_mistral_lora
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 2.0e-4
logging_steps: 10
save_steps: 500

# evaluation
val_size: 0.05
eval_steps: 500
per_device_eval_batch_size: 4

# quantization
quantization_bit: 4
quantization_method: nf4
```

### 2. Run training

```bash
llamafactory-cli train train_wfb.yaml
```

### 3. Inference with trained adapter

```python
from llamafactory import LoraModel

model = LoraModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path="./output/wfb_mistral_lora",
    device_map="auto",
)

response = model.chat(
    messages=[{"role": "user", "content": "Explain attention mechanisms."}],
    max_new_tokens=256,
)
print(response)
```

## Advanced Usage / Configuration

- **Dataset format**: Place JSON/JSONL files in `data/` with `"instruction"`, `"input"`, `"output"` fields (or Alpaca/sharegpt format). Register in `data/dataset_info.json`.
- **RLHF**: Stage `dpo` or `ppo` with preference datasets. Use `pref_loss` (sigmoid/ipo/kto) and a reference model.
- **Merge LoRA**: After training, run `python src/export_model.py --model_name_or_path ... --adapter_name_or_path ... --export_dir ./merged`.
- **Multi-GPU**: Set `ddp_timeout: 180000`, `per_device_train_batch_size < global_batch / num_gpus`. Use `torchrun --nproc_per_node=8`.
- **FlashAttention**: Set `flash_attn: fa2` in config for 2x training speed on Ampere+ GPUs.
- **Deepspeed**: Add `deepspeed: configs/ds_z3_offload.json` for ZeRO-3 offloading on limited VRAM.
- **API server**: `llamafactory-cli api --model_name_or_path ... --adapter_name_or_path ...` launches a REST endpoint compatible with OpenAI SDK.

## Integration with the WFB Model Project

```yaml
# wfb_model/configs/llamafactory/wfb_mistral.yaml
dataset: wfb_instructions_modeling  # Custom dataset with WFB domain Q&A
# Training outputs go to wfb_model/checkpoints/llamafactory/
```

WFB uses LLaMA Factory as the primary fine-tuning interface for domain-specific instruction tuning. Dataset files live in `wfb_model/data/llm/` and are registered in `dataset_info.json`. The resulting LoRA adapters are versioned and stored in `wfb_model/checkpoints/llamafactory/`. Evaluation uses held-out WFB benchmark questions.

## Common Pitfalls / Troubleshooting

- **Bitsandbytes on Windows**: Use the Windows wheel link above; the PyPI version does not compile. For WSL2, install `sudo apt install libcudart` first.
- **Dataset not found**: Every dataset listed in the YAML must be defined in `data/dataset_info.json` with the correct file path and format.
- **OOM during training**: Reduce `cutoff_len`, lower `per_device_train_batch_size`, enable `gradient_checkpointing: true`, use `quantization_bit: 4`.
- **LoRA target mismatch**: Set `lora_target: all` to automatically target all linear layers. For specific architectures, check the model's module list in `src/llamafactory/extras/misc.py`.
- **Template not matching**: Wrong `template` (e.g., using `mistral` for a Llama model) causes garbled chat formatting. Use `auto` to auto-detect.
- **Adapter not loading after merge**: Merged models lose LoRA weight separation. Keep the original adapter directory and re-merge if needed.

## Documentation Links

- LLaMA Factory GitHub: https://github.com/hiyouga/LLaMA-Factory
- Wiki / full docs: https://github.com/hiyouga/LLaMA-Factory/wiki
- Dataset preparation: https://github.com/hiyouga/LLaMA-Factory/wiki/Data-preparation
- Supported models: https://github.com/hiyouga/LLaMA-Factory/wiki/Supported-Models
