# NVIDIA NeMo — LLM Training Framework

**Version:** 2.0 / `nemo>=2.0.0`

## Purpose

NVIDIA NeMo provides a scalable framework for training, customizing, and deploying large language models (LLMs) using NeMo Megatron Core. It supports distributed training across hundreds of GPUs with tensor/pipeline parallelism, sequence parallelism, and selective activation recomputation. Key capabilities include P-Tuning, LoRA, SFT (supervised fine-tuning), and RLHF.

## Installation

```bash
# Requires CUDA 12.1+ and PyTorch 2.3+
pip install "nemo>=2.0.0" "nemo-megatron>=2.0.0" --extra-index-url https://pypi.nvidia.com

# For Apex and TransformerEngine support:
pip install "nemo[all]>=2.0.0" --extra-index-url https://pypi.nvidia.com
```

NeMo expects `torch.distributed` launch via `torchrun` or Slurm.

## Basic Usage Example

```python
import nemo.collections.nlp as nemo_nlp
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg):
    # Load a pretrained GPT model
    model = nemo_nlp.models.MegatronGPTModel.restore_from(
        "/models/megatron_gpt.nemo"
    )
    # Or create from scratch with config
    # model = nemo_nlp.models.MegatronGPTModel(cfg.model)

    # P-Tuning for prompt-based fine-tuning
    model.add_adapter(
        adapter_type="ptuning",
        adapter_config={
            "virtual_tokens": 20,
            "hidden_dim": 512,
            "token_dim": model.cfg.hidden_size,
        },
    )

    # Set up training with NeMo's exp_manager
    trainer = nemo_nlp.models.setup_trainer(cfg)

    # Train (forward/backward handled internally)
    trainer.fit(model)

if __name__ == "__main__":
    main()
```

Run on multiple GPUs:

```bash
torchrun --nproc_per_node=8 train.py model.micro_batch_size=2 model.tensor_model_parallel_size=4
```

## Advanced Usage / Configuration

- **SFT (Supervised Fine-Tuning)**: Use `model.sft()` or configure the `sft` recipe in the YAML config. NeMo automatically packs data and handles loss masking.
- **LoRA**: Enable via `model.add_adapter("lora", {"rank": 8, "alpha": 16})` for memory-efficient fine-tuning.
- **NeMo Megatron Core**: Supports tensor parallel (TP), pipeline parallel (PP), sequence parallel (SP), and expert parallel (EP) for Mixture-of-Experts.
- **Evaluation**: `model.evaluate()` runs perplexity, accuracy, and downstream benchmarks via `nemo_nlp.datasets`.
- **Checkpointing**: Periodic `.nemo` checkpoint saves via `exp_manager`; supports resume from interrupted runs.
- **Distillation**: `nemo_nlp.models.MegatronGPTModel.distill()` using teacher-student KL divergence.

## Integration with the WFB Model Project

```python
# Place in wfb_model/nemo/sft_wfb.py
import nemo.collections.nlp as nemo_nlp

def train_wfb_sft(base_model: str = "mistral-7b.nemo"):
    model = nemo_nlp.models.MegatronGPTModel.restore_from(base_model)
    model.add_adapter("lora", {"rank": 16, "alpha": 32})
    # WFB custom training loop wraps NeMo's trainer
    # Configs stored in wfb_model/configs/nemo/
```

WFB stores NeMo configs (`megatron_gpt_config.yaml`) under `wfb_model/configs/nemo/`. The project's Slurm launcher invokes `torchrun` on multi-node GPU partitions, and all checkpoints are synced to shared NFS or S3 storage.

## Common Pitfalls / Troubleshooting

- **CUDA OOM during forward**: Reduce `micro_batch_size` or increase `tensor_model_parallel_size`. Enable `model.activations_checkpoint_granularity="selective"`.
- **NCCL timeouts on multi-node**: Set `NCCL_IB_TIMEOUT=22` and `NCCL_ASYNC_ERROR_HANDLING=1`. Ensure InfiniBand is configured on all nodes.
- **Config merge errors**: NeMo uses Hydra and OmegaConf. Use `++model.micro_batch_size=1` to override from CLI.
- **Checkpoint resume mismatch**: Ensure the `.nemo` file matches the model config exactly; NeMo validates `cfg.model` on restore.
- **Tokenizer not found**: Pass `tokenizer_model_path` or use `nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset.MegatronGPTDataset` with the correct vocabulary path.
- **Slow data loading**: Increase `data.num_workers` (typically 2–4 per GPU) and enable `data.prefetch_factor=2`.

## Documentation Links

- NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/
- NeMo Megatron Core: https://github.com/NVIDIA/Megatron-LM
- P-Tuning and SFT recipes: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/gpt/gpt_tuning.html
- Example configs: https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/conf
