# ExLlamaV2

**Version:** 0.2+

**Purpose:** High-throughput inference engine optimized for quantized Llama-family models. ExLlamaV2 provides custom CUDA kernels with efficient KV cache management, 4-bit (GPTQ/AWQ) kernel fusion, speculative decoding, and multi-LoRA support. It achieves among the highest token throughput for quantized models by minimizing GPU kernel launch overhead. In the WFB pipeline, ExLlamaV2 serves as the inference engine for quantized models when maximum throughput is required.

## Installation

```powershell
pip install "exllamav2==0.2.0"
```

## Basic Usage

### Loading and Generating

```python
import torch
from exllamav2 import (
    ExLlamaV2Config, ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer,
)

config = ExLlamaV2Config()
config.model_dir = "./checkpoints/gptq"
config.prepare()

model = ExLlamaV2(config)
model.load_autosplit()

cache = ExLlamaV2Cache(model, max_seq_len=32768)
tokenizer = ExLlamaV2Tokenizer(config)

input_ids = tokenizer.encode("ExLlamaV2 is fast because", encode_special_tokens=True)
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()

output_ids = model.generate(input_ids, cache=cache, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(output_ids[0], decode_special_tokens=True)
print(response)
```

### Streaming

```python
from exllamav2.generator import ExLlamaV2StreamGenerator

generator = ExLlamaV2StreamGenerator(model, cache, tokenizer)
generator.set_stop_conditions(["<|eot_id|>", tokenizer.eos_token_id])
generator.begin_stream(input_ids, settings)
for chunk in generator.stream():
    print(chunk["text"], end="", flush=True)
```

## Advanced Usage

### Speculative Decoding

```python
draft_cfg = ExLlamaV2Config(); draft_cfg.model_dir = "./checkpoints/draft"
draft_model = ExLlamaV2(draft_cfg); draft_model.load()
target_cfg = ExLlamaV2Config(); target_cfg.model_dir = "./checkpoints/target"
target_model = ExLlamaV2(target_cfg); target_model.load_autosplit()

draft_cache = ExLlamaV2Cache(draft_model, max_seq_len=4096)
target_cache = ExLlamaV2Cache(target_model, max_seq_len=4096)
```

### Multi-LoRA

```python
from exllamav2.lora import ExLlamaV2Lora

model = ExLlamaV2(config); model.load()
lora_1 = ExLlamaV2Lora.from_directory(model, "./loras/coding")
lora_2 = ExLlamaV2Lora.from_directory(model, "./loras/chat")
model.set_lora(lora_1)  # Switch adapter
model.set_lora(None)     # Disable LoRA
```

## Integration with WFB Model

ExLlamaV2 is used in **Part 7** (Optimization & Quantization) and **Part 8** (Production & Deployment) of the WFB pipeline. After quantizing with GPTQ or AWQ, ExLlamaV2 provides the fastest inference path for Llama-architecture models on consumer/workstation GPUs. Speculative decoding (draft + target model) is particularly valuable for latency-sensitive apps. For single-GPU setups with limited VRAM, ExLlamaV2 with 4-bit quantization often outperforms vLLM.

## Common Pitfalls

- **Model format**: Requires GPTQ or native format. AWQ support is experimental.
- **Tokenizer**: Uses its own `ExLlamaV2Tokenizer`. HF tokenizers need conversion via `convert_tokenizer.py`.
- **CUDA graphs**: `max_seq_len` is baked into cache. Changing requires reinitialization.
- **Multi-GPU**: `load_autosplit()` distributes layers. Monitor with `model.load(progress=True)`.
- **Draft model spec**: Must match target vocabulary. Use the same tokenizer.
- **Community**: Smaller than vLLM. Check GitHub issues for model-specific quirks.

## Documentation

- https://github.com/turboderp/exllamav2
- https://github.com/turboderp/exllamav2/wiki
