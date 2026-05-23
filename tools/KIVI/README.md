# KIVI (2-bit KV Cache Quantization)

**Version:** 0.1+ (GitHub-based — no PyPI release; install from source)

**Purpose:** Compress the KV cache to 2-bit precision, enabling 4-8x longer context lengths on the same GPU hardware. KIVI uses group-wise quantization for keys and values independently, with per-channel and per-token residual storage to preserve quality. Essential for inference on very long sequences (128K–1M tokens).

## Installation

```powershell
# KIVI is not on PyPI — install from source
git clone https://github.com/jy-yuan/KIVI.git
cd KIVI
pip install -e .
```

**Requirements:** CUDA 12.1+, PyTorch 2.1+, 80+ GB GPU memory recommended for large models. Requires `ninja` for building CUDA kernels.

### Verify
```python
from kivi import KIVIQuantizedLLM
print("KIVI imported successfully")
```

## Basic Usage

```python
from kivi import KIVIQuantizedLLM

llm = KIVIQuantizedLLM(
    model="/path/to/model",
    k_bits=2,              # 2-bit keys (4x compression)
    v_bits=2,              # 2-bit values (4x compression)
    group_size=32,         # Group size for quantization
    residual_length=128,   # Full-precision residual length
    max_length=200000,     # Max context length
    device="cuda",
)

output = llm.generate("Long context prompt..." * 1000, max_new_tokens=512)
print(output)
```

## Memory Savings

| Context Length | FP16 KV Cache | KIVI (2-bit) | Savings |
|---------------|--------------|--------------|---------|
| 32K | 4 GB | 0.5 GB | 8x |
| 128K | 16 GB | 2 GB | 8x |
| 1M | 128 GB | 16 GB | 8x |

## Advanced Usage / Configuration
- **`k_bits`/`v_bits`:** 2-bit is default; try 3-bit or 4-bit if quality drops
- **`group_size`:** Smaller groups (16) → higher quality but less compression; larger groups (64) → more compression, may degrade quality
- **`residual_length`:** Keep more full-precision tokens at the end of the sequence for better generation quality (default 128)
- **`prefill_chunk_size`:** Controls GPU memory during prefill — reduce if OOM during the first forward pass
- **KIVI + FlashAttention:** Compatible; both optimizations apply simultaneously

## Integration with WFB Model
KIVI is used in the WFB model's long-context inference pipeline (`scripts/long_context_infer.py`). It pairs with RingFlashAttention for distributed long-context serving. The WFB model uses KIVI primarily for evaluation on long-document benchmarks (128K+ tokens) where FP16 KV cache would exceed GPU memory.

## Common Pitfalls / Troubleshooting
- **Build fails:** Ensure `ninja` is installed (`pip install ninja`) and CUDA toolkit is on PATH
- **"CUDA out of memory" with KIVI enabled:** Reduce `group_size` (try 64) or reduce `residual_length` (try 0); if still OOM, the model itself is too large for the GPU — KIVI only compresses KV cache, not model weights
- **Quality degradation:** Increase `k_bits` and `v_bits` to 3 or 4; increase `residual_length`; decrease `group_size`; some models are more sensitive — test with a validation set
- **Slow first generation:** KIVI calibrates quantization scales on the first forward pass — subsequent generations are fast
- **Model compatibility:** KIVI supports Llama-family and GPT-NeoX architectures — check the repo for the latest model support list

## When to Use KIVI
KIVI is beneficial when:
- Context length exceeds what fits in GPU memory with standard KV cache
- Running long-document QA, summarization, or multi-turn chat (128K+ tokens)
- Inference latency is less critical than memory savings (KIVI adds ~10-15% overhead vs FP16)

Not beneficial when:
- Context fits easily in GPU memory (most <32K contexts)
- Maximum throughput is required (FP16 is faster)
- Quality cannot be sacrificed (use 4-bit or skip quantization)

## Integration with WFB Model
KIVI is integrated into the long-context inference pipeline (`scripts/long_context_infer.py`). It pairs with RingFlashAttention — RFA handles the distributed prefill over 1M+ tokens, then KIVI compresses the KV cache during autoregressive generation. Configuration is in `configs/inference/long_context.yaml`:

```yaml
kivi:
  enabled: true
  k_bits: 2
  v_bits: 2
  group_size: 32
  residual_length: 128
```

## Common Pitfalls / Troubleshooting
- **Build fails:** Ensure `ninja` is installed (`pip install ninja`) and CUDA toolkit is on PATH
- **"CUDA out of memory" with KIVI enabled:** Reduce `group_size` (try 64) or reduce `residual_length` (try 0); if still OOM, the model itself is too large for the GPU — KIVI only compresses KV cache, not model weights
- **Quality degradation:** Increase `k_bits` and `v_bits` to 3 or 4; increase `residual_length`; decrease `group_size`; some models are more sensitive — test with a validation set
- **Slow first generation:** KIVI calibrates quantization scales on the first forward pass — subsequent generations are fast
- **Model compatibility:** KIVI supports Llama-family and GPT-NeoX architectures — check the repo for the latest model support list

## Documentation
- https://github.com/jy-yuan/KIVI
- https://arxiv.org/abs/2402.02750 (KIVI paper)
