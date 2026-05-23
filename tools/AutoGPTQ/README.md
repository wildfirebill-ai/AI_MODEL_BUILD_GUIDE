# AutoGPTQ

**Version:** 0.7+

**Purpose:** GPTQ (Generative Pretrained Transformer Quantization) for 4-bit weight quantization using an Optimal Brain Quantizer (OBQ) approach. GPTQ minimizes quantization error layer-by-layer on a calibration dataset, achieving near-lossless 4-bit compression. In the WFB pipeline, AutoGPTQ provides an alternative to AWQ for quantizing trained models before deployment.

## Installation

```powershell
pip install "auto-gptq==0.7.1" "optimum==1.24.0"
```

## Basic Usage

### Quantize a Model

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_path = "./checkpoints/final"
quant_path = "./checkpoints/gptq"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_texts = [t["text"] for t in dataset.select(range(128))]
calib_ids = tokenizer(calib_texts, return_tensors="pt", padding=True)["input_ids"].to("cuda")

quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True, damp_percent=0.01)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=quant_config)
model.quantize(calib_ids, use_triton=False, batch_size=1)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Inference with Quantized Model

```python
model = AutoGPTQForCausalLM.from_quantized(
    "./checkpoints/gptq", device="cuda:0", use_triton=False,
)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/gptq")
inputs = tokenizer("GPTQ enables efficient", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Advanced Usage

### Group Size and desc_act Tuning

```python
configs = [
    {"group_size": 128, "desc_act": False},  # Fastest
    {"group_size": 128, "desc_act": True},   # Best quality
    {"group_size": 64,  "desc_act": True},   # Highest quality
]
for cfg in configs:
    qc = BaseQuantizeConfig(bits=4, **cfg)
    m = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=qc)
    m.quantize(calib_ids, use_triton=False)
```

### Perplexity Evaluation

```python
import torch
def perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

ppl = perplexity(model, tokenizer, "Attention mechanisms model long-range dependencies.")
print(f"PPL: {ppl:.2f}")
```

## Integration with WFB Model

AutoGPTQ is used in **Part 7** (Optimization & Quantization) alongside AWQ. While AWQ uses activation statistics, GPTQ uses Hessian-based OBQ for layer-wise quantization. GPTQ often achieves slightly better perplexity on larger models (34B+) with higher quantization latency. Both produce models loadable by vLLM, TGI, and ExLlamaV2.

## Common Pitfalls

- **Triton dependency**: `use_triton=True` requires the Triton compiler. Use `False` on Windows.
- **desc_act=True overhead**: Improves quality but slows inference 10-20%. Disable for latency-sensitive apps.
- **damp_percent tuning**: If NaN, increase `damp_percent` to 0.1 or 0.5.
- **Batch size**: Large calibration sets may OOM. Use `batch_size=1`.
- **Model compatibility**: Check `auto_gptq/models/` for supported architectures.

## Documentation

- https://github.com/AutoGPTQ/AutoGPTQ
- https://huggingface.co/docs/transformers/quantization/gptq
