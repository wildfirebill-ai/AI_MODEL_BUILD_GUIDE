# AutoAWQ

**Version:** 0.2+

**Purpose:** Activation-Aware Weight Quantization (AWQ) that produces 4-bit quantized models with minimal perplexity loss. AWQ identifies the 1% most important weights (based on activation magnitudes) and protects them during quantization, achieving near-lossless compression. In the WFB pipeline, AutoAWQ quantizes trained models from FP16 to 4-bit for efficient deployment.

## Installation

```powershell
pip install "autoawq==0.2.8"
```

## Basic Usage

### Quantize a Model

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

model_path = "./checkpoints/final"
quant_path = "./checkpoints/awq"

model = AutoAWQForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_data = [tokenizer(t["text"]) for t in dataset.select(range(128))]

model.quantize(
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
    calib_dataset=calib_data,
)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Inference with Quantized Model

```python
model = AutoAWQForCausalLM.from_quantized(
    "./checkpoints/awq", device_map="auto", fuse_layers=True,
)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/awq")
inputs = tokenizer("Quantization preserves", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Advanced Usage

### Custom Quant Config

```python
quant_config = {
    "zero_point": True,
    "q_group_size": 128,       # 128 or 64; smaller = higher quality
    "w_bit": 4,                 # 4-bit standard; 3-bit experimental
    "version": "GEMM",          # GEMM or GEMV (faster row-wise)
    "calib_dataset_size": 256,
}
```

### Integration with vLLM

```python
from vllm import LLM, SamplingParams
llm = LLM(model="./checkpoints/awq", quantization="AWQ", dtype="half")
outputs = llm.generate(["AWQ quantized inference"], SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
```

## Integration with WFB Model

AutoAWQ is used in **Part 7** (Optimization & Quantization) of the WFB pipeline. After pretraining (Part 5) and post-training (Part 6), the FP16 checkpoint is quantized to 4-bit AWQ for production. AWQ models use 3-4x less GPU memory and achieve 2-3x faster inference with <1% perplexity loss. Deploy via vLLM or TGI with `--quantize awq`. AWQ is preferred over GPTQ when activation patterns are known and calibration data matches the target domain.

## Common Pitfalls

- **Calibration data mismatch**: Use calibration data similar to your target domain.
- **CUDA OOM**: Quantization loads the full FP16 model. Use `device_map="sequential"` or multi-GPU.
- **fuse_layers errors**: Model-specific. Check `awq/models/` for supported architectures.
- **Version mismatch**: Config keys differ between v0.1 and v0.2. Check `model.quantize.__doc__`.
- **Perplexity degradation**: If >2%, try `q_group_size=64` or `calib_dataset_size=512`.

## Documentation

- https://github.com/casper-hansen/AutoAWQ
- https://huggingface.co/docs/transformers/quantization/awq
