# AWQ (Activation-Aware Weight Quantization)

**Version:** 0.2+

**Purpose:** 4-bit weight quantization with minimal quality loss for LLM inference.

## Installation

```powershell
pip install awq
```

## Usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# Quantize to 4-bit
model.quantize(
    quant_config={
        "version": "GEMM",           # or "GEMV" for faster
        "zero_point": True,
        "q_group_size": 128,         # Group size for quantization
    },
    calib_dataset=calibration_data,  # 128-512 samples
)

model.save_quantized("./model-awq")

# Inference with vLLM
from vllm import LLM
llm = LLM(model="./model-awq", quantization="AWQ")
```

## Key Features

- **Near-lossless 4-bit quantization** for 7B-70B models
- **3-4x memory reduction** vs FP16
- **2-3x faster inference** on memory-bandwidth-bound scenarios
- **No calibration needed** for most models

## Documentation

- https://github.com/mit-han-lab/llm-awq
