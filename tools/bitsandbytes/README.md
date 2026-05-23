# bitsandbytes

**Version:** 0.43+

**Purpose:** 4-bit and 8-bit quantization for memory-efficient model training and inference. Core dependency for QLoRA.

## Installation

```powershell
pip install bitsandbytes
```

## Usage

```python
import torch
import bitsandbytes as bnb

# 4-bit quantization config
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

# 8-bit Adam optimizer
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-4)

# 32-bit stable embedding layer
bnb.nn.StableEmbedding(...)
```

## Key Features

- **NF4** (4-bit NormalFloat) -- optimal for normally distributed weights
- **FP4** (4-bit Floating Point) -- alternative format
- **8-bit Adam** -- reduces optimizer memory by 2x
- **Double quantization** -- quantizes quantization constants
- **CPU offloading** -- offload optimizer states to CPU

## Documentation

- https://github.com/bitsandbytes-foundation/bitsandbytes
