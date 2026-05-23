# CTranslate2

**Version:** 4.5.0

## Purpose

CTranslate2 is a fast inference engine for transformer models, optimized for both CPU and GPU deployment. It converts models from PyTorch, TensorFlow, and ONNX into a custom binary format with INT8/FP16 quantization, weight sharing, and in-place operations. CTranslate2 delivers 2-4x faster inference and reduced memory usage compared to vanilla PyTorch, making it ideal for low-latency production deployments.

## Installation

```bash
pip install ctranslate2==4.5.0
```

## Basic Usage

### Model Conversion

```python
import ctranslate2

# Convert a PyTorch model to CTranslate2 format
converter = ctranslate2.converters.OpenNMT_PyTorchConverter(
    model_path="model.pt",
)
converter.convert(
    output_dir="ct2_model",
    quantization="int8",
    force=True,
)
```

### Running Inference

```python
import ctranslate2
import numpy as np

# Load the converted model
translator = ctranslate2.Translator(
    model_path="ct2_model",
    device="cpu",
    compute_type="int8",
)

# Prepare input tokens
tokens = ["▁The", "▁wind", "▁farm", "▁produces", "▁energy"]
results = translator.translate_batch(
    [tokens],
    beam_size=4,
    max_length=100,
    return_scores=True,
)

output_tokens = results[0].hypotheses[0]
score = results[0].scores[0]
print(f"Output: {output_tokens}, Score: {score:.3f}")
```

### Direct PyTorch Model Support

```python
import ctranslate2

# For HuggingFace models
converter = ctranslate2.converters.TransformersConverter(
    model_name_or_path="bert-base-uncased",
)
converter.convert(output_dir="ct2_bert", quantization="int8")
```

## Advanced Usage / Configuration

### GPU Inference with FP16

```python
translator = ctranslate2.Translator(
    model_path="ct2_model",
    device="cuda",
    device_index=0,
    compute_type="float16",
)
```

### Batch Translation with Async

```python
batch = [
    ["▁Farm", "▁A", "▁output"],
    ["▁Farm", "▁B", "▁efficiency"],
]

results = translator.translate_batch(
    batch,
    batch_type="tokens",
    max_batch_size=16,
)
for result in results:
    print(" ".join(result.hypotheses[0]))
```

### Server Mode

```python
# Start a REST server
translator.serve(port=8080)

# Or use the CLI:
# ct2-rest --model_path ct2_model --port 8080
```

## Integration with WFB Model Project

CTranslate2 serves WFB transformer models (e.g., time-series forecasters, text encoders) for low-latency CPU inference in edge deployments. Convert trained PyTorch models to CTranslate2 format with INT8 quantization for 4x smaller footprint and faster inference on commodity hardware without GPUs.

## Common Pitfalls / Troubleshooting

- **Model not supported:** Only transformer-based models are supported. Use `ctranslate2.converters` to check compatibility with your architecture.
- **Quantization loss:** Try `compute_type="int8_float16"` for mixed precision, or use `"float16"` for higher accuracy.
- **Batch dimension mismatch:** CTranslate2 expects tokenized inputs. Ensure your tokenizer matches the model's vocabulary (e.g., sentencepiece, BPE).
- **Missing CUDA libraries:** Install CUDA toolkit or use `device="cpu"` for CPU inference. CTranslate2 uses cuBLAS for GPU.

## Documentation Links

- [CTranslate2 Docs](https://opennmt.net/CTranslate2/)
- [Conversion Guide](https://opennmt.net/CTranslate2/guides/converters.html)
- [Quantization Reference](https://opennmt.net/CTranslate2/quantization.html)
