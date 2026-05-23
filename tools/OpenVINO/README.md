# OpenVINO

**Version:** 2025.2.0

## Purpose

Intel OpenVINO (Open Visual Inference and Neural Network Optimization) is a toolkit for optimizing and deploying deep learning models on Intel hardware: CPU, integrated GPU, and NPU (Neural Processing Unit). It converts models from frameworks like PyTorch, ONNX, and TensorFlow into an optimized Intermediate Representation (IR), applies FP16/INT8 quantization, and accelerates inference using the OpenVINO Runtime API.

## Installation

```bash
pip install openvino==2025.2.0 openvino-dev==2025.2.0
```

## Basic Usage

### Model Conversion and Inference

```python
import openvino as ov
import numpy as np
from pathlib import Path

# Create Core and read a model
core = ov.Core()
model = core.read_model("model.onnx")

# Compile for the available device (CPU, GPU, NPU)
compiled = core.compile_model(model, "CPU")

# Create inference request
infer_request = compiled.create_infer_request()

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_tensor = ov.Tensor(input_data)
infer_request.set_input_tensor(input_tensor)

# Run inference
infer_request.infer()
output = infer_request.get_output_tensor().data
print(f"Output shape: {output.shape}")
```

### Model Optimization (FP16)

```python
from openvino.runtime import serialize
from openvino.tools import mo

# Convert ONNX to OpenVINO IR with FP16
model = mo.convert_model("model.onnx", compress_to_fp16=True)
serialize(model, "model_fp16.xml", "model_fp16.bin")
```

## Advanced Usage / Configuration

### INT8 Quantization with Post-Training Optimization Tool (POT)

```python
from openvino.tools.pot import create_pipeline, save_model
from openvino.tools.pot.configs import QuantizationConfig

config = QuantizationConfig("model_fp16.xml", "model_fp16.bin")
config.engine.config.device = "CPU"
config.engine.config.stat_requests_number = 100

pipeline = create_pipeline(config)
quantized_model = pipeline.run()
save_model(quantized_model, "quantized", "model_int8")
```

### Multi-Device Execution

```python
# Use HETERO plugin to split layers across devices
compiled = core.compile_model(model, "HETERO:GPU,CPU")

# Or use AUTO plugin for automatic selection
compiled = core.compile_model(model, "AUTO")
```

## Integration with WFB Model Project

OpenVINO is the deployment bridge from training (PyTorch/ONNX) to Intel-based production. Export WFB models to ONNX, convert to OpenVINO IR with FP16/INT8 quantization, and deploy on edge servers with Intel CPUs or integrated GPUs. This enables low-latency, power-efficient inference in the WFB pipeline.

## Common Pitfalls / Troubleshooting

- **Model conversion fails:** Install the full `openvino-dev` package for `mo` tools. Verify ONNX opset version (prefer opset 15+).
- **GPU inference not working:** Install Intel GPU drivers. Check available devices: `core.available_devices`.
- **Quantization accuracy drop:** Use `--preset=performance` (INT8) or `--preset=mixed` (partial FP16) to balance accuracy vs speed.
- **Shape mismatch:** OpenVINO requires fixed input shapes. Use `model.reshape(partial_shape)` for dynamic shapes.

## Documentation Links

- [OpenVINO Docs](https://docs.openvino.ai/2025/)
- [Model Conversion Guide](https://docs.openvino.ai/2025/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- [Post-Training Optimization](https://docs.openvino.ai/2025/pot_introduction.html)
