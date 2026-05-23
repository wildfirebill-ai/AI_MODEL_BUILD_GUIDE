# ONNX + TensorRT

**Purpose:** Model optimization and deployment for maximum inference performance on NVIDIA GPUs.

---

## ONNX (Open Neural Network Exchange)

### Installation

```powershell
pip install onnx onnxruntime-gpu
```

### Convert PyTorch to ONNX

```python
import torch
import onnx

dummy_input = torch.randn(1, 128, dtype=torch.long)
torch.onnx.export(
    model, dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
)
```

---

## TensorRT

### Installation

```powershell
# Download from NVIDIA: https://developer.nvidia.com/tensorrt
pip install tensorrt
```

### Build Engine

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # FP16 inference

engine = builder.build_serialized_network(network, config)
with open("model.trt", "wb") as f:
    f.write(engine)
```

### Performance Gains

| Optimization | Latency vs PyTorch Eager |
|-------------|-------------------------|
| ONNX Runtime | 1.2-1.5x faster |
| TensorRT FP16 | 2-4x faster |
| TensorRT INT8 | 4-8x faster |

## Documentation

- https://onnx.ai/
- https://developer.nvidia.com/tensorrt
