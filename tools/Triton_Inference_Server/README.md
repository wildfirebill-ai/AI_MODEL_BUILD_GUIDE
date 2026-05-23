# NVIDIA Triton Inference Server

**Version:** 2.50.0

## Purpose

Triton Inference Server is NVIDIA's production-grade serving solution for deploying AI models at scale. It supports multiple frameworks (ONNX, TensorRT, PyTorch, TensorFlow, vLLM) simultaneously, enabling concurrent model execution with dynamic batching, model pipelines (ensembles/BLS), and GPU/CPU scheduling. Triton is designed for high-throughput, low-latency inference in cloud and edge environments.

## Installation

```bash
# Pull the official Docker image (recommended)
docker pull nvcr.io/nvidia/tritonserver:25.04-py3

# For development via pip (Python client only)
pip install tritonclient[all]==2.50.0
```

## Basic Usage

### 1. Model Repository Structure

```
model_repository/
├── resnet50/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
├── bert/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.pt
```

### 2. Start the Server

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models
```

### 3. Python Client Inference

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("output")]

result = client.infer("resnet50", inputs, outputs=outputs)
output = result.as_numpy("output")
print(f"Output shape: {output.shape}")
```

## Advanced Usage / Configuration

### Dynamic Batching

```python
# config.pbtxt for resnet50
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32, 64]
  max_queue_delay_microseconds: 100
}
```

### Model Pipeline (Ensemble)

Chain pre-processing and inference into a single endpoint using model ensembles or the Business Logic Scripting (BLS) API for complex DAGs.

### Concurrent Model Execution

```bash
tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=resnet50 \
  --load-model=bert
```

## Integration with WFB Model Project

Triton serves as the production serving layer for WFB models. After training (PyTorch) and optimization (ONNX/TensorRT), export models into the Triton model repository. Use dynamic batching for traffic spikes and model ensembles to chain pre/post-processing. The server supports rolling updates for A/B testing new model versions without downtime.

## Common Pitfalls / Troubleshooting

- **Out of memory:** Reduce `max_batch_size` or use `--memory-limit` to cap GPU memory per model.
- **Model fails to load:** Verify the `config.pbtxt` matches the model's input/output names and data types exactly. Check logs with `docker logs <container>`.
- **Client timeout:** Increase `network_timeout` in the client: `InferenceServerClient(url=..., network_timeout=120)`.
- **Version mismatch:** Ensure `tritonclient` version matches the server image version.

## Documentation Links

- [Triton Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Client API Reference](https://github.com/triton-inference-server/client)
- [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)
