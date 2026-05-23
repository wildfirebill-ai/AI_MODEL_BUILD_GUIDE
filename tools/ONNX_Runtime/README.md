# ONNX Runtime — Inference Engine for ONNX Models

**Version:** 1.18.x (stable)

## Purpose

ONNX Runtime is a cross-platform inference engine for machine learning models in the ONNX format. It is distinct from ONNX_TensorRT (which focuses on model *conversion*): ONNX Runtime *optimizes and executes* ONNX graphs on CPU, GPU, and specialised accelerators. Key capabilities:

- **Multi-provider execution** — swap between CPU, CUDA, TensorRT, DirectML, OpenVINO, CoreML, and more via a unified API.
- **Graph optimizations** — automatic operator fusion, constant folding, and layout transformation.
- **Quantization** — INT8 / FP16 quantization with QAT or post-training calibration.
- **Cross-platform** — Windows, Linux, macOS, Android, iOS, and Web (WASM).
- **C, C++, Python, C#, Java, Rust, and JavaScript bindings.**

In the WFB model project, ONNX Runtime serves exported ONNX models for fast, production-grade inference (Section 46 — ONNX/TensorRT Export).

## Installation

```bash
# CPU version (stable)
pip install onnxruntime==1.18.1

# GPU version (CUDA 12.x)
pip install onnxruntime-gpu==1.18.1

# Conda
conda install -c conda-forge onnxruntime=1.18.1
conda install -c conda-forge onnxruntime-gpu=1.18.1  # GPU

# Build from source (advanced)
pip install cmake ninja
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime && ./build.sh --config Release --build_wheel --use_cuda
```

**Platform notes:**
- Windows: `onnxruntime-gpu` requires CUDA 12.x and cuDNN 8.x. Use the CUDA provider explicitly: `['CUDAExecutionProvider', 'CPUExecutionProvider']`.
- Linux: the default CUDA provider is automatically selected if CUDA is detected.
- macOS: use `onnxruntime` only; GPU support via CoreML (`CoreMLExecutionProvider`).
- Web: use `onnxruntime-web` for browser-based inference.

## Basic Usage

```python
import onnxruntime as ort
import numpy as np

# Create inference session with provider priority
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

# Prepare input (batch_size=1, seq_len=128, vocab=30522)
input_ids = np.ones((1, 128), dtype=np.int64)
attention_mask = np.ones((1, 128), dtype=np.int64)

# Run inference
outputs = session.run(
    output_names=["logits"],
    input_feed={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    },
)

print(outputs[0].shape)  # (1, 2)
```

**Quantization example:**

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8,
)
```

## Advanced Usage / Configuration

| API / Parameter | Description |
|---|---|
| `InferenceSession(model_path, providers=...)` | Load an ONNX model. Providers are tried in order; first available wins. |
| `session.run(output_names, input_feed)` | Execute the graph. `output_names` can be `None` to return all. |
| `session.get_providers()` | Query the list of available execution providers. |
| `session.get_inputs()` / `.get_outputs()` | Inspect model input/output metadata (name, shape, type). |
| `ort.SessionOptions()` | Configure session: intra/inter-op threads, graph optimization level, memory pattern. |
| `quantize_static(...)` | Post-training static quantization with a calibration dataset. |
| `quantize_dynamic(...)` | Dynamic quantization (weights only) — no calibration needed. |
| `transformers.onnx.export(...)` | Export HuggingFace models directly to ONNX (helper). |

**Session options:**

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_profiling = True

session = ort.InferenceSession("model.onnx", opts, providers=["CUDAExecutionProvider"])
```

**ORT configuration for TensorRT provider:**

```python
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_fp16_enable": True,
            "trt_int8_enable": False,
        },
    ),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

## Integration with the WFB Model Project

ONNX Runtime is the inference backend for the WFB model, corresponding to **Section 46 (ONNX/TensorRT Export)**:

1. **Model export** — after training, the model is exported to ONNX via `transformers.onnx.export()` or `torch.onnx.export()`.
2. **Inference pipeline** — the serving container loads the ONNX model with ONNX Runtime, configured for CUDA execution.
3. **Performance tuning** — `SessionOptions` are tuned for batch size, number of threads, and graph optimization level specific to WFB's production hardware.
4. **Quantization** — static quantization (INT8) is applied for latency-critical deployment paths, reducing model size ~4x with minimal accuracy loss.
5. **Fallback** — the providers list includes `CPUExecutionProvider` as the last entry so inference degrades gracefully on non-GPU nodes.

## Common Pitfalls / Troubleshooting

- **Provider not available** — verify CUDA and cuDNN versions match ORT requirements. Use `session.get_providers()` to list available providers.
- **Shape mismatches** — ONNX graphs have fixed or dynamic axes. If dynamic, pass `symbolic_dynamic` axes during export (e.g., `dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}}`).
- **GPU memory growth** — ORT allocates all GPU memory on session creation. Set `session.set_providers(...)` with `"CUDAExecutionProvider"` options like `{"cudnn_conv_algo_search": "DEFAULT"}` to control memory usage.
- **Debugging slow inference** — enable profiling with `opts.enable_profiling = True`; the profiler writes a JSON file readable by Chrome tracing (`chrome://tracing`).
- **ONNX model is too large** — apply `quantize_dynamic` or convert to FP16 with `onnxconverter_common.float16.convert_float_to_float16()`.
- **Windows CUDA DLL issues** — ensure `cudnn64_8.dll` and `cublas64_12.dll` are on `PATH`.

## Documentation Links

- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [HuggingFace ONNX Export](https://huggingface.co/docs/optimum/onnxruntime/overview)
