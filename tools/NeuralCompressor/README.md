# Intel Neural Compressor

**Version:** 3.0.1

## Purpose

Intel Neural Compressor (INC) provides automated model compression through quantization, pruning, and knowledge distillation. It supports post-training quantization (PTQ) and quantization-aware training (QAT) with automatic precision recipe tuning. INC integrates with PyTorch, ONNX, TensorFlow, and OpenVINO, applying INT8, INT4, and mixed-precision optimizations without requiring manual tuning.

## Installation

```bash
pip install neural-compressor==3.0.1
```

## Basic Usage

### Post-Training Quantization (PTQ)

```python
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import quantize_model
from neural_compressor.data import Datasets, DataLoader
import torch

# Define a simple calibration dataloader
dataset = Datasets("pytorch")["dummy"](shape=(64, 3, 224, 224), label_shape=(64,))
dataloader = DataLoader(framework="pytorch", dataset=dataset)

# Configure PTQ
config = PostTrainingQuantConfig(
    approach="static",
    backend="default",
    calibration_sampling_size=[100],
)

# Quantize the model
model = torch.load("model.pt")
quantized_model = quantize_model(model, config, calib_dataloader=dataloader)

# Save the quantized model
quantized_model.save("quantized_model.pt")
```

### Quantization-Aware Training (QAT)

```python
from neural_compressor.training import prepare_compression
from neural_compressor.config import QuantizationAwareTrainingConfig

config = QuantizationAwareTrainingConfig()
compression = prepare_compression(model, config)
compression.callbacks.on_train_begin()

for epoch in range(num_epochs):
    compression.callbacks.on_epoch_begin(epoch)
    for batch in dataloader:
        compression.callbacks.on_batch_begin()
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        compression.callbacks.on_step_end()
    compression.callbacks.on_epoch_end()

compression.callbacks.on_train_end()
quantized_model = compression.model
```

## Advanced Usage / Configuration

### Auto-Tuning Precision Recipes

```python
from neural_compressor.config import TuningConfig
from neural_compressor.quantization import fit

tuning_config = TuningConfig(
    accuracy_criterion={"relative": 0.01},
    objective="performance",
    strategy="bayesian",
    timeout=3600,
)

tuned_model = fit(
    model=model,
    conf=tuning_config,
    calib_dataloader=dataloader,
    eval_func=eval_fn,
)
```

### Pruning

```python
from neural_compressor.config import WeightPruningConfig

pruning_config = WeightPruningConfig(
    pruning_type="snip_momentum",
    target_sparsity=0.5,
    start_epoch=0,
    end_epoch=5,
)
```

## Integration with WFB Model Project

Neural Compressor optimizes WFB models for deployment on resource-constrained Intel hardware. Apply PTQ after ONNX export to shrink model size by 4x with INT8. For accuracy-critical WFB components, use QAT during fine-tuning. Auto-tuning recipes find the optimal speed/accuracy trade-off without manual iteration.

## Common Pitfalls / Troubleshooting

- **Quantization degrades accuracy >1%:** Switch from PTQ to QAT or enable `approach="static"` with more calibration samples (500+).
- **Unsupported ops:** Use `op_type_dict` in config to exclude problematic ops from quantization.
- **Slow calibration:** Reduce `calibration_sampling_size` or use `approach="dynamic"` for faster but less optimized quantization.
- **Framework mismatch:** Ensure `backend` matches the model framework (e.g., `"pytorch"`, `"onnxrt"`).

## Documentation Links

- [Neural Compressor Docs](https://intel.github.io/neural-compressor/)
- [Quantization Guide](https://intel.github.io/neural-compressor/latest/docs/quantization.html)
- [API Reference](https://intel.github.io/neural-compressor/latest/docs/api.html)
