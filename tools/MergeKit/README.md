# MergeKit

**Purpose:** Merge multiple fine-tuned models into one without additional training (TIES, DARE, Linear, SLERP).

## Installation

```powershell
pip install mergekit
```

## Usage

### Configuration (YAML)

```yaml
# merge_config.yaml
slices:
  - sources:
      - model: ./model-base
        layer_range: [0, 32]
      - model: ./model-finetune-1
        layer_range: [0, 32]
merge_method: ties  # ties, dare, linear, slerp
base_model: ./model-base
parameters:
  normalize: true
dtype: bfloat16
```

### Run

```powershell
mergekit-yaml merge_config.yaml ./merged-model
```

### Python

```python
from mergekit import merge

merge(
    models=["./model-a", "./model-b"],
    method="ties",
    output="./merged-model",
)
```

## Merge Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Linear | Weighted average | Simple blending |
| SLERP | Spherical interpolation | Two models |
| TIES | Trim + Elect Sign + Merge | Multiple models, conflicting changes |
| DARE | Drop And REscale | Model soups with sparsity |

## Documentation

- https://github.com/arcee-ai/mergekit
