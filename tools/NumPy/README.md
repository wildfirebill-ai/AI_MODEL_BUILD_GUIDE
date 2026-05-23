# NumPy

**Version:** 1.26+

**Purpose:** Fundamental numerical computing library for Python. Foundation for all ML frameworks.

## Installation

```powershell
pip install numpy
```

## Key Usage in LLM Workflows

```python
import numpy as np

# Embedding storage and similarity search
embeddings = np.array(model_outputs)  # (n_docs, dim)
similarities = embeddings @ embeddings.T  # Cosine similarity

# Statistical analysis of training metrics
losses = np.array(training_losses)
print(f"Mean loss: {np.mean(losses):.4f}")
print(f"Std loss: {np.std(losses):.4f}")
print(f"Min loss: {np.min(losses):.4f}")
print(f"P95 loss: {np.percentile(losses, 95):.4f}")

# Anomaly detection (IQR)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

# Random sampling
indices = np.random.choice(len(dataset), size=1000, replace=False)
```

## Key Functions

| Function | Use |
|----------|-----|
| `np.mean`, `np.std`, `np.var` | Statistics |
| `np.percentile`, `np.median` | Quantiles |
| `np.random.choice`, `np.random.permutation` | Sampling |
| `np.linalg.norm`, `np.dot`, `np.matmul` | Linear algebra |
| `np.save`, `np.load` | Serialization |
| `np.concatenate`, `np.stack` | Array operations |
| `np.where`, `np.clip` | Conditional operations |
| `np.argmax`, `np.argsort` | Index operations |

## Documentation

- https://numpy.org/doc/
