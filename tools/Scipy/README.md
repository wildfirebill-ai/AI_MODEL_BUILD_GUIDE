# SciPy

**Version:** 1.13+

**Purpose:** Scientific computing library built on NumPy. Statistics, optimization, signal processing.

## Installation

```powershell
pip install scipy
```

## Usage in LLM Workflows

```python
from scipy import stats
from scipy.spatial.distance import cosine, cdist
import numpy as np

# t-test for A/B testing
t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
print(f"Statistically significant: {p_value < 0.05} (p={p_value:.4f})")

# Cosine distance between embedding vectors
distance = cosine(embedding_a, embedding_b)
similarity = 1 - distance

# Pairwise distance matrix
dist_matrix = cdist(all_embeddings, all_embeddings, metric="cosine")

# Sparse matrix operations
from scipy.sparse import csr_matrix
sparse_attention = csr_matrix(attention_weights)  # Memory efficient

# Interpolation for LR scheduling
from scipy.interpolate import interp1d
scheduler = interp1d([0, 0.1, 1.0], [0, lr_max, lr_min * 0.1])
```

## Key Modules

| Module | Use Case |
|--------|----------|
| `scipy.stats` | Hypothesis testing, distributions |
| `scipy.spatial.distance` | Distance metrics (cosine, euclidean) |
| `scipy.sparse` | Sparse matrices for attention |
| `scipy.interpolate` | Smooth learning rate schedules |
| `scipy.cluster` | Hierarchical clustering |
| `scipy.special` | Special functions (softmax, logsumexp) |

## Documentation

- https://docs.scipy.org/doc/scipy/
