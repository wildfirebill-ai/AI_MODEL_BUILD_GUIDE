# FAISS (Facebook AI Similarity Search)

**Version:** 1.9+

**Purpose:** Efficient similarity search and clustering of dense vectors for RAG and embedding retrieval.

## Installation

```powershell
pip install faiss-gpu  # GPU version
pip install faiss-cpu  # CPU version
```

## Usage

```python
import faiss
import numpy as np

# Build index
dimension = 768
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
index.add(embeddings)  # (n_docs, dimension)

# Search
D, I = index.search(query_embedding, k=10)  # Distances, Indices

# Advanced: IVF + HNSW for billion-scale
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids=100)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # Number of clusters to search
```

## Index Types

| Index | Scalability | Speed | Accuracy |
|-------|-------------|-------|----------|
| FlatIP / FlatL2 | <1M | Slow | Perfect |
| IVF + Flat | <10M | Fast | High |
| HNSW | <100M | Very fast | Very high |
| IVF + PQ | <1B | Very fast | Medium |
| ScaNN (Google) | <1B | Fastest | High |

## Documentation

- https://github.com/facebookresearch/faiss
