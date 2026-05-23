# datasketch

**Purpose:** Probabilistic data structures for efficient near-duplicate detection (MinHash LSH) at scale.

## Installation

```powershell
pip install datasketch
```

## Usage for Deduplication

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, k=13):
    """Character n-grams"""
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def compute_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for shingle in shingles(text):
        m.update(shingle.encode())
    return m

# Build LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=128)
deduped_data = []

for i, doc in enumerate(dataset):
    m = compute_minhash(doc["text"])

    # Check for near-duplicates
    if not lsh.query(m):
        lsh.insert(f"doc_{i}", m)
        deduped_data.append(doc)

print(f"Deduped: {len(dataset)} -> {len(deduped_data)} ({len(dataset)-len(deduped_data)} removed)")
```

## Key Features

| Data Structure | Purpose | Accuracy |
|---------------|---------|----------|
| MinHash | Set similarity estimation | Configurable (more perms = higher) |
| MinHashLSH | Near-duplicate detection | Threshold + false positive rate |
| HyperLogLog | Cardinality estimation | ~2% error |
| Weighted MinHash | Weighted set similarity | For weighted features |

## Configuration

- `num_perm=128` -- higher = more accurate but slower
- `threshold=0.8` -- Jaccard similarity threshold for "duplicate"
- `weights=(0.5, 0.5)` -- balance false positives/negatives

## Documentation

- https://github.com/ekzhu/datasketch
