# Sentence-Transformers

**Version:** 3.0+

**Purpose:** Compute dense vector embeddings for sentences, paragraphs, and documents.

## Installation

```powershell
pip install sentence-transformers
```

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = model.encode([
    "This is a sentence",
    "Another sentence here",
])
# embeddings.shape == (2, 1024)

# Semantic similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
```

## Training Custom Embeddings

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train_data = [InputExample(texts=["query", "positive_doc"], label=1.0)]
loader = NoDuplicatesDataLoader(train_data, batch_size=32)
model.fit(train_objectives=[(loader, losses.CoSENTLoss(model))], epochs=10)
```

## Popular Models

| Model | Dim | Use Case |
|-------|-----|----------|
| BAAI/bge-large-en-v1.5 | 1024 | General retrieval |
| intfloat/e5-mistral-7b-instruct | 4096 | High-quality |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Speed-critical |

## Documentation

- https://www.sbert.net/
