# Cross-Encoder

**Version:** Provided via `sentence-transformers` 3.0+; models are loaded from HuggingFace Hub.

**Purpose:** Rerank retrieved documents by jointly encoding query+document pairs through cross-attention. Cross-encoders achieve significantly higher relevance accuracy than bi-encoders (dense retrievers) because the query and document attend to each other. However, they are too slow for first-pass retrieval — use them to rerank the top-50 to top-100 results from a bi-encoder into a final top-5.

## Installation

```powershell
pip install sentence-transformers
```

Cross-encoder models are loaded at runtime — no additional install needed.

Pin version:
```powershell
pip install "sentence-transformers>=3.0,<4"
```

## Basic Usage

```python
from sentence_transformers import CrossEncoder

# Load model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs
pairs = [
    ["What is Python?", "Python is a programming language"],
    ["What is Python?", "Snakes are reptiles"],
]
scores = model.predict(pairs)
print(scores)  # e.g., [0.95, 0.12] — first is more relevant

# Rerank retrieved documents
def rerank(query, documents, top_k=5):
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]
```

## Advanced Usage / Configuration

### Choosing a model
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `ms-marco-MiniLM-L-6-v2` | 80 MB | Fast | Good |
| `ms-marco-MiniLM-L-12-v2` | 180 MB | Medium | Better |
| `cross-encoder/stsb-roberta-large` | 1.5 GB | Slow | Best |

### Batch inference
```python
# Batch predict for speed
all_pairs = [[query, doc] for doc in all_docs]
scores = model.predict(all_pairs, batch_size=64, show_progress_bar=True)
```

### Integration with retrieval pipelines
```python
# Hybrid retrieval: bi-encoder + cross-encoder reranker
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query_emb = bi_encoder.encode(query)
candidates = faiss_search(query_emb, top_k=100)  # Bi-encoder first pass
reranked = cross_encoder.rerank(query, candidates, top_k=10)  # Cross-encoder rerank
```

## Integration with WFB Model
Cross-encoders are used in the WFB model's RAG evaluation pipeline (`scripts/rag_eval.py`) for:
- Reranking retrieved context chunks before feeding into the LLM
- Evaluating retrieval quality with NDCG and MRR metrics using cross-encoder scores as relevance labels
- Training data filtering — scoring candidate training pairs for quality

## Common Pitfalls / Troubleshooting
- **`CUDA out of memory` with large batches:** Reduce `batch_size` (try 16 or 8); cross-encoders use attention on each pair, so memory scales with `batch_size * (query_len + doc_len)^2`
- **Slow on CPU:** Cross-encoders require GPU for acceptable speed — ensure `model.device` is `cuda`
- **Max sequence length:** Most models have a 512-token limit — truncate documents: `model.predict(pairs, truncation=True)`
- **Model not found:** HuggingFace model ID may require authentication — use `model = CrossEncoder("model-id", token=True)` if gated
- **Inconsistent scores:** Cross-encoder outputs are not calibrated — use them for ranking, not as absolute relevance probabilities; apply softmax or min-max normalize within each query's results

## Documentation
- https://www.sbert.net/docs/cross_encoder/usage/usage.html
- https://huggingface.co/cross-encoder
