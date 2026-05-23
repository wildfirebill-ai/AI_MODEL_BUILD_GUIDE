# Haystack — NLP Framework for RAG & Search

End-to-end framework for retrieval-augmented generation pipelines.

## Installation

```bash
pip install haystack-ai
```

## Quick Start — Document Store & Indexing

```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

doc_store = InMemoryDocumentStore()

docs = [
    Document(content="Haystack is an NLP framework for building RAG systems."),
    Document(content="It supports multiple document stores and retrievers."),
]

embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()
docs_with_emb = embedder.run(docs)
doc_store.write_documents(docs_with_emb["documents"])
```

## Building a RAG Pipeline

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

prompt_template = """
Documents:
{% for doc in documents %}{{ doc.content }}{% endfor %}
Question: {{ question }}
Answer:
"""

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=doc_store))
pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("generator", OpenAIGenerator())
pipeline.connect("embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "generator.prompt")

result = pipeline.run({"embedder": {"text": "What is Haystack?"}, "prompt_builder": {"question": "What is Haystack?"}})
print(result["generator"]["replies"][0])
```

## Indexing Pipeline

```python
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

indexing = Pipeline()
indexing.add_component("splitter", DocumentSplitter(split_by="word", split_length=500, split_overlap=50))
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store=doc_store))
indexing.connect("splitter.documents", "embedder.documents")
indexing.connect("embedder.documents", "writer.documents")
```

## Different Retrievers

```python
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

bm25 = InMemoryBM25Retriever(document_store=doc_store)
results = bm25.run(query="NLP framework")
print(results["documents"])
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `InMemoryDocumentStore` | Lightweight storage |
| `EmbeddingRetriever` | Dense retrieval |
| `BM25Retriever` | Keyword retrieval |
| `PromptBuilder` | Prompt construction |
| `Generator` | LLM response |
| `DocumentSplitter` | Chunk documents |

## Integration

Pipelines are directed graphs using `Pipeline.connect()`. Swap `InMemoryDocumentStore` for Qdrant, Weaviate, or PGVector in production.
