# PostgreSQL — Advanced Relational Database

PostgreSQL is a powerful, open-source object-relational database system with extensibility and SQL compliance. For ML applications, it offers JSONB, full-text search, and the pgvector extension.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Table** | Structured rows and columns |
| **Index** | Accelerates query lookups (B-tree, GiST, GIN) |
| **Full-Text Search** | Text search with tsvector/tsquery |
| **JSONB** | Binary JSON for semi-structured data |
| **pgvector** | Extension for vector similarity search |
| **Extension** | Add-on modules (pgvector, PostGIS, etc.) |

## Basic SQL

```sql
CREATE TABLE experiments (
    id          SERIAL PRIMARY KEY,
    model_name  TEXT NOT NULL,
    params      JSONB,
    accuracy    FLOAT,
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO experiments (model_name, params, accuracy)
VALUES ('rf_v1', '{"n_estimators": 100, "max_depth": 10}', 0.942);
```

## Full-Text Search

```sql
SELECT * FROM documents
WHERE to_tsvector('english', body) @@ to_tsquery('english', 'machine & learning')
ORDER BY ts_rank(to_tsvector('english', body), to_tsquery('machine & learning')) DESC;
```

## pgvector — Vector Similarity Search

```sql
CREATE EXTENSION vector;

CREATE TABLE embeddings (
    id        SERIAL PRIMARY KEY,
    item_id   TEXT,
    embedding VECTOR(768)
);

-- Create index for approximate nearest neighbor
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Query
SELECT item_id, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM embeddings
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

## Indexing Strategies

```sql
-- B-tree (default, for equality/range)
CREATE INDEX idx_accuracy ON experiments (accuracy);

-- GIN (for JSONB, full-text, arrays)
CREATE INDEX idx_params ON experiments USING GIN (params);

-- GiST (for full-text, geometric)
CREATE INDEX idx_fts ON documents USING GIST (to_tsvector('english', body));
```

## Python Integration

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_metadata",
    user="ml_user",
    password="secret",
)
cur = conn.cursor()
cur.execute("SELECT model_name, accuracy FROM experiments ORDER BY accuracy DESC LIMIT 5")
rows = cur.fetchall()
```

## Integration Patterns

- **Metadata Storage**: Experiment configs, model versions, run logs
- **Feature Storage**: Structured feature tables for training/serving
- **Experiment Results**: Accuracy, loss, hyperparameter records
- **pgvector**: Embedding storage and semantic search for RAG
- **JSONB**: Flexible storage of model hyperparameters/configs

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector](https://github.com/pgvector/pgvector)
- [psycopg2](https://www.psycopg.org/docs/)
