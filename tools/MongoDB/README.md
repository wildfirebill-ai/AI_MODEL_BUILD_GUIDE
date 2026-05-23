# MongoDB — Document-Oriented NoSQL Database

MongoDB is a source-available, document-oriented NoSQL database designed for high performance, high availability, and easy scalability using documents with optional schemas.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Document** | BSON (Binary JSON) record — the fundamental data unit |
| **Collection** | Group of documents (analogous to a table) |
| **Index** | Optimizes query performance (single, compound, text, geospatial) |
| **Aggregation Pipeline** | Data processing pipeline with stages ($match, $group, $sort) |
| **Change Streams** | Real-time notifications on data changes |
| **Atlas** | Fully managed MongoDB cloud service |
| **Atlas Search** | Full-text search powered by Lucene |

## Basic CRUD

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client.ml_experiments
collection = db.runs

# Insert
run = {
    "model": "resnet50",
    "accuracy": 0.937,
    "params": {"lr": 0.001, "batch_size": 64},
    "metrics": {"loss": 0.21, "f1": 0.91},
}
result = collection.insert_one(run)

# Find
for doc in collection.find({"model": "resnet50"}).sort("accuracy", -1).limit(5):
    print(doc["accuracy"])
```

## Aggregation Pipeline

```python
pipeline = [
    {"$match": {"model": "resnet50"}},
    {"$group": {"_id": "$params.lr", "avg_acc": {"$avg": "$accuracy"}}},
    {"$sort": {"avg_acc": -1}},
]
results = collection.aggregate(pipeline)
```

## Indexes

```python
# Single field
collection.create_index("accuracy")

# Compound
collection.create_index([("model", 1), ("accuracy", -1)])

# Text
collection.create_index([("notes", "text")])
collection.find({"$text": {"$search": "overfitting"}})
```

## Change Streams (Real-Time)

```python
with collection.watch() as stream:
    for change in stream:
        print(change["operationType"], change["documentKey"])
```

## Atlas Search (Full-Text)

```json
{
  "$search": {
    "index": "default",
    "text": {
      "query": "transformer attention",
      "path": "description"
    }
  }
}
```

## Schema Validation

```python
db.command("collMod", "runs", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["model", "accuracy"],
        "properties": {
            "accuracy": {"bsonType": "double", "minimum": 0, "maximum": 1}
        }
    }
})
```

## Integration Patterns

- **Unstructured Data Storage**: Logs, raw experiment outputs, artifacts
- **Document Retrieval**: Flexible schema for model metadata and configs
- **Experiment Logging**: Store runs, params, metrics without fixed schema
- **Real-Time Feeds**: Change streams for reactive ML pipelines
- **Atlas Search**: Searchable model catalog / experiment database

## References

- [MongoDB Documentation](https://www.mongodb.com/docs/)
- [PyMongo](https://pymongo.readthedocs.io/)
- [MongoDB Atlas Search](https://www.mongodb.com/docs/atlas/atlas-search/)
