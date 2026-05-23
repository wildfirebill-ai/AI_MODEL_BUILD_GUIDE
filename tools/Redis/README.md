# Redis — In-Memory Data Structure Store

Redis is an open-source, in-memory data structure store used as a database, cache, message broker, and streaming engine. Redis Stack adds search, JSON, time series, and probabilistic data structures.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Key-Value** | Simple string key to any data structure value |
| **Strings** | Text, binary, or numeric values |
| **Lists** | Ordered sequences (LPUSH, RPOP) |
| **Sets** | Unordered unique elements |
| **Sorted Sets** | Ordered unique elements with scores |
| **Hashes** | Maps of field-value pairs |
| **Pub/Sub** | Publish-subscribe messaging |
| **Streams** | Append-only log for event streaming |

## Basic Operations

```python
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

# String
r.set("model:version", "v2.1")
print(r.get("model:version"))

# Hash
r.hset("model:v2", mapping={"accuracy": "0.94", "latency": "12ms"})
print(r.hgetall("model:v2"))
```

## Caching Patterns

```python
# Cache-aside (lazy population)
def get_prediction(features: list) -> dict:
    cache_key = f"pred:{hash(tuple(features))}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    result = run_model(features)
    r.setex(cache_key, 3600, json.dumps(result))  # TTL 1 hour
    return result
```

## Pub/Sub for Real-Time Predictions

```python
# Publisher
r.publish("predictions", json.dumps({"features": [1.2, 3.4]}))

# Subscriber
pubsub = r.pubsub()
pubsub.subscribe("predictions")
for msg in pubsub.listen():
    data = json.loads(msg["data"])
    print(data)
```

## Rate Limiting

```python
def check_rate_limit(user_id: str, max_requests: int, window: int) -> bool:
    key = f"ratelimit:{user_id}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window)
    return current <= max_requests
```

## Redis Stack — Vector Search (pgvector alternative)

```python
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition

r.ft("idx:vectors").create_index(
    [VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"})],
    definition=IndexDefinition(prefix=["doc:"]),
)
```

## Integration Patterns

- **LLM Response Caching**: Cache expensive LLM completions by prompt hash
- **Rate Limiting**: Per-user or per-API-key request throttling
- **Queue Backend**: Broker backend for Celery / RQ
- **Feature Store**: Low-latency read of precomputed features
- **Session Storage**: User session state for web apps

## References

- [Redis Documentation](https://redis.io/docs/)
- [Redis Stack](https://redis.io/docs/stack/)
- [redis-py](https://github.com/redis/redis-py)
