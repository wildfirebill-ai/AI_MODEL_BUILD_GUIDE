# Kafka — Distributed Event Streaming Platform

Apache Kafka is a distributed event streaming platform capable of handling high-throughput, fault-tolerant, real-time data feeds.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Topic** | A category/feed name to which records are published |
| **Partition** | Ordered, immutable sequence of records within a topic |
| **Producer** | Publishes records to a topic partition |
| **Consumer** | Reads records from a topic partition |
| **Consumer Group** | Group of consumers that divide topic partitions among themselves |
| **Offset** | Unique integer identifying a record's position within a partition |
| **Broker** | A Kafka server that stores data and serves clients |
| **ZooKeeper / KRaft** | Metadata and cluster coordination service |

## Core Operations

### Producing Messages

```python
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers="localhost:9092")
producer.send("my_topic", key=b"key", value=b"hello")
producer.flush()
```

### Consuming Messages

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "my_topic",
    bootstrap_servers="localhost:9092",
    group_id="my_group",
    auto_offset_reset="earliest",
)
for msg in consumer:
    print(f"offset={msg.offset}, value={msg.value}")
```

### Exactly-Once Semantics

```python
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    enable_idempotence=True,
    acks="all",
)
```

## Integration Patterns

- **Data Pipeline Streaming**: Feed feature vectors from event streams into online feature stores
- **Log Aggregation**: Collect application logs centrally for monitoring and alerting
- **Real-Time Feature Computation**: Compute aggregations (sliding windows) over event streams for ML features
- **Model Serving Trigger**: Consume prediction requests and produce scored responses

## Configuration

```python
consumer = KafkaConsumer(
    "predictions",
    bootstrap_servers=["broker1:9092", "broker2:9092"],
    group_id="model-serving",
    enable_auto_commit=False,
    max_poll_records=100,
    session_timeout_ms=30000,
)
```

## Testing Locally

```bash
# Start Kafka with KRaft (no ZooKeeper needed)
docker run -p 9092:9092 apache/kafka:latest
```

## References

- [Official Kafka Docs](https://kafka.apache.org/documentation/)
- [kafka-python](https://github.com/dpkp/kafka-python)
- [Confluent Python Client](https://github.com/confluentinc/confluent-kafka-python)
