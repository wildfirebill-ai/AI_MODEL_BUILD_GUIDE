# Celery — Distributed Task Queue

Celery is an asynchronous task queue/job queue based on distributed message passing. It is focused on real-time operation and supports scheduling.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **App** | `Celery()` instance that serves as the entry point |
| **Task** | Function decorated with `@app.task` that runs asynchronously |
| **Worker** | Process that executes tasks pulled from the broker |
| **Broker** | Message transport (Redis, RabbitMQ) that queues tasks |
| **Result Backend** | Stores task return values for retrieval |
| **Beat** | Scheduler that triggers periodic tasks |
| **Queue** | Named routing channel for task distribution |

## Basic Setup

```python
# tasks.py
from celery import Celery

app = Celery(
    "ml_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

@app.task
def predict(model_path: str, features: list) -> dict:
    # Load model and run inference
    return {"prediction": 0.95, "confidence": 0.87}
```

## Running Workers

```bash
celery -A tasks worker --loglevel=info --concurrency=4
celery -A tasks beat --loglevel=info   # for periodic tasks
```

## Calling Tasks

```python
from tasks import predict

# Async
result = predict.delay("/models/v1", [1.2, 3.4, 5.6])
print(result.get(timeout=10))  # blocks until ready

# With callback
result = predict.apply_async(
    args=["/models/v1", [1.2, 3.4, 5.6]],
    queue="ml_high_priority",
)
```

## Periodic / Scheduled Tasks

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    "retrain-every-night": {
        "task": "tasks.retrain_model",
        "schedule": crontab(hour=2, minute=0),
        "args": ("/models/prod",),
    },
}
```

## Task Chaining & Groups

```python
from celery import chain, group

# Chain: step1 -> step2 -> step3
chain(preprocess.s(data), predict.s(), postprocess.s())()

# Group: parallel tasks
group(train.s("model_a"), train.s("model_b"), train.s("model_c"))()
```

## Integration Patterns

- **Async ML Job Processing**: Offload inference requests to worker pool
- **Batch Inference**: Fan-out batch items across workers, collect results
- **Training Job Queuing**: Queue training runs with priority routing
- **Scheduled Retraining**: Beat scheduler triggers nightly retraining

## Monitoring

```bash
# Flower monitoring dashboard
celery -A tasks flower --port=5555
```

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [Celery GitHub](https://github.com/celery/celery)
