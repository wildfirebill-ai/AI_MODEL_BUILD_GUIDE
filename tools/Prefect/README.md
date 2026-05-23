# Prefect — Workflow Orchestration & Dataflow Automation

Framework for building, scheduling, and monitoring data pipelines with retries, caching, and concurrency controls.

## Installation

```bash
pip install prefect
```

## Quick Start — Flows & Tasks

```python
from prefect import flow, task

@task(retries=2, retry_delay_seconds=10)
def fetch_data(url: str) -> list:
    return [{"id": 1, "value": 100}, {"id": 2, "value": 200}]

@task(cache_key_fn=lambda x: "processed_data")
def process_data(data: list) -> list:
    return [{"id": d["id"], "value": d["value"] * 2} for d in data]

@task
def store_results(data: list) -> str:
    print(f"Storing {len(data)} records")
    return "success"

@flow(name="etl_pipeline")
def etl_pipeline(url: str = "https://api.example.com/data"):
    raw = fetch_data(url)
    processed = process_data(raw)
    return store_results(processed)

etl_pipeline()
```

## Deployments & Scheduling

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

@flow
def nightly_training():
    print("Running nightly training...")

deployment = Deployment.build_from_flow(
    flow=nightly_training,
    name="nightly-train",
    schedule=CronSchedule(cron="0 2 * * *"),
    work_queue_name="ml",
)
deployment.apply()
```

## Advanced Configuration

```python
from prefect import task, flow
from prefect.task_runners import ConcurrentTaskRunner
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    retries=3,
    retry_delay_seconds=[5, 30, 120],
    timeout_seconds=300,
    cache_expiration=timedelta(hours=1),
)
def expensive_compute(n: int) -> int:
    return n * n

@flow(task_runner=ConcurrentTaskRunner())
def parallel_pipeline():
    results = expensive_compute.map([1, 2, 3, 4, 5])
    for r in results:
        print(r.result())
```

## Concurrency Limits

```python
from prefect.concurrency import concurrency

@task
def gpu_task():
    with concurrency("gpu", occupy=1):
        print("Running on GPU...")

# CLI: prefect concurrency create gpu 2
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `@flow` | Top-level orchestrator |
| `@task` | Unit of work |
| `Deployment` | Scheduled remote execution |
| `TaskRunner` | Execution backend |
| `WorkQueue` | Priority queue for deployments |
| `ConcurrencyLimit` | Resource-based gating |
| `Retries` | Automatic failure recovery |

## Integration

Use Prefect for ML training pipelines, feature engineering, and batch inference. Deployments enable scheduled retraining. Concurrency limits prevent GPU contention. Task caching avoids recomputation.
