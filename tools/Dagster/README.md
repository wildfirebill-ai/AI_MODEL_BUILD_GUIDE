# Dagster — Data Orchestration for ML Pipelines

Orchestration platform built around software-defined assets with automatic dependency resolution and materialization tracking.

## Installation

```bash
pip install dagster dagit
```

## Software-Defined Assets

```python
from dagster import asset, materialize

@asset
def raw_data():
    return [{"text": "sample 1"}, {"text": "sample 2"}]

@asset
def processed_data(raw_data):
    return [{"text": d["text"].upper()} for d in raw_data]

@asset
def trained_model(processed_data):
    print(f"Training on {len(processed_data)} samples")
    return {"model_path": "/models/classifier.pkl", "accuracy": 0.95}

result = materialize([raw_data, processed_data, trained_model])
print(result.success)
```

## Ops & Jobs

```python
from dagster import op, job

@op
def load_data():
    return list(range(100))

@op
def train_model(data):
    return {"accuracy": len(data) * 0.01}

@op
def evaluate_model(model):
    return model["accuracy"]

@job
def ml_pipeline():
    evaluate_model(train_model(load_data()))

result = ml_pipeline.execute_in_process()
print(result.output_for_node("evaluate_model"))
```

## IOManager

```python
from dagster import IOManager, io_manager, Definitions

class ParquetIOManager(IOManager):
    def handle_output(self, context, obj):
        import pandas as pd
        pd.DataFrame(obj).to_parquet(f"data/{context.asset_key.path[-1]}.parquet")

    def load_input(self, context):
        import pandas as pd
        return pd.read_parquet(f"data/{context.asset_key.path[-1]}.parquet")

@io_manager
def parquet_io_manager():
    return ParquetIOManager()

defs = Definitions(assets=[raw_data, processed_data, trained_model],
                   resources={"io_manager": parquet_io_manager})
```

## Partitioning & Schedules

```python
from dagster import DailyPartitionsDefinition, asset, ScheduleDefinition, Definitions

daily = DailyPartitionsDefinition(start_date="2025-01-01")

@asset(partitions_def=daily)
def daily_model(context):
    date = context.asset_partition_key_for_output()
    return {"date": date, "model": "daily_model.pkl"}

schedule = ScheduleDefinition(job=ml_pipeline, cron_schedule="0 9 * * 1")
defs = Definitions(assets=[daily_model], schedules=[schedule])
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `@asset` | Software-defined asset |
| `@op` | Discrete computation unit |
| `@job` | Graph of ops |
| `@sensor` | Event-driven triggers |
| `IOManager` | Serialization/storage |
| `PartitionsDefinition` | Data partitioning |

## Integration

Use Dagster to orchestrate training, evaluation, and deployment pipelines. Assets track lineage. IOManagers decouple compute from storage. Sensors enable event-driven retraining.
