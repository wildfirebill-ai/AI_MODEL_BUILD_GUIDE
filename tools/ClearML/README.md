# ClearML — ML Lifecycle Management Platform

ClearML is an open-source MLOps platform for experiment tracking, data management, orchestration, and model deployment with minimal code changes.

## Key Concepts

- **Task** — a unit of work (experiment, processing, deployment)
- **Queue** — a FIFO queue for task orchestration across workers
- **Pipeline** — a DAG of tasks for automated workflows
- **StorageManager** — unified interface for data (local/S3/GCS/NFS)
- **Logger** — API for reporting scalars, plots, and debug samples

## Basic Experiment Tracking

```python
from clearml import Task, Logger

task = Task.init(project_name="my_project", task_name="rf_baseline")
task.set_parameter("n_estimators", 100)
task.set_parameter("max_depth", 10)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

logger = task.get_logger()
for epoch in range(50):
    train_loss = compute_loss(model, X_train, y_train)
    val_acc = compute_accuracy(model, X_val, y_val)
    logger.report_scalar("loss", "train", iteration=epoch, value=train_loss)
    logger.report_scalar("accuracy", "val", iteration=epoch, value=val_acc)

logger.report_matrix("confusion", "test", matrix=confusion_matrix, iteration=0)
task.upload_artifact("model", model.pkl)
task.close()
```

## Data Management

```python
from clearml import StorageManager

# Download from remote storage
local_path = StorageManager.get_local_copy("s3://my-bucket/datasets/train.parquet")

# Upload to remote storage
StorageManager.upload_file("model.pkl", "s3://my-bucket/models/rf_v1.pkl")
```

## Pipelines

```python
from clearml import PipelineController

pipe = PipelineController(project="my_project", name="training_pipeline")
pipe.add_step(name="preprocess", command="python preprocess.py", queue_name="default")
pipe.add_step(name="train", parents=["preprocess"], command="python train.py", queue_name="gpu_queue")
pipe.add_step(name="evaluate", parents=["train"], command="python evaluate.py", queue_name="default")
pipe.start(queue_name="default")
```

## Hyperparameter Optimization

```python
from clearml.automation import HyperParameterOptimizer, UniformParameterRange

optimizer = HyperParameterOptimizer(
    base_task_id="BASE_TASK_ID",
    hyper_parameters=[
        UniformParameterRange("n_estimators", min_value=50, max_value=500),
        UniformParameterRange("max_depth", min_value=3, max_value=20),
    ],
    objective_metric_title="accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=4,
    total_max_jobs=20,
)
optimizer.start_locally()
optimizer.wait()
best = optimizer.get_top_jobs(1)[0]
print(f"Best: {best.id}, score: {best.metrics['accuracy']}")
```

## Integration: Complete Training Script

```python
from clearml import Task, Logger

def main():
    task = Task.init(project_name="fraud", task_name="xgb_v2",
                     output_uri="s3://my-bucket/artifacts")
    task.set_parameter("model_type", "XGBoost")

    model = XGBClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    logger = task.get_logger()
    logger.report_scalar("accuracy", "val", iteration=0, value=accuracy)
    logger.report_histogram("feature_importance", "values",
                            values=model.feature_importances_, iteration=0)
    task.upload_artifact("model.pkl", model)
    task.close()

if __name__ == "__main__":
    main()
```

## Best Practices

- Use `task.connect(args)` to auto-log argparse parameters.
- Set `output_uri` in `Task.init` to persist artifacts to remote storage.
- Use pipelines for multi-step workflows, not monolithic tasks.
- Use tags and system tags for run organization and filtering.
