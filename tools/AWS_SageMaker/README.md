# AWS SageMaker — Fully Managed ML Platform

Amazon SageMaker provides integrated tooling for building, training, and deploying ML models at scale with automatic scaling.

## Key Concepts

- **Estimator** — manages training with a specific framework/image
- **HyperparameterTuner** — automated hyperparameter optimization
- **Endpoint** — hosted inference with auto-scaling
- **Processing Job** — managed data processing and feature engineering

## Setup

```python
import sagemaker

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
```

## Basic Training

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    framework_version="2.0.0",
    py_version="py310",
    hyperparameters={"epochs": 50, "batch-size": 256, "learning-rate": 0.001},
    output_path=f"s3://{bucket}/models",
    checkpoint_s3_uri=f"s3://{bucket}/checkpoints",
    use_spot_instances=True,
    max_wait=7200,
)

estimator.fit({"training": f"s3://{bucket}/data/train"})
```

## Training Script

```python
# train.py
import argparse, os, torch, torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    model = nn.Linear(784, 10)
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(args.epochs):
        pass

    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == "__main__":
    main()
```

## Distributed Training

```python
dist_estimator = PyTorch(
    entry_point="train_distributed.py",
    role=role,
    instance_count=4,
    instance_type="ml.p4d.24xlarge",
    distribution={"torch_distributed": {"enabled": True}},
    framework_version="2.0.0",
)
dist_estimator.fit({"training": f"s3://{bucket}/data"})
```

## Hyperparameter Tuning

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:accuracy",
    objective_type="Maximize",
    max_jobs=20,
    max_parallel_jobs=4,
    hyperparameter_ranges={
        "learning-rate": ContinuousParameter(0.0001, 0.1, scaling_type="Log"),
        "batch-size": CategoricalParameter([32, 64, 128, 256]),
    },
)
tuner.fit({"training": f"s3://{bucket}/data/train"},
          {"validation": f"s3://{bucket}/data/val"})
```

## Deploying

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="my-model",
)
result = predictor.predict({"features": [0.1, 0.2, 0.3]})
predictor.delete_endpoint()
```

## Batch Transform

```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/predictions",
)
transformer.transform(data=f"s3://{bucket}/data/batch_input", content_type="text/csv")
transformer.wait()
```

## Integration: Full Workflow

```python
tuner = HyperparameterTuner(
    estimator=PyTorch(entry_point="train.py", role=role,
        instance_count=1, instance_type="ml.p3.2xlarge",
        framework_version="2.0.0", py_version="py310",
        output_path=f"s3://{bucket}/models"),
    objective_metric_name="loss", objective_type="Minimize",
    max_jobs=10, max_parallel_jobs=2,
    hyperparameter_ranges={"learning-rate": ContinuousParameter(1e-5, 1e-1)})

tuner.fit({"training": f"s3://{bucket}/data"})
predictor = sagemaker.estimator.Estimator.attach(
    tuner.best_training_job()).deploy(instance_type="ml.g4dn.xlarge", initial_instance_count=1)
```

## Best Practices

- Use Spot Instances with `max_wait` to reduce costs significantly.
- Set `checkpoint_s3_uri` for fault-tolerant long training jobs.
- Use Batch Transform for large-scale offline inference instead of endpoints.
- Version models and endpoints with descriptive names and tags.
