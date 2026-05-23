# GCP Vertex AI — Managed ML Platform on Google Cloud

Vertex AI provides unified tooling for AutoML, custom training (with TPU support), hyperparameter tuning, model registry, and online/batch prediction endpoints.

## Custom Training Job

```python
import google.cloud.aiplatform as aiplatform
aiplatform.init(project="my-project", location="us-central1",
                staging_bucket="gs://my-bucket/staging")

train_job = aiplatform.CustomTrainingJob(

```python
train_job = aiplatform.CustomTrainingJob(
    display_name="pytorch-training",
    script_path="trainer.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    requirements=["torch==2.0.1"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
)

model = train_job.run(
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=2,
    args=["--epochs=100", "--learning-rate=0.001"],
    base_output_dir="gs://my-bucket/models",
)
```

## Training Script

```python
# trainer.py
import argparse, os, torch, torch.nn as nn

AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    model = nn.Linear(784, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for _ in range(args.epochs):
        pass
    torch.save(model.state_dict(), os.path.join(AIP_MODEL_DIR, "model.pth"))

if __name__ == "__main__":
    main()
```

## Distributed Training with TPU

```python
model = train_job.run(
    replica_count=1,
    machine_type="ct4p-hightpu-1t",
    accelerator_type="TPU_V4",
    accelerator_count=8,
    args=["--epochs=200", "--use-tpu"],
)
```

## Hyperparameter Tuning

```python
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="resnet-hpo",
    custom_job=train_job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning-rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=1e-5, max=1e-1, scale="log"),
        "batch-size": aiplatform.hyperparameter_tuning.CategoricalParameterSpec(
            values=[32, 64, 128]),
    },
    max_trial_count=30,
    parallel_trial_count=4,
)
hp_job.run(sync=True)
```

## Deploying

```python
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5,
)

response = endpoint.predict(instances=[[0.1] * 784])
print(response.predictions)

batch_job = model.batch_predict(
    job_display_name="batch-inference",
    machine_type="n1-standard-4",
    gcs_source=["gs://my-bucket/input/instances.jsonl"],
    gcs_destination_prefix="gs://my-bucket/predictions/",
)
batch_job.wait()
```

## Model Registry

```python
model = aiplatform.Model.upload(
    display_name="resnet_v2",
    artifact_uri="gs://my-bucket/models/resnet_v2/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
)

model_v2 = aiplatform.Model(
    model_name="projects/my-project/locations/us-central1/models/12345@2")
model_v2.deploy(machine_type="n1-standard-4", min_replica_count=1)
```

## Integration: End-to-End

```python
job = aiplatform.CustomTrainingJob(
    display_name="xgboost-train",
    script_path="xgboost_train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-3:latest",
)
model = job.run(machine_type="n1-highmem-8", args=["--n-estimators=500"],
                base_output_dir="gs://my-bucket/xgboost")
endpoint = model.deploy(machine_type="n1-standard-4", min_replica_count=1)
print(endpoint.resource_name)
```

## Best Practices

- Use TPU VMs for large-scale transformer training — massively faster than GPUs for suitable models.
- Set `sync=False` for non-blocking training job submission in scripts.
- Use the Model Registry with version aliases for deployment tracking.
- Monitor endpoint latency via Cloud Monitoring dashboards.
