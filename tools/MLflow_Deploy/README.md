# MLflow Deploy — Model Serving & Deployment

MLflow Deployment tools provide standardized interfaces for packaging, serving, and deploying ML models to production endpoints.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **pyfunc** | Python function model flavor for custom deployment |
| **mlflow.serve** | Local REST API server for a logged model |
| **Docker Deployment** | Containerized model serving |
| **SageMaker** | Deploy to AWS SageMaker endpoints |
| **Azure ML** | Deploy to Azure ML endpoints |
| **Flask** | Underlying web server for local serving |
| **Model Registry** | Versioned, stage-managed model storage |

## Logging a Model

```python
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", 0.15)
```

## Custom pyfunc Model

```python
import mlflow
from mlflow.pyfunc import PythonModel

class Predictor(PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

mlflow.pyfunc.log_model(
    "custom_model",
    python_model=Predictor(),
    artifacts={"model_path": "model.joblib"},
)
```

## Local Serving

```bash
# Serve from a run ID
mlflow models serve -m runs:/<run_id>/model -p 5001

# Serve from the Model Registry
mlflow models serve -m models:/MyModel/Production -p 5001
```

## Docker Deployment

```bash
# Build a Docker image
mlflow models build-docker -m runs:/<run_id>/model -n my-model:latest

# Run
docker run -p 5001:8080 my-model:latest
```

## Deploy to SageMaker

```python
import mlflow.sagemaker

mlflow.sagemaker.deploy(
    app_name="my-model-prod",
    model_uri="runs:/<run_id>/model",
    region_name="us-west-2",
    mode="create",
    execution_role_arn="arn:aws:iam::123456:role/SageMakerRole",
    instance_type="ml.m5.large",
    instance_count=2,
)
```

## Inference Request

```python
import requests

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    json={"inputs": [[5.1, 3.5, 1.4, 0.2]]},
)
print(response.json())
```

## Integration Patterns

- **Deploying MLflow-Registered Models**: Promote from registry staging to production
- **A/B Testing**: Deploy multiple versions behind a router
- **CI/CD Pipeline**: Automate model building, testing, and deployment
- **Containerized Serving**: Docker images for Kubernetes or ECS

## References

- [MLflow Deployment Guide](https://mlflow.org/docs/latest/deployment/index.html)
- [MLflow Pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
- [MLflow SageMaker](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html)
