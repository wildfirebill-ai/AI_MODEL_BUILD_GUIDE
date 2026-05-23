# ZenML — ML Pipeline Orchestration Framework

ZenML is an open-source MLOps framework for building reproducible ML pipelines. It provides standard abstractions for infrastructure (stacks), pipeline steps (decorators), and metadata tracking.

## Key Concepts

- **Pipeline** — directed graph of steps defining an ML workflow
- **Step** — a single function decorated with `@step`
- **Stack** — collection of components (orchestrator, artifact store)
- **Materializer** — handles serialization/deserialization of step data
- **Orchestrator** — execution backend (local, Airflow, Vertex AI)

## Defining a Pipeline

```python
from zenml import pipeline, step
from typing_extensions import Annotated
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@step
def load_data() -> Annotated[pd.DataFrame, "dataset"]:
    return pd.read_csv("data.csv")

@step
def train_model(data: pd.DataFrame) -> Annotated[RandomForestClassifier, "model"]:
    X, y = data.drop("target", axis=1), data["target"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

@step
def evaluate_model(model: RandomForestClassifier,
                   data: pd.DataFrame) -> Annotated[float, "accuracy"]:
    X, y = data.drop("target", axis=1), data["target"]
    acc = accuracy_score(y, model.predict(X))
    return acc

@pipeline
def training_pipeline():
    data = load_data()
    model = train_model(data)
    evaluate_model(model, data)

if __name__ == "__main__":
    training_pipeline()
```

## Stack Configuration

```bash
# Register local stack
zenml stack register local -o default -a default

# Register GCP stack with Vertex AI orchestrator
zenml orchestrator register vertex_orc --flavor=vertex \
    --project=gcp-project --region=us-central1
zenml artifact_store register gcs --flavor=gcp \
    --path=gs://my-bucket/artifacts
zenml stack register gcp -o vertex_orc -a gcs
zenml stack set gcp
```

## Custom Materializer

```python
from zenml.materializers.base_materializer import BaseMaterializer
import torch

class TorchModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (torch.nn.Module,)

    def handle_input(self, data_type) -> torch.nn.Module:
        return torch.load(self.uri, map_location="cpu")

    def handle_output(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.uri)
```

## Scheduling

```python
from zenml.config.schedule import Schedule

schedule = Schedule(cron_expression="0 6 * * 1-5")
training_pipeline.with_config(schedule=schedule).run()
```

## Integration: End-to-End Pipeline

```python
from zenml import pipeline, step
from xgboost import XGBClassifier

@step
def prepare() -> pd.DataFrame:
    return pd.read_csv("data.csv")

@step
def train(data: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(n_estimators=200)
    model.fit(data.drop("target", axis=1), data["target"])
    return model

@step
def deploy(model: XGBClassifier) -> None:
    import joblib
    joblib.dump(model, "model/prod/model.pkl")

@pipeline
def etl_train_deploy():
    deploy(train(prepare()))

if __name__ == "__main__":
    etl_train_deploy()
```

## Best Practices

- Type-annotate step inputs/outputs for automatic materialization.
- Use stack components for environment abstraction — never hardcode paths.
- Pin ZenML and integration versions for reproducible pipeline runs.
- Separate data, training, and deployment into distinct pipeline steps.
