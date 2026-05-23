# MLRun — ML Orchestration & Automation

Open-source MLOps framework for the end-to-end ML lifecycle: feature engineering, training, deployment, monitoring.

## Key Components

| Component | Description |
|-----------|-------------|
| `mlrun.code_to_function()` | Convert Python code to serverless functions |
| Artifacts | Track datasets, models, metrics |
| Feature Store | Centralized feature repository with real-time serving |
| Serving Graphs | Real-time inference pipelines |
| Projects | Organize workflows and resources |

## Installation

```bash
pip install mlrun
pip install mlrun[api,db,docker]  # full deps
```

## Quick Start

```python
import mlrun
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

project = mlrun.get_or_create_project("my-project", "./project", user_project=False)

@mlrun.handler(outputs=["model", "accuracy"])
def train(context, dataset, lr=0.01):
    X, y = dataset.drop("label", axis=1), dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(C=1/lr).fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    context.log_result("accuracy", acc)
    context.log_model("model", body=model, model_file="model.pkl")
    return model, acc

train_fn = project.set_function(func=train, name="train", kind="job", image="mlrun/mlrun")
```

## Running & Artifacts

```python
# Local run
import pandas as pd
df = pd.DataFrame({"x1": range(100), "x2": range(100), "label": [0, 1] * 50})
run = train_fn.run(handler="train", params={"dataset": df, "lr": 0.01}, local=True)
print(f"Accuracy: {run.outputs['accuracy']}")

# Artifact logging
@mlrun.handler(outputs=["model"])
def train_artifacts(context, dataset):
    from sklearn.ensemble import RandomForestClassifier
    X, y = dataset.drop("label", axis=1), dataset["label"]
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    context.log_model("rf_model", body=model, model_file="rf.pkl",
                       metrics={"acc": model.score(X, y)})
    context.log_dataset("training_data", dataset)
    return model
```

## Feature Store

```python
from mlrun.feature_store import FeatureSet, FeatureVector

fset = FeatureSet("transactions", entities=["customer_id"], timestamp_key="timestamp")
fset.add_feature("amount", float)
fset.add_feature("merchant_category", str)
fset.add_feature("is_fraud", int)

df = pd.DataFrame({"customer_id": [1, 1, 2], "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                    "amount": [100, 200, 50], "merchant_category": ["retail", "food", "travel"],
                    "is_fraud": [0, 0, 1]})
fset.ingest(df)

fv = FeatureVector("fraud_features", features=["transactions.amount", "transactions.merchant_category"])
print(fv.get_offline_features().to_dataframe())
```

## Serving Graph

```python
from mlrun.serving.steps import ModelStep

serving_fn = mlrun.code_to_function(name="fraud-serving", kind="serving")
graph = serving_fn.set_topology("flow", engine="sync")
graph.to(ModelStep(model_name="rf_model", model_path="store://models/rf_model"))
serving_fn.set_graph(graph)

server = serving_fn.to_mock_server()
print(server.test(body={"amount": 150, "merchant_category": "travel"}))
```

## Pipeline Orchestration

```python
pipe = project.get_pipeline("training-pipeline")
pipe.add_step("prepare", handler=prepare_data, outputs=["data"])
pipe.add_step("train", handler=train_artifacts, inputs={"dataset": "prepare.data"})
pipe.run(name="my-pipeline")
```

## Auto-Scaling

```python
serving_fn.with_auto_scaling(min_replicas=1, max_replicas=10, target_utilization=0.7)
```

## Resources

- [MLRun Docs](https://docs.mlrun.org/)
- [GitHub](https://github.com/mlrun/mlrun)
- [Tutorials](https://github.com/mlrun/tutorials)
