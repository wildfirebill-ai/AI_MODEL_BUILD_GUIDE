# Neptune AI — Experiment Tracking and Model Registry

Neptune AI is a metadata store for MLOps that provides experiment tracking, model registry, and monitoring. It enables teams to log, compare, and organize ML experiments.

## Key Concepts

- **Run** — a single execution of an ML script
- **Project** — a namespace for organizing related runs
- **Metrics** — numeric values logged over time (loss, accuracy)
- **Parameters** — configuration values (learning rate, batch size)
- **Model Registry** — versioned storage for production models

## Basic Experiment Tracking

```python
import neptune

run = neptune.init_run(
    project="my-workspace/my-project",
    api_token="YOUR_API_TOKEN",
    name="rf-baseline",
    tags=["classification", "baseline"],
)

run["parameters"] = {"n_estimators": 100, "max_depth": 10}

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

for epoch in range(10):
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    run["metrics/train_accuracy"].append(train_acc)
    run["metrics/val_accuracy"].append(val_acc)

run["model"].upload("model.pkl")
run.stop()
```

## Logging Rich Media

```python
# Images
run["plots/feature_importance"].upload("feature_importance.png")

# Plotly charts
import plotly.express as px
fig = px.scatter(x=x, y=y)
run["plots/interactive"].upload(fig)

# Text and tables
run["config/dataset_version"] = "v3.2"
run["results/cv_scores"] = [0.85, 0.87, 0.84, 0.88]
```

## Model Registry

```python
model_version = neptune.init_model_version(
    model="YOUR_MODEL_ID",
    project="my-workspace/my-project",
    api_token="YOUR_API_TOKEN",
)
model_version["model/score"] = 0.92
model_version["model"].upload("model.pkl")
model_version.change_stage("staging")
model_version.change_stage("production")
```

## Fetching and Comparing Runs

```python
project = neptune.init_project(
    project="my-workspace/my-project",
    api_token="YOUR_API_TOKEN",
)

table = project.fetch_runs_table(
    columns=["sys/id", "parameters/lr", "metrics/val_accuracy"],
    tags=["experiment"],
).to_pandas()
print(table)
```

## Integration: Training Logger

```python
class NeptuneLogger:
    def __init__(self, project: str, api_token: str, config: dict):
        self.run = neptune.init_run(project=project, api_token=api_token)
        self.run["config"] = config

    def log_epoch(self, epoch: int, metrics: dict):
        for name, value in metrics.items():
            self.run[f"metrics/{name}"].append(value, step=epoch)

    def log_model(self, path: str, score: float):
        self.run["model"].upload(path)
        self.run["final_score"] = score

    def stop(self):
        self.run.stop()

logger = NeptuneLogger("my-workspace/my-project", "token", {"lr": 0.001})
logger.log_epoch(1, {"loss": 0.5, "acc": 0.85})
logger.log_model("model.pkl", 0.92)
logger.stop()
```

## Best Practices

- Use Neptune projects to logically separate experiment groups.
- Log hyperparameters at run creation time for better filtering.
- Upload model binaries as artifacts, not string representations.
- Use tags (`baseline`, `ablation`, `production`) for run organization.
