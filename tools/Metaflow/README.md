# Metaflow — ML Infrastructure Framework by Netflix

Metaflow provides a simple Python API for building, deploying, and managing ML workflows with seamless scaling from local development to cloud compute.

## Key Concepts

- **FlowSpec** — base class for defining a Metaflow pipeline
- **`@step`** — decorator marking a phase of the workflow
- **`@batch`** — decorator to run step on cloud batch compute
- **`@resources`** — decorator to request CPU, memory, and GPU
- **`@conda`** — decorator to specify conda environments

## Defining a Flow

```python
from metaflow import FlowSpec, step, resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

class TrainingFlow(FlowSpec):

    @step
    def start(self):
        self.data = pd.read_csv("iris.csv")
        self.X = self.data.drop("species", axis=1)
        self.y = self.data["species"]
        self.next(self.train)

    @resources(memory=4096, cpu=4)
    @step
    def train(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(self.X, self.y)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        self.accuracy = accuracy_score(self.y, self.model.predict(self.X))
        self.next(self.end)

    @step
    def end(self):
        print(f"Accuracy: {self.accuracy:.4f}")

if __name__ == "__main__":
    TrainingFlow()
```

## Running

```bash
python training_flow.py run
python training_flow.py resume
python training_flow.py run --with batch
```

## Cloud Compute with `@batch`

```python
from metaflow import FlowSpec, step, batch, resources

class CloudFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.train_gpu, self.train_cpu)

    @batch(queue="gpu-queue", image="docker.io/pytorch/pytorch:latest")
    @resources(gpu=4, memory=64000, cpu=16)
    @step
    def train_gpu(self):
        self.gpu_result = "trained on A100"
        self.next(self.join)

    @batch(queue="cpu-queue")
    @step
    def train_cpu(self):
        self.cpu_result = "trained on c5.xlarge"
        self.next(self.join)

    @step
    def join(self, inputs):
        self.gpu_result = inputs.train_gpu.gpu_result
        self.cpu_result = inputs.train_cpu.cpu_result
        self.next(self.end)

    @step
    def end(self):
        print(f"GPU: {self.gpu_result}, CPU: {self.cpu_result}")
```

## Foreach & Inspecting Runs

```python
from metaflow import Flow, Run

flow = Flow("TrainingFlow")
for run in flow:
    print(f"Run {run.id}: success={run.successful}")

run = Run("TrainingFlow/42")
print(f"Accuracy: {run.data.accuracy}")
```

## Integration: Linear Pipeline

```python
from metaflow import FlowSpec, step
from xgboost import XGBClassifier
import joblib

class ProdPipeline(FlowSpec):

    @step
    def start(self):
        self.df = pd.read_parquet("s3://bucket/features.parquet")
        self.next(self.split)

    @step
    def split(self):
        from sklearn.model_selection import train_test_split
        self.X_train, _, self.y_train, _ = train_test_split(
            self.df.drop("target", axis=1), self.df["target"])
        self.next(self.train)

    @step
    def train(self):
        self.model = XGBClassifier().fit(self.X_train, self.y_train)
        self.next(self.deploy)

    @step
    def deploy(self):
        joblib.dump(self.model, "model/prod/xgb.pkl")
        self.next(self.end)
```

## Best Practices

- Use `foreach` for hyperparameter search — Metaflow handles parallelism.
- Keep steps small for better caching and resume behavior.
- Use `@conda` to pin dependency versions for reproducibility.
- Use `@batch` with `@resources` to request cloud resources declaratively.
