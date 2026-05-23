# Kubeflow — ML Workflow Orchestration on Kubernetes

**Version:** 1.9 / SDK `kubeflow-pipeline-sdk>=2.7.0`

## Purpose

Kubeflow orchestrates end-to-end machine learning pipelines on Kubernetes. It provides components for data preprocessing, model training, hyperparameter tuning (Katib), model serving, and experiment tracking — all defined as reusable, versioned pipeline graphs that compile to Argo Workflows.

## Installation

```bash
# Pin to a known-good version
pip install kubeflow-pipeline-sdk==2.7.0
pip install kfp==2.7.0
```

Requires access to a Kubeflow-enabled Kubernetes cluster (v1.27+). For local development, install `kind` and deploy Kubeflow via `kfctl` or the official manifests.

## Basic Usage Example

```python
import kfp
from kfp import dsl

@dsl.component(base_image="python:3.11", packages_to_install=["pandas"])
def preprocess(data_path: str) -> str:
    import pandas as pd
    df = pd.read_csv(data_path)
    processed = df.dropna()
    out_path = "/tmp/processed.csv"
    processed.to_csv(out_path, index=False)
    return out_path

@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn"])
def train(data_path: str) -> str:
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    model = LogisticRegression().fit(X, y)
    model_path = "/tmp/model.pkl"
    import joblib; joblib.dump(model, model_path)
    return model_path

@dsl.pipeline(name="wfb-training-pipeline", description="WFB model training")
def wfb_pipeline(data_path: str = "gs://bucket/data.csv"):
    preprocess_task = preprocess(data_path=data_path)
    train_task = train(data=preprocess_task.output)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(wfb_pipeline, "pipeline.yaml")
    client = kfp.Client(host="https://your-kubeflow-host.com")
    client.create_run_from_pipeline_func(
        wfb_pipeline, arguments={"data_path": "gs://bucket/data.csv"}
    )
```

## Advanced Usage / Configuration

- **Katib hyperparameter tuning**: Define a `Experiment` spec with search space (uniform, random, grid) and objective metric. Kubeflow spawns trial jobs in parallel.
- **Cache and reuse**: `@dsl.component` outputs are automatically cached by hash; set `dsl.CachingMetrics` to control invalidation.
- **Pipeline versions and experiments**: Group runs under `client.create_experiment()` for clean lineage.
- **Resource constraints**: Attach `dsl.ResourceSpec(cpu=4, memory="16Gi", accelerator="nvidia.com/gpu", accelerator_count=1)` to components.
- **Conditional execution**: Use `with dsl.Condition(...)` to branch on upstream outputs.

## Integration with the WFB Model Project

```python
# Place in wfb_model/pipelines/kubeflow/pipeline.py
@dsl.component(base_image="wfb-base:latest")
def train_wfb(epochs: int = 10, lr: float = 1e-4) -> str:
    # Imports and trains the WFB model defined in wfb_model/train.py
    return "gs://models/wfb/run-001"
```

The WFB project stores pipeline definitions in `pipelines/`, shares artifacts via MinIO (S3-compatible), and registers tuned hyperparameters in Katib's experiment DB. All training logs flow to Kubeflow's TensorBoard integration.

## Common Pitfalls / Troubleshooting

- **Out-of-memory (OOM)**: GPU pods often OOM when `accelerator_count` is omitted. Always set `accelerator_count=1` for GPU components.
- **Pipeline compilation fails**: Ensure `kfp==2.7.0` is installed; v1 SDK functions like `kfp.dsl.PipelineParam` are removed in v2.
- **Katib trials stuck**: Check the experiment's `maxTrialCount` and `parallelTrialCount`. Set `metricsCollectorSpec` to `StdOut` collector for debugging.
- **Artifact passing failures**: v2 uses MLMD-based artifact passing; ensure all component outputs are typed (string paths) and file-based.
- **Authentication errors**: Use `kfp.Client(host=..., cookies="auth=...")` or configure an IAP/OIDC proxy for cloud deployments.

## Documentation Links

- Official Kubeflow docs: https://www.kubeflow.org/docs/
- Kubeflow Pipelines v2 SDK reference: https://kubeflow-pipelines.readthedocs.io/
- Katib: https://www.kubeflow.org/docs/components/katib/
- Pipeline examples: https://github.com/kubeflow/pipelines/tree/master/samples
