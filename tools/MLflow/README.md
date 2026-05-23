# MLflow — Model Registry & Experiment Tracking

**Version:** 2.16.x / 2.17.x (stable)

## Purpose

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides:

- **Experiment Tracking** — log parameters, metrics, artifacts, and source code for every training run.
- **Model Registry** — store, version, stage (Staging / Production / Archived), and deploy models.
- **MLflow Models** — package models in a standard format (`MLmodel`) consumable by any downstream tool.
- **Deployments** — serve models via REST API, batch inference, or export to Docker / SageMaker / Azure.

In the WFB model project, MLflow is the central nervous system for experiment tracking and model versioning, feeding into the A/B testing and model registry pipeline.

## Installation

```bash
pip install mlflow==2.17.0
# Optional plugins:
pip install mlflow[extras]==2.17.0   # includes TensorFlow, PyTorch, etc.
# For conda:
conda install -c conda-forge mlflow=2.17.0
```

**Platform notes:**
- Windows: use `mlflow server` with absolute paths for `--backend-store-uri` and `--default-artifact-root`.
- Linux / macOS: default SQLite backend works out of the box.
- For multi-user setups, use a production database (PostgreSQL / MySQL) and an artifact store (S3 / GCS / Azure Blob).

## Basic Usage

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wfb-model-v2")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 3e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_metric("val_accuracy", 0.943)
    mlflow.log_artifact("confusion_matrix.png")

    # Log a transformers model (WFB's base model)
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="wfb_bert",
        registered_model_name="WFB_BERT_Base",
    )
```

**Register a model from a run:**

```python
result = mlflow.register_model(
    model_uri="runs:/<run_id>/wfb_bert",
    name="WFB_BERT_Base"
)
# Transition stage:
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="WFB_BERT_Base",
    version=result.version,
    stage="Production"
)
```

## Advanced Usage / Configuration

| Parameter / Method | Description |
|---|---|
| `mlflow.set_tracking_uri(uri)` | Set MLflow server URI (`http://`, `databricks://`, `file:`). |
| `mlflow.log_params(dict)` | Log multiple hyperparameters at once. |
| `mlflow.log_metrics(dict, step)` | Log metrics per training step (for learning curves). |
| `mlflow.transformers.log_model(...)` | Log HuggingFace `transformers` models with tokenizer. |
| `mlflow.pytorch.log_model(...)` | Log raw PyTorch models. |
| `client.transition_model_version_stage(...)` | Promote / demote model versions between stages. |
| `mlflow models serve -m "models:/WFB_BERT_Base/Production" -p 5001` | Serve a registered model as a REST endpoint. |
| `mlflow.evaluate()` | Evaluate a model on a dataset (built-in metrics). |

**Experiment tracking server:**

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://wfb-mlflow-artifacts \
    --host 0.0.0.0 --port 5000
```

## Integration with the WFB Model Project

MLflow is the central tracking and registry layer, corresponding to **Section 49 (A/B Testing & Model Registry)** of the guide:

1. **During training** — every fold / hyperparameter sweep logs params, metrics, and the model artifact to MLflow.
2. **Model Registry** — the best model is registered as `WFB_BERT_Base`, versioned, and promoted to Staging → Production.
3. **A/B Testing** — the deployment pipeline queries the registry for the Production-tagged model, while shadow-deploying the Staging version.
4. **Downstream consumers** — the inference API, batch jobs, and evaluation scripts load models via `models:/<name>/<stage>` URIs, decoupling deployment from model storage.

## Common Pitfalls / Troubleshooting

- **Tracking URI mismatch** — the training script and the model server must point to the same MLflow server. Use environment variable `MLFLOW_TRACKING_URI`.
- **Artifact store permissions** — ensure the process has read/write access to the artifact root (local dir / S3 bucket).
- **Model signature warnings** — always pass an `input_example` to `log_model()` so MLflow can infer the model signature. This is required for the Model Registry REST serving.
- **SQLite concurrency** — SQLite does not handle concurrent writes well. Use PostgreSQL for team setups.
- **Windows path separators** — use `file:///C:/path/to/mlruns` format; avoid backslashes in `--default-artifact-root`.

## Documentation Links

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Transformers Flavor](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html)
- [MLflow Models — Serving & Deployment](https://mlflow.org/docs/latest/models.html)
