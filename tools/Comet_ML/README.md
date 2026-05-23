# Comet ML

**Version:** 4.0.5

## Purpose

Comet ML is an experiment tracking and model management platform for machine learning workflows. It logs hyperparameters, metrics, code versions, model artifacts, and system metrics with an organized workspace/project/experiment hierarchy. Comet also includes `CometLLM` for tracking LLM prompts/responses and `CometOptimizer` for hyperparameter tuning. It serves as an alternative or complement to Weights & Biases.

## Installation

```bash
pip install comet_ml==4.0.5
```

## Basic Usage

### Experiment Tracking

```python
import comet_ml
import numpy as np

# Initialize experiment (set COMET_API_KEY env var)
experiment = comet_ml.Experiment(
    project_name="wfb-model-training",
    workspace="wfb-team",
    auto_metric_logging=False,
)

# Log hyperparameters
params = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adamw",
}
experiment.log_parameters(params)

# Training loop
for epoch in range(50):
    train_loss = np.random.randn() * 0.1 + 1.0
    val_loss = np.random.randn() * 0.1 + 1.2

    experiment.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, step=epoch)

# Log model artifact
experiment.log_model("wfb-model", "model.pt")

experiment.end()
```

### Using CometLLM for Prompt Tracking

```python
from comet_llm import CometLLM

CometLLM.log_prompt(
    prompt="Summarize the WFB model output",
    output="The model predicts wind farm efficiency of 92.3%",
    prompt_template="Summarize the {input}",
    prompt_template_variables={"input": "WFB model output"},
    metadata={"model": "gpt-4", "temperature": 0.7},
    duration=1.2,
)
```

## Advanced Usage / Configuration

### Hyperparameter Optimization

```python
from comet_ml import Optimizer

optimizer = Optimizer(
    config={
        "algorithm": "bayes",
        "parameters": {
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
            "dropout": {"type": "uniform", "min": 0.1, "max": 0.5},
        },
        "spec": {"maxCombo": 50, "objective": "minimize", "metric": "val_loss"},
    },
    project_name="wfb-hpo",
)

for experiment in optimizer.get_experiments():
    lr = experiment.get_parameter("learning_rate")
    dropout = experiment.get_parameter("dropout")
    val_loss = run_training(lr=lr, dropout=dropout)
    experiment.log_metric("val_loss", val_loss)
```

### Offline Experiment Logging

```python
experiment = comet_ml.OfflineExperiment(
    project_name="wfb-model",
    offline_directory="./comet_offline",
)
# ... log metrics ...
experiment.end()
# Upload later: comet upload ./comet_offline
```

## Integration with WFB Model Project

Comet ML tracks all WFB training runs with hyperparameters, metrics, and model checkpoints. Use `CometLLM` to log LLM-based components (e.g., report generation from WFB predictions). The Workspace dashboard enables cross-team visibility into experiment progress and model version lineage.

## Common Pitfalls / Troubleshooting

- **API key not found:** Set `COMET_API_KEY` env var or pass `api_key="..."` to `Experiment()`. Get key from comet.com settings.
- **Upload fails offline:** Use `OfflineExperiment` and later upload with `comet upload <dir>`.
- **Too many metrics:** Disable auto-logging (`auto_metric_logging=False`) and log only relevant metrics manually.
- **Rate limiting:** For large sweeps, batch experiments or use `Optimizer` which manages API calls efficiently.

## Documentation Links

- [Comet ML Docs](https://www.comet.com/docs/v2/)
- [Experiment API](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/)
- [CometLLM Guide](https://www.comet.com/docs/v2/guides/llm-tracking/)
