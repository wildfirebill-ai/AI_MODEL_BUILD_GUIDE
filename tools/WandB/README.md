# Weights & Biases (WandB)

**Version:** 0.17+

**Purpose:** Experiment tracking, hyperparameter optimization, and collaboration for ML training.

## Installation

```powershell
pip install wandb
wandb login
```

## Usage

```python
import wandb

# Initialize
run = wandb.init(
    project="wfb-model",
    config={
        "learning_rate": 1e-4,
        "batch_size": 4,
        "model_size": "125M",
        "architecture": "transformer",
    },
    tags=["experiment", "baseline"],
)

# Log metrics
wandb.log({
    "loss": 2.34,
    "accuracy": 0.67,
    "learning_rate": 1e-4,
    "gradient_norm": 0.85,
    "tokens_per_second": 12500,
    "gpu_memory_gb": 18.5,
}, step=100)

# Log model graph
wandb.watch(model, log="all", log_freq=100)

# Log artifacts
wandb.log_artifact("checkpoints/step_1000", type="model")
wandb.log_artifact("data/tokenized_fineweb", type="dataset")

# Log images/tables
wandb.log({"attention_map": wandb.Image(attn_map)})
wandb.log({"predictions": wandb.Table(data=[[...]])})

# Finish
run.finish()

# Hyperparameter sweep
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [2, 4, 8]},
    },
}
sweep_id = wandb.sweep(sweep_config, project="wfb-model")
wandb.agent(sweep_id, function=train_fn, count=50)
```

## Key Features

- **Real-time dashboards** -- loss curves, GPU usage, learning rate
- **Sweeps** -- automated hyperparameter search
- **Reports** -- share results with team
- **Artifacts** -- dataset/model versioning
- **Alerts** -- email/slack on training complete or failure

## Documentation

- https://docs.wandb.ai/
