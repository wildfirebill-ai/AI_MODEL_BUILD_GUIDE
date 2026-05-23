# Determined AI — Deep Learning Training Platform

Determined AI simplifies distributed training, hyperparameter search, experiment management, and GPU resource allocation with fault-tolerant checkpointing.

## Key Concepts

- **Experiment** — a set of trials with shared configuration and search method
- **Trial** — a single training run with specific hyperparameters
- **Master** — central service managing scheduling and state
- **Checkpoint** — saved model state for resumption or export
- **Slot** — a unit of GPU compute (1 slot = 1 GPU)

## Experiment Configuration

```yaml
# experiment.yaml
name: resnet_cifar10
entrypoint: train:ResNetTrial

searcher:
  name: adaptive_asha
  metric: validation_error
  smaller_is_better: true
  max_length: {epochs: 100}
  max_trials: 16

hyperparameters:
  learning_rate: {type: log, minval: -5.0, maxval: -1.0, base: 10}
  momentum: {type: categorical, vals: [0.9, 0.95, 0.99]}

resources:
  slots_per_trial: 4
  max_slots: 16
```

## Defining a Trial (PyTorch)

```python
from determined.pytorch import PyTorchTrial, PyTorchTrialContext
import torch.nn as nn
import torch.optim as optim

class ResNetTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        self.context = context
        hparams = context.get_hparams()
        self.model = context.wrap_model(nn.Linear(784, 10))
        self.optimizer = context.wrap_optimizer(
            optim.SGD(self.model.parameters(), lr=hparams["learning_rate"])
        )

    def train_batch(self, batch, epoch_idx, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.model(x), y)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {"loss": loss.item()}

    def evaluate_batch(self, batch):
        x, y = batch
        acc = (self.model(x).argmax(1) == y).float().mean()
        return {"validation_error": 1 - acc.item()}

    def build_training_data_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        return DataLoader(TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,))),
                          batch_size=64)
```

## Running Experiments

```python
from determined.experimental import client

experiment = client.create_experiment(
    config="experiment.yaml",
    model_dir=".",
)
experiment.wait(max_wait_minutes=300)
best_trial = experiment.get_best_trial()
print(f"Best trial: {best_trial.id}, error: {best_trial.validation_error}")

checkpoints = best_trial.list_checkpoints()
checkpoint_path = checkpoints[0].download()
```

## Native API

```python
experiment = client.create_experiment(
    config={
        "name": "sweep",
        "entrypoint": "train:MyTrial",
        "searcher": {"name": "grid", "metric": "val_loss", "max_length": {"batches": 500}},
        "hyperparameters": {
            "lr": {"type": "categorical", "vals": [0.001, 0.01]},
        },
        "resources": {"slots_per_trial": 1},
    },
    model_dir=".",
)
```

## Integration: Submit Training Job

```python
from determined.experimental import client

config = {
    "name": "distributed_resnet",
    "entrypoint": "train:ResNetTrial",
    "searcher": {"name": "single", "metric": "validation_error",
                  "max_length": {"epochs": 50}},
    "hyperparameters": {"learning_rate": {"type": "constant", "val": 0.001}},
    "resources": {"slots_per_trial": 8},
}

exp = client.create_experiment(config=config, model_dir=".")
exp.wait()
print(f"Experiment {exp.id} completed")
```

## Best Practices

- Use `adaptive_asha` to terminate poor trials early and save GPU hours.
- Set `max_restarts` based on debugging vs production needs.
- Use the Native API for programmatic experiment creation.
- Download checkpoints for deployment — never retrain from scratch.
