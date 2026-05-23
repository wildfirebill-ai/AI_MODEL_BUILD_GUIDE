# Weights & Biases (WandB) + TensorBoard

**Purpose:** Experiment tracking, visualization, and logging for training runs.

---

## WandB

### Installation

```powershell
pip install wandb
```

### Setup

1. Create free account at https://wandb.ai
2. Login:
```powershell
wandb login
# Paste your API key when prompted
```

### Usage

```python
import wandb

# Initialize
wandb.init(project="wfb-model", config={
    "learning_rate": 1e-4,
    "batch_size": 4,
    "model_size": "125M",
})

# Log metrics
wandb.log({"loss": loss.item(), "lr": current_lr, "step": global_step})

# Watch model gradients
wandb.watch(model, log="all", log_freq=100)

# Finish
wandb.finish()
```

### Dashboard

View live charts at https://wandb.ai — loss curves, learning rate, gradient norms, hardware utilization.

---

## TensorBoard

### Installation

```powershell
pip install tensorboard
```

### Usage

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")
writer.add_scalar("Loss/train", loss, global_step)
writer.add_scalar("LR", lr, global_step)
writer.add_histogram("gradients", grad_norm, global_step)
writer.close()
```

### Launch

```powershell
tensorboard --logdir runs --port 6006
# Open http://localhost:6006
```

---

## Comparison

| Feature | WandB | TensorBoard |
|---------|-------|-------------|
| Hosting | Cloud (free tier) | Local |
| Collaboration | Team dashboards | Single user |
| Hardware monitoring | Built-in | Manual |
| Alerts | Yes | No |
| Open source | Partially | Yes |
