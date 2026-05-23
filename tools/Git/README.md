# Git

**Version:** 2.45+

**Purpose:** Version control for tracking changes to model code, configurations, and experiments.

## Download

- https://git-scm.com/downloads

## Installation

1. Download and run installer
2. Use default options (recommended)
3. Verify:
```powershell
git --version
```

## Basic Workflow

```powershell
# Initialize
git init
git add .
git commit -m "Initial commit: model architecture and training scripts"

# Track experiments
git add .
git commit -m "Experiment: 125M training run #5 - lr=3e-4, bs=8"

# Branch for experiments
git checkout -b exp/qlora-7b
```

## .gitignore for ML Projects

```
venv/
__pycache__/
*.pyc
.DS_Store
data/
checkpoints/
*.gguf
wandb/
runs/
logs/
*.log
```

## Large File Storage (LFS)

For model checkpoints >100MB:
```powershell
git lfs install
git lfs track "*.gguf"
git lfs track "*.bin"
git lfs track "*.safetensors"
```

## Documentation

- https://git-scm.com/doc
