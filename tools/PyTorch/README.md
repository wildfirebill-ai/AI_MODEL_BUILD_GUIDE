# PyTorch

**Version:** 2.3+ (recommended: 2.5+)

**Purpose:** Deep learning framework — tensor computation, automatic differentiation, GPU acceleration.

## Installation

Choose the command for your system at https://pytorch.org/get-started/locally/

### CUDA 12.1
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU Only
```powershell
pip install torch torchvision torchaudio
```

## Verify Installation

```powershell
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"N/A\"}')
"
```

## Key Features Used

- `torch.nn` — neural network layers
- `torch.optim` — optimizers (AdamW)
- `torch.cuda.amp` — mixed precision training
- `torch.nn.functional` — activation functions, loss
- `torch.distributed` — multi-GPU training

## Documentation

- https://pytorch.org/docs/stable/
