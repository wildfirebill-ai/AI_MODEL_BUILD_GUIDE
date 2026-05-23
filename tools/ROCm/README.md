# ROCm — AMD GPU Computing Platform for ML

[ROCm](https://rocm.docs.amd.com) (Radeon Open Compute) is AMD's open-source GPU computing platform. It provides HIP, rocBLAS, MIOpen, RCCL, and PyTorch/TensorFlow support for AMD GPUs.

## Installation

```bash
# Windows: Install ROCm for Windows via AMD installer
# Or use Docker
docker pull rocm/pytorch:latest

# Linux (Ubuntu)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install.deb
sudo dpkg -i amdgpu-install.deb
sudo amdgpu-install --usecase=rocm
```

Verify:

```bash
rocm-smi
hipconfig --full
```

## ROCm Stack Overview

| Component  | Purpose                        |
|------------|--------------------------------|
| **HIP**    | CUDA-compatible runtime API    |
| **rocBLAS**| BLAS on AMD GPUs               |
| **MIOpen** | Deep learning primitives       |
| **RCCL**   | Collective communication (like NCCL) |
| **rocFFT** | FFT library                    |

## Hipify — Converting CUDA to HIP

```bash
# Convert CUDA source files to HIP automatically
hipify-perl cuda_kernel.cu > hip_kernel.cpp

# Or use hipify-clang
hipify-clang cuda_kernel.cu --output-dir=./hip
```

Before (CUDA):

```cpp
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

After (HIP — identical API):

```cpp
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

## PyTorch on ROCm

```python
import torch

# Check if ROCm is available
print(torch.cuda.is_available())        # True if ROCm is installed
print(torch.version.hip)                # HIP version string
print(torch.cuda.get_device_name(0))    # e.g., "AMD Instinct MI250"

# Use hip instead of cuda under the hood
device = torch.device("cuda")           # same torch.cuda API
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.mm(x, y)
print(z)
```

Note: `torch.cuda` works on ROCm. Replace `nvcc` builds with `hipcc`.

## MIOpen — Deep Learning Primitives

```python
import torch
# MIOpen is the default backend on AMD GPUs
# No code changes needed — PyTorch uses MIOpen automatically

# Force MIOpen benchmarks
torch.backends.cudnn.benchmark = True   # maps to MIOpen on AMD
```

## Distributed Training with RCCL

```python
import torch.distributed as dist
import os

# RCCL is the default backend on AMD GPUs
dist.init_process_group(
    backend="nccl",          # maps to RCCL on AMD
    init_method="env://",
    rank=int(os.environ["RANK"]),
    world_size=int(os.environ["WORLD_SIZE"]),
)

model = torch.nn.Linear(100, 10).cuda()
model = torch.nn.parallel.DistributedDataParallel(model)
```

## TensorFlow on ROCm

```bash
pip install tensorflow-rocm
```

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# Build and train models normally — ROCm handles the GPU backend
```

## Memory Management

```python
import torch

# ROCm supports unified memory
x = torch.empty(1000, device="cuda")
torch.cuda.empty_cache()     # same API as CUDA
print(torch.cuda.memory_summary())
```

## Supported AMD GPUs

- **Instinct MI250** (dual-die, 128GB HBM2e)
- **Instinct MI300X** (192GB HBM3, CDNA3)
- **Radeon RX 7900 XTX** (consumer, ROCm support)
- **Radeon Pro W7900** (workstation)

Check compatibility: `rocm-smi --showid`

## Use Cases

- **Training on AMD GPUs** — full ML pipeline with PyTorch/TF
- **Porting CUDA code** — hipify for automatic conversion
- **Distributed training** — RCCL scales across nodes
- **HPC inference** — serve models on AMD hardware at scale
