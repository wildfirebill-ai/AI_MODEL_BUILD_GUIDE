# NVIDIA Drivers

**Purpose:** GPU drivers required for CUDA acceleration.

## Download

- https://www.nvidia.com/download/index.aspx
- Or use GeForce Experience: https://www.nvidia.com/geforce/geforce-experience/

## Minimum Driver Versions

| CUDA Version | Min. Driver Version |
|-------------|-------------------|
| CUDA 11.8   | 520.xx |
| CUDA 12.1   | 530.xx |
| CUDA 12.4   | 550.xx |
| CUDA 12.5+  | 555.xx |

## Check Current Driver

```powershell
nvidia-smi
```

Look for "CUDA Version" in the top-right — this shows the max CUDA version supported.

## Installation

1. Download driver for your GPU model
2. Run installer (Express Installation)
3. Reboot
4. Verify:
```powershell
nvidia-smi
```

## Recommended GPUs for AI

### Consumer Grade (GeForce RTX)

| GPU | VRAM | Compute Capability | Good For |
|-----|------|-------------------|----------|
| RTX 3060 | 12 GB | 8.6 | Small models, inference |
| RTX 3090 | 24 GB | 8.6 | 125M-1.3B training |
| RTX 4090 | 24 GB | 8.9 | 125M-7B (with QLoRA) |
| **RTX 5090** | **32 GB** | **10.0** | **125M-13B (with QLoRA)** |

### Commercial Grade (Datacenter / Workstation)

| GPU | VRAM | Compute Capability | Good For |
|-----|------|-------------------|----------|
| A6000 | 48 GB | 8.6 | 3B-13B training |
| L40S | 48 GB | 8.9 | 3B-13B inference |
| A100 | 40/80 GB | 8.0 | 7B-70B training |
| H100 | 80 GB | 9.0 | 13B-70B training |
| **B100** | **192 GB** | **10.0** | **7B-70B training** |
| **B200** | **192 GB** | **10.0** | **13B-180B training** |
| **GB200 (Grace-Blackwell)** | **192 GB/node** | **10.0** | **70B-1T+ training (72 GPUs)** |
