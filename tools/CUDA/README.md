# CUDA Toolkit

**Version:** 12.1 or 12.4 (check PyTorch compatibility at https://pytorch.org)

**Purpose:** NVIDIA's parallel computing platform and programming model. The CUDA toolkit provides `nvcc` (compiler), GPU-accelerated libraries (cuBLAS, cuRAND, cuFFT), and the runtime needed to execute kernels on NVIDIA GPUs. All deep learning frameworks depend on CUDA for GPU training and inference.

## Installation

### Verify GPU and Driver First
```powershell
nvidia-smi
```
If `nvidia-smi` fails, install the NVIDIA driver first from https://www.nvidia.com/download/index.aspx. The CUDA driver must support the toolkit version you install.

### Windows
1. Download the network or local installer from https://developer.nvidia.com/cuda-downloads
2. Run the installer (default settings recommended)
3. Verify:
```powershell
nvcc --version
```
Expected: `Cuda compilation tools, release 12.X, V12.X.XX`

### Linux (Ubuntu/Debian)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

## Environment Variables
Installer typically sets these automatically:
- `CUDA_PATH` = `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- `PATH` should include `%CUDA_PATH%\bin` and `%CUDA_PATH%\libnvvp`

If missing, add them manually.

## Version Compatibility Matrix
| PyTorch Version | Recommended CUDA | cuDNN | Compute Capability |
|----------------|-----------------|-------|-------------------|
| 2.0 – 2.1 | 11.8 | 8.7+ | 7.0+ (V100/RTX 20) |
| 2.2 – 2.5 | 12.1 | 8.9+ | 7.0+ |
| 2.6+ | 12.4 | 9.0+ | 8.0+ (A100/RTX 40) |
| Nightly | 12.8 | 9.6+ | 9.0+ (B200/RTX 50) |

Check PyTorch compatibility at https://pytorch.org/get-started/locally/

## Verify PyTorch sees CUDA
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0)}')"
```

## Advanced Usage / Configuration
- **Multiple CUDA versions:** Coexist via `CUDA_HOME` or `CUDA_PATH` — switch by updating the env var and reinstalling PyTorch
- **GPU selection:** `$env:CUDA_VISIBLE_DEVICES = "0,1"` restricts which GPUs the process sees
- **Memory management:** `torch.cuda.empty_cache()` releases PyTorch's cached allocator (call between evaluation runs, not during training)
- **Mixed precision:** Use `torch.cuda.amp.autocast()` for automatic mixed precision — halves memory usage with minimal quality loss
- **Deterministic mode:** `torch.use_deterministic_algorithms(True)` + `torch.backends.cudnn.deterministic = True` for reproducible runs (slower)

## Integration with WFB Model
All training scripts require a CUDA-capable GPU. The WFB model assumes CUDA 12.1+ and uses PyTorch's CUDA runtime. Setup order:
1. Install NVIDIA driver (latest Game Ready/Studio driver)
2. Install CUDA toolkit 12.1 or 12.4
3. Install PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
4. Verify: run `python scripts/check_cuda.py`

## Common Pitfalls / Troubleshooting
- **`nvcc` not found:** Installer did not add to PATH — add `%CUDA_PATH%\bin` manually and restart shell
- **PyTorch CUDA version mismatch:** If `torch.version.cuda` differs from `nvcc --version`, PyTorch may not use your installed CUDA — reinstall PyTorch with matching index URL
- **"No CUDA-capable device detected":** Driver issue — run `nvidia-smi` to check; update driver if needed
- **Out of memory during training:** Reduce batch size, use gradient accumulation, or enable AMP
- **Blackwell (RTX 50-series) support:** Requires CUDA 12.8+ and PyTorch nightly — standard PyTorch releases may not support Blackwell yet
- **WSL2 CUDA:** For WSL2, install the CUDA toolkit inside WSL2, **not** on Windows — use `apt install cuda-toolkit-12-4`
- **Driver toolkit mismatch:** The CUDA driver is backward-compatible — a newer driver works with older CUDA toolkits; the driver version must be >= the minimum required by the toolkit

## Documentation
- https://docs.nvidia.com/cuda/
- https://pytorch.org/get-started/locally/
