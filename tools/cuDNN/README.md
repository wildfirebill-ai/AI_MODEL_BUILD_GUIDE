# cuDNN (NVIDIA CUDA Deep Neural Network library)

**Version:** 8.9+ (compatible with CUDA 12.x)

**Purpose:** GPU-accelerated primitives for deep neural networks — convolution, pooling, normalization, activation, and tensor transformations. PyTorch, TensorFlow, and practically all deep learning frameworks depend on cuDNN for peak GPU performance.

## Installation

### Windows
1. Register at https://developer.nvidia.com/cudnn (free NVIDIA Developer account required)
2. Download `cudnn-windows-x86_64-<version>.zip` matching your CUDA version
3. Extract to a folder (e.g., `C:\tools\cudnn`) or copy DLLs into the CUDA toolkit directory:
   - `bin\*.dll` → `%CUDA_PATH%\bin`
   - `include\*.h` → `%CUDA_PATH%\include`
   - `lib\*.lib` → `%CUDA_PATH%\lib`
4. Add `C:\tools\cudnn\bin` to your system `PATH`

### Linux
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn-linux-x86_64-<version>.tar.xz
tar -xvf cudnn-linux-x86_64-*.tar.xz
sudo cp cudnn-*/lib/* /usr/local/cuda/lib64/
sudo cp cudnn-*/include/* /usr/local/cuda/include/
```

## Verify
```powershell
python -c "import torch; print('cuDNN enabled:', torch.backends.cudnn.enabled); print('cuDNN version:', torch.backends.cudnn.version())"
```
Expected: `enabled: True`, `version: 8900+`.

```powershell
# Check cuDNN from CUDA toolkit directly
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\demo_suite\deviceQuery.exe" | Select-String "cuDNN"
```

## Linux Verification
```bash
python -c "import torch; print(torch.backends.cudnn.version())"
# Check system-wide cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## Configuration Notes
- cuDNN auto-tunes at first run — the first few iterations may be slow as it benchmarks kernels
- `torch.backends.cudnn.benchmark = True` enables auto-tuner selection of fastest convolution algorithm (beneficial when input sizes are fixed)
- `torch.backends.cudnn.deterministic = True` disables auto-tune for reproducible results (slower but deterministic)

## Integration with WFB Model
cuDNN is consumed indirectly through PyTorch. The model build pipeline relies on cuDNN for all GPU-accelerated operations — ensure the cuDNN version matches the CUDA version PyTorch was built with. Mismatches cause silent fallback to slow reference kernels.

## Common Pitfalls / Troubleshooting
- **cuDNN not found:** PyTorch may fall back to CPU — verify with `torch.backends.cudnn.is_available()`
- **Version mismatch:** `torch.backends.cudnn.version()` returning 0 means cuDNN is not loaded — reinstall PyTorch matching your CUDA/cuDNN versions
- **Auto-tuner crashing on Windows:** Set `CUDNN_LOGDEST_DBG=stderr` for debug logs
- **Out of memory on first convolution:** cuDNN auto-tuner allocates scratch space — try `torch.backends.cudnn.benchmark = False`

## Version Compatibility Matrix
| PyTorch Version | CUDA Version | cuDNN Version |
|----------------|-------------|---------------|
| 2.0 – 2.1 | 11.8 | 8.7+ |
| 2.2 – 2.5 | 12.1 | 8.9+ |
| 2.6+ | 12.4 | 9.0+ |

Always match cuDNN to the CUDA version PyTorch was compiled with, not your driver's CUDA version.

## Integration with WFB Model
cuDNN is consumed indirectly through PyTorch. The WFB model build pipeline relies on cuDNN for all GPU-accelerated operations. The `torch.backends.cudnn` flags are configured in `wfb_model/utils/cuda.py`. Key settings:

```python
# wfb_model/utils/cuda.py
torch.backends.cudnn.benchmark = True   # Auto-tune for fixed input sizes
torch.backends.cudnn.enabled = True      # Enable cuDNN (default)
```

## Common Pitfalls / Troubleshooting
- **cuDNN not found:** PyTorch may fall back to CPU — verify with `torch.backends.cudnn.is_available()`
- **Version mismatch:** `torch.backends.cudnn.version()` returning 0 means cuDNN is not loaded — reinstall PyTorch matching your CUDA/cuDNN versions
- **Auto-tuner crashing on Windows:** Set `CUDNN_LOGDEST_DBG=stderr` for debug logs
- **Out of memory on first convolution:** cuDNN auto-tuner allocates scratch space — try `torch.backends.cudnn.benchmark = False`
- **cuDNN error "CUDNN_STATUS_NOT_INITIALIZED":** GPU driver too old — update to latest NVIDIA driver (545+ recommended)

## Documentation
- https://developer.nvidia.com/cudnn
- https://docs.nvidia.com/deeplearning/cudnn/latest/index.html
