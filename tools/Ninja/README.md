# Ninja

**Version:** 1.11+

**Purpose:** Fast, low-level build system designed for incremental compilation. Ninja is required as the build backend for compiling CUDA extensions — FlashAttention, xFormers, DeepSpeed custom ops, and other PyTorch C++/CUDA extensions all depend on it for acceptable build times.

## Installation

```powershell
pip install ninja
```

Pin to a specific version:
```powershell
pip install ninja==1.11.1
```

### Platform-specific notes
- **Windows:** Ninja requires Visual Studio build tools. Ensure `cl.exe` is on PATH (run `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"` or use a Developer PowerShell). MSVC version must match the CUDA toolkit's supported compiler version.
- **Linux:** Works out of the box via pip. GCC 9+ recommended for CUDA kernel compilation.
- **macOS:** Requires Xcode Command Line Tools (`xcode-select --install`). Clang is used as the host compiler for CUDA.

## Verify
```powershell
python -c "import ninja; print(ninja.__version__)"
ninja --version
```

## Configuring PyTorch to Use Ninja
PyTorch's `torch.utils.cpp_extension` uses Ninja by default if installed. To force its use:
```python
import torch
from torch.utils.cpp_extension import load
# PyTorch auto-detects ninja — no special config needed
# Set USE_NINJA=0 to force disable
```
Set `$env:USE_NINJA = "0"` to disable Ninja (useful for debugging build issues line-by-line).


## Basic Usage (via PyTorch extensions)
You typically don't invoke Ninja directly — PyTorch's `torch.utils.cpp_extension` uses it behind the scenes when building CUDA extensions. Example:
```python
from torch.utils.cpp_extension import load
my_ext = load(name="my_ext", sources=["src.cpp", "src.cu"], verbose=True)
# This uses Ninja for parallel compilation
```

## Advanced Usage / Configuration
- Set `MAX_JOBS=4` environment variable to limit parallel compilation jobs (helps with OOM during builds)
- Set `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` to target specific GPU architectures (reduces build time)
- Ninja build artifacts are cached in `$HOME/.cache/torch_extensions/` — delete this folder to force a clean rebuild

## Integration with WFB Model
Ninja is a transitive dependency — installed automatically when you install FlashAttention, DeepSpeed, or any CUDA extension. If builds of these packages fail, manually installing ninja first often resolves the issue. The WFB model build pipeline (`pip install -e .`) will use Ninja for any CUDA ops.

## Common Pitfalls / Troubleshooting
- **"Unable to find ninja" during flash-attn install:** Run `pip install ninja` before `pip install flash-attn --no-build-isolation`
- **Long build times:** Check if Ninja is actually being used — set `VERBOSE=1` to see the build command; if you see `cl.exe` invocations without Ninja, something is wrong
- **Windows build failures:** Ensure `cl.exe` is accessible from the shell where you run pip — Ninja on Windows requires a detectable MSVC toolchain
- **"subprocess.CalledProcessError" in CUDA extension build:** Usually a compiler error in the .cu file — scroll up to find the actual error; try `MAX_JOBS=1` to serialize compilation for clearer error messages

## Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `MAX_JOBS` | Limit parallel compilation | `$env:MAX_JOBS = 4` |
| `VERBOSE` | Show full compiler command | `$env:VERBOSE = 1` |
| `DISTUTILS_DEBUG` | Debug setuptools build | `$env:DISTUTILS_DEBUG = 1` |
| `TORCH_CUDA_ARCH_LIST` | Target GPU architectures | `$env:TORCH_CUDA_ARCH_LIST = "8.0;9.0"` |

## Common Pitfalls / Troubleshooting
- **"Unable to find ninja" during flash-attn install:** Run `pip install ninja` before `pip install flash-attn --no-build-isolation`
- **Long build times:** Check if Ninja is actually being used — set `VERBOSE=1` to see the build command; if you see `cl.exe` invocations without Ninja, something is wrong
- **Windows build failures:** Ensure `cl.exe` is accessible from the shell where you run pip — Ninja on Windows requires a detectable MSVC toolchain
- **"subprocess.CalledProcessError" in CUDA extension build:** Usually a compiler error in the .cu file — scroll up to find the actual error; try `MAX_JOBS=1` to serialize compilation for clearer error messages
- **Build cache causing stale artifacts:** Delete `$HOME/.cache/torch_extensions/` to force a full rebuild

## Documentation
- https://ninja-build.org/
- https://github.com/ninja-build/ninja
