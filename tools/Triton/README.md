# OpenAI Triton

**Version:** 3.0+

**Purpose:** GPU kernel programming language for writing custom CUDA kernels in Python.

## Installation

```powershell
pip install triton
```

## Usage

```python
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_kernel(x_ptr, output_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    output = x / tl.sqrt(mean_sq + eps)
    tl.store(output_ptr + row * n_cols + cols, output, mask=mask)

def fused_rmsnorm(x, eps=1e-5):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    fused_rmsnorm_kernel[grid](x, output, n_cols, eps, BLOCK_SIZE=1024)
    return output
```

## Key Features

- **2-10x speedup** over naive PyTorch
- **Automatic tiling** and memory coalescing
- **torch.compile backend** uses Triton internally
- **Custom attention kernels** beyond FlashAttention

## Documentation

- https://triton-lang.org/
