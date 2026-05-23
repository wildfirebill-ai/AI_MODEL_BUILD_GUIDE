# Ring Flash Attention

**Version:** 0.1+ (GitHub-based — install from source or via `pip install ring-flash-attn`)

**Purpose:** Distribute attention computation across multiple GPUs in a ring topology to support extremely long sequences (1M+ tokens). Each GPU holds a chunk of the sequence; KV chunks are passed around the ring so each GPU accumulates attention over the full sequence while only storing O(chunk_len) in memory per GPU. Essential for training and inference on long-document and multi-turn conversations.

## Installation

```powershell
pip install ring-flash-attn
```

Or install from source:
```powershell
git clone https://github.com/zhuzilin/ring-flash-attention.git
cd ring-flash-attention
pip install -e .
```

**Requirements:** PyTorch 2.1+, CUDA 12.1+, NCCL, multiple GPUs (requires `torch.distributed`). Windows support requires NCCL via torch distributed.

## Basic Usage

```python
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_func

# Initialize distributed (one process per GPU)
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each GPU holds a chunk
chunk_len = 16384  # Per GPU
num_heads = 32
head_dim = 128

q_chunk = torch.randn(1, chunk_len, num_heads, head_dim, device="cuda")
k_chunk = torch.randn(1, chunk_len, num_heads, head_dim, device="cuda")
v_chunk = torch.randn(1, chunk_len, num_heads, head_dim, device="cuda")

# Ring attention — each GPU processes full sequence
output = ring_flash_attn_func(
    q_chunk, k_chunk, v_chunk,
    causal=True,                    # Causal masking
    ring_type="basic",              # "basic" or "zigzag"
)
# output shape: (1, chunk_len, num_heads, head_dim)
```

## How It Works

1. Each GPU computes local attention on its chunk
2. KV chunks are asynchronously passed to the next GPU in the ring
3. Attention is accumulated (rescaling log-sum-exp) as each GPU receives new KV chunks
4. Memory per GPU stays O(chunk_len), not O(total_seq_len)

## Performance

| GPUs | Max Context | Memory per GPU |
|------|-------------|---------------|
| 1 | 128K | 80 GB |
| 8 | 1M | 80 GB |
| 64 | 8M | 80 GB |

## Advanced Usage / Configuration

### Zigzag vs basic ring
- **`basic`:** Simple ring — each GPU processes its own chunk then passes KV to the next
- **`zigzag`:** Load-balanced variant — every GPU gets roughly equal work regardless of causal masking; recommended for most use cases

### Combining with other attention kernels
```python
# Ring flash attention can wrap other flash attention implementations
from ring_flash_attn import ring_flash_attn_func
# Internally uses flash_attn_func for local chunk computation
```

### For training (backward pass)
The function supports `requires_grad=True` on q, k, v — the backward pass also communicates across the ring.

## Integration with WFB Model
RingFlashAttention is used in the WFB model's distributed long-context training pipeline (`scripts/long_context_train.py`). It pairs with KIVI for long-context inference:
- **Training:** RingFlashAttention for forward/backward over 1M+ token sequences across 8+ GPUs
- **Inference:** RingFlashAttention for prefill, KIVI for KV cache compression during generation
- Activated via `configs/training/long_context.yaml` with `attention_backend: "ring"`

## Common Pitfalls / Troubleshooting
- **Only works with `nccl` backend:** Ring attention requires NCCL for GPU-to-GPU communication — `gloo` and `mpi` backends won't work
- **All GPUs must be on same node** for low-latency NVLink/NVSwitch; cross-node ring has higher latency
- **CUDA graphs not supported:** The dynamic communication pattern prevents CUDA graph capture — disable CUDA graphs when using ring attention
- **Load imbalance with causal masking:** Without `zigzag` mode, GPU 0 processes the full sequence while GPU N processes only early chunks — always use `zigzag` for causal attention
- **NCCL timeout with large sequences:** Increase `NCCL_TIMEOUT_MS` or `NCCL_ASYNC_ERROR_HANDLING` environment variables
- **Numerical drift vs standard attention:** Ring flash attention accumulates with floating-point error over many chunks — use `torch.float32` accumulation if precision is critical

## Documentation
- https://github.com/zhuzilin/ring-flash-attention
- https://arxiv.org/abs/2307.05442 (Ring Attention paper)
