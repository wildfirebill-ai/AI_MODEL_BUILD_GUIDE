# NCCL (NVIDIA Collective Communications Library)

**Purpose:** High-performance multi-GPU and multi-node communication library for distributed training.

## Verification

```powershell
# Check NCCL version shipped with PyTorch
python -c "import torch; print(torch.cuda.nccl.version())"
```

## Environment Variables

```powershell
# Algorithm selection
$env:NCCL_ALGO = "Ring"               # Ring all-reduce (default for <=8 GPUs)
$env:NCCL_ALGO = "Tree"               # Tree all-reduce (better for >8 GPUs)
$env:NCCL_PROTO = "Simple"             # Protocol: Simple, LL, LL128
$env:NCCL_CROSS_NIC = "0"             # Disable cross-NIC (single node)
$env:NCCL_NET_GDR_LEVEL = "PHB"       # GPU Direct RDMA level
$env:NCCL_P2P_DISABLE = "0"           # Enable P2P (NVLink)
$env:NCCL_IB_DISABLE = "1"            # Disable InfiniBand if using Ethernet
$env:NCCL_SOCKET_IFNAME = "eth0"      # Network interface name
$env:NCCL_DEBUG = "WARN"              # Debug level: VERSION, INFO, WARN
$env:NCCL_DEBUG_SUBSYS = "INIT,COLL"  # Debug subsystems
$env:NCCL_TIMEOUT_MS = "600000"       # Timeout (10 min)
$env:NCCL_IGNORE_CPU_AFFINITY = "1"   # Ignore CPU affinity warnings
$env:NCCL_NTHREADS = "512"            # Communication threads
```

## Benchmark

```powershell
# Test NCCL performance
python -m torch.distributed.run --nproc_per_node=8 \
    torch.distributed.benchmark --model-type resnet50 --batch-size 64
```

## Troubleshooting

| Issue | Check |
|-------|-------|
| Slow communication | `NCCL_ALGO=Ring`, NVLink active (`nvidia-smi topo -m`) |
| Timeout | `NCCL_TIMEOUT_MS=600000`, network stability |
| Connection errors | `NCCL_SOCKET_IFNAME` set correctly, firewall |
| Out of memory (NCCL buffers) | Reduce `NCCL_NTHREADS` |

## Documentation

- https://docs.nvidia.com/deeplearning/nccl/
