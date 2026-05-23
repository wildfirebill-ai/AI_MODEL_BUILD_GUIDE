# NVIDIA Nsight Systems — GPU Performance Profiling

**Version:** 2024.5 (latest stable)

## Purpose

NVIDIA Nsight Systems (nsys) is a system-wide performance analysis tool for visualizing and optimising application performance on NVIDIA GPUs. It provides:

- **Timeline view** — see CPU and GPU activity on a unified timeline, including kernel launches, memory copies, CUDA API calls, and CPU-side computation.
- **Kernel analysis** — identify GPU kernel launch configurations, occupancy, memory throughput, and compute utilisation.
- **CUDA API tracing** — trace every `cudaMemcpy`, `cudaLaunchKernel`, `cudaStreamSynchronize`, etc., with precise timestamps.
- **OS and runtime tracing** — thread scheduling, I/O, page faults, and Python / PyTorch operations.
- **GUI and CLI** — `nsys-ui` for interactive analysis and `nsys profile` for command-line capture.
- **Export** — save reports (.nsys-rep) for sharing and comparison. Export to text, JSON, or SQLite for automated analysis.

In the WFB model project, Nsight Systems is the primary tool for profiling training and inference performance, identifying GPU bottlenecks, and guiding optimisation decisions (Section 45 — Monitoring).

## Installation

1. Download from [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (free registration required).
2. Installer available for Windows, Linux (x86_64, ARM64), and macOS (host-only, can profile remote Linux targets).

**Linux (Ubuntu/Debian):**

```bash
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2024.5_2024.5.1.112-1_amd64.deb
sudo dpkg -i nsight-systems-2024.5_2024.5.1.112-1_amd64.deb
sudo apt-get install -f
```

**Windows:** Download and run the `.exe` installer. Add `C:\Program Files\NVIDIA Corporation\Nsight Systems <version>\targets\<platform>\` to PATH.

**Verify installation:**

```bash
nsys --version
nsys-ui  # launches GUI (if installed)
```

**Platform notes:**
- On Linux, `nsys profile` requires `perf_event_paranoid` to be ≤ 2 (or run as root/sudo).
- On Windows, Nsight Systems works with WSL2 for Linux profiling; native Windows profiling is also supported (CLI only, no GUI on Windows).
- For containerised environments, run with `--cap-add SYS_ADMIN` or use the NVIDIA container toolkit.

## Basic Usage

```bash
# Profile a training script
nsys profile -o training_report python src/train.py --epochs 2

# Profile with GPU and CUDA tracing
nsys profile -t cuda,nvtx,osrt -o inference_profile python src/run_inference.py

# Profile a specific duration (first 30 seconds)
nsys profile -d 30 -o quick_profile python src/train.py

# Capture OS runtime and GPU metrics
nsys profile --gpu-metrics-device=0 -o gpu_metrics python src/train.py
```

**Generating a report:**

```bash
nsys stats training_report.nsys-rep  # text summary
nsys-ui training_report.nsys-rep      # open in GUI
```

**Export to SQLite for custom analysis:**

```bash
nsys export --type=sqlite training_report.nsys-rep -o training_report.sqlite
sqlite3 training_report.sqlite "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 10;"
```

## Advanced Usage / Configuration

| Option / API | Description |
|---|---|
| `-t cuda,nvtx,osrt` | Trace CUDA API, NVTX markers, and OS runtime. |
| `--gpu-metrics-device=0` | Collect GPU performance counters (SM utilisation, memory throughput, etc.). |
| `-d <seconds>` | Capture duration limit. |
| `-o <filename>` | Output filename (.nsys-rep). |
| `--stats=true` | Print a text summary after profiling. |
| `--cuda-memory-usage=true` | Track CUDA memory allocation / deallocation. |
| `--sample-process-tree` | Profile child processes. |
| `-w <N>` | Number of worker threads for trace processing. |

**NVTX annotation (Python):**

```python
import nvtx

@nvtx.annotate("training_step", color="red")
def training_step(batch):
    # ... training code ...

# Or with ranges:
r = nvtx.start_range(message="forward_pass", color="green")
output = model(input_ids)
nvtx.end_range(r)
```

**Profiling PyTorch dataloader and training loop:**

```python
import torch
import nvtx

model = WFBModel()
dataloader = DataLoader(...)

for epoch in range(5):
    for batch in dataloader:
        with nvtx.annotate("data_loading", color="yellow"):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

        with nvtx.annotate("forward_backward", color="blue"):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Key Metrics to Examine

| Metric | What to Look For |
|---|---|
| **Kernel duration** | Which kernels dominate? Unexpectedly long kernels may indicate suboptimal launch config. |
| **GPU utilisation** | Low utilisation (< 50%) → bottleneck is CPU data loading or CPU-side computation. |
| **Memory bandwidth** | Near-peak bandwidth → memory-bound kernel (compute vs memory bound analysis). |
| **CUDA API overhead** | Frequent small `cudaMemcpy` calls → batch or pin memory issues. |
| **Stream contention** | All kernels on default stream → no overlap of compute and data transfer. |

## Integration with the WFB Model Project

Nsight Systems is the profiling tool referenced in **Section 45 (Monitoring)** of the guide:

1. **Training profiling** — profile full training epochs to identify GPU idle time caused by CPU data loading, inefficient preprocessing, or small batch sizes.
2. **Inference profiling** — profile production inference to identify latency bottlenecks: kernel launch overhead versus actual compute.
3. **Memory analysis** — track GPU memory usage over time to detect leaks or fragmentation.
4. **NVTX markers** — the training and inference scripts annotate key phases (data loading, forward, backward, evaluation) with NVTX ranges for clear visualisation in the timeline.
5. **Regression testing** — compare `nsys stats` output between commits to detect performance regressions in CI.

## Common Pitfalls / Troubleshooting

- **Missing permissions on Linux** — set `sudo sysctl -w kernel.perf_event_paranoid=2` to allow non-root profiling.
- **Profiling adds overhead** — tracing adds ~5–15% overhead depending on trace options. Use `-t cuda,nvtx` (not OS runtime tracing) for lower overhead.
- **Large trace files** — traces for long training runs can be gigabytes. Use `-d` to limit duration, or sample every N iterations with NVTX ranges.
- **No CUDA kernels visible** — ensure the application actually uses CUDA (PyTorch model on GPU). Verify with `nvidia-smi`.
- **GUI not available on Windows** — use `nsys-ui` on a Linux machine or export to SQLite for programmatic analysis.
- **NVTX not working in Python** — install `pip install nvtx` or `conda install -c conda-forge nvtx`.
- **Profiling containers** — pass `--cap-add SYS_ADMIN` and mount `/proc` when running Docker. For Kubernetes, set `securityContext.capabilities.add: ["SYS_ADMIN"]`.

## Documentation Links

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Systems CLI Reference](https://docs.nvidia.com/nsight-systems/CLIReference/)
- [NVTX Python API](https://nvidia.github.io/NVTX/python.html)
- [Nsight Systems — PyTorch Profiling Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
