# pynvml — NVIDIA GPU Monitoring from Python

**Version:** 11.5.x (stable, distributed with NVIDIA tools)

## Purpose

pynvml (Python bindings for the NVIDIA Management Library — NVML) provides a Pythonic interface to query and monitor NVIDIA GPU state. It gives programmatic access to the same data that `nvidia-smi` displays, including:

- **GPU utilisation** — compute and memory utilisation percentages.
- **Memory usage** — total, used, free, and reserved memory per GPU.
- **Temperature and power** — GPU temperature, power draw, and power limits.
- **Process information** — which processes (PIDs) are using each GPU and how much memory.
- **Clocks and throttling** — GPU, memory, and video clocks; thermal and power throttling status.
- **ECC and PCIe info** — error counts, PCIe link generation and width.
- **NVLink** — peer-to-peer bandwidth between GPUs.

In the WFB model project, pynvml is the runtime GPU monitoring library that powers the monitoring dashboard and alerting system (Section 45 — Monitoring).

## Installation

pynvml ships with the NVIDIA GPU driver — `nvml.dll` (Windows) / `libnvidia-ml.so` (Linux) is installed as part of the driver. The Python wrapper can be installed via:

```bash
pip install nvidia-ml-py==11.515.105
# OR (deprecated alias, same package):
pip install pynvml==11.5.0

# Conda:
conda install -c conda-forge nvidia-ml-py=11.515.105
```

**Platform notes:**
- **Linux and Windows** only — macOS does not have NVIDIA GPU support.
- The Python package is a thin wrapper around the NVML shared library. It requires NVIDIA drivers ≥ 450.x (R470+ recommended for full functionality).
- The package is `nvidia-ml-py`, but the import name is `pynvml` (backward compatible).

## Basic Usage

```python
import pynvml

# Initialise NVML
pynvml.nvmlInit()

# Get number of GPUs
device_count = pynvml.nvmlDeviceGetCount()
print(f"Found {device_count} GPU(s)")

# Iterate over GPUs
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"\nGPU {i}: {name}")

    # Utilisation
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"  GPU Util:     {util.gpu}%")
    print(f"  Memory Util:  {util.memory}%")

    # Memory
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"  Memory:       {mem.used / 1024**3:.2f} / {mem.total / 1024**3:.2f} GB")

    # Temperature
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f"  Temperature:  {temp}°C")

    # Power
    power = pynvml.nvmlDeviceGetPowerUsage(handle)
    print(f"  Power Draw:   {power / 1000:.1f} W")

    # Processes
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        print(f"  PID {p.pid}: {p.usedGpuMemory / 1024**2:.0f} MB")

# Shutdown
pynvml.nvmlShutdown()
```

**Equivalent to `nvidia-smi` summary:**

```python
def print_gpu_summary():
    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        print(f"{i}: {name} | Mem: {mem.used//1024**3}/{mem.total//1024**3} GB | GPU: {util.gpu}%")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    print_gpu_summary()
```

## Advanced Usage / Configuration

| Function | Returns | Description |
|---|---|---|
| `nvmlInit()` | — | Initialise NVML. Call once at application start. |
| `nvmlShutdown()` | — | Clean up NVML resources. |
| `nvmlDeviceGetHandleByIndex(i)` | handle | Get handle for GPU at index `i` (0-based). |
| `nvmlDeviceGetHandleBySerial(serial)` | handle | Get handle by GPU serial number. |
| `nvmlDeviceGetName(handle)` | str | GPU product name (e.g., "NVIDIA A100 80GB"). |
| `nvmlDeviceGetUtilizationRates(handle)` | (gpu, memory) | Utilisation percentages (0–100). |
| `nvmlDeviceGetMemoryInfo(handle)` | (total, free, used) | Memory stats in bytes. |
| `nvmlDeviceGetTemperature(handle, sensor)` | int | Temperature in °C. Sensor: `NVML_TEMPERATURE_GPU`. |
| `nvmlDeviceGetPowerUsage(handle)` | int | Power draw in milliwatts. |
| `nvmlDeviceGetEnforcedPowerLimit(handle)` | int | Power limit in milliwatts. |
| `nvmlDeviceGetComputeRunningProcesses(handle)` | list[ProcInfo] | Running processes and their GPU memory usage. |
| `nvmlDeviceGetClockInfo(handle, type)` | int | Clock speed in MHz. Types: `NVML_CLOCK_GRAPHICS`, `NVML_CLOCK_SM`, `NVML_CLOCK_MEM`. |
| `nvmlDeviceGetPerformanceState(handle)` | int | P-State (0=full perf, 12=lowest). |
| `nvmlDeviceGetPcieThroughput(handle, counter)` | int | PCIe throughput in bytes/sec. |
| `nvmlDeviceGetNvLinkUtilizationCounter(handle, link, counter)` | (rx, tx) | NVLink data counters. |

**Monitoring in a training loop:**

```python
import pynvml
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

for step in range(num_steps):
    # Training step
    train_step()

    # Log metrics every 100 steps
    if step % 100 == 0:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mlflow.log_metrics({
            "gpu_util": util.gpu,
            "gpu_mem_used_gb": mem.used / 1024**3,
        }, step=step)

pynvml.nvmlShutdown()
```

**Detecting thermal throttling:**

```python
def is_throttling(handle):
    throttling = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
    if throttling & pynvml.nvmlClocksThrottleReasonGpuIdle():
        return False  # idle
    if throttling & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown():
        return True  # thermal throttle
    return False
```

## Integration with the WFB Model Project

pynvml is the low-level GPU monitoring library powering the observability stack in **Section 45 (Monitoring)**:

1. **Training dashboard** — a background thread (or sidecar process) polls pynvml every 5 seconds and pushes GPU utilisation, memory, temperature, and power metrics to MLflow / Prometheus.
2. **Alerting** — if GPU temperature exceeds 85°C or memory usage hits 95%, pynvml triggers an alert (email, Slack, or pager).
3. **Job scheduling** — when launching distributed training jobs, pynvml checks available GPU memory to prevent OOM crashes before they happen.
4. **Inference monitoring** — production inference servers report per-GPU utilisation via pynvml, feeding into the auto-scaling decision system.

## Common Pitfalls / Troubleshooting

- **`nvmlInit()` fails with `NVML_ERROR_DRIVER_NOT_LOADED`** — NVIDIA drivers are not installed or not loaded. Run `nvidia-smi` to verify.
- **Permission errors on Linux** — pynvml uses `/dev/nvidia*` devices, which may require root or membership in the `video` group: `sudo usermod -aG video $USER`.
- **Stale handles after GPU reset** — call `nvmlInit()` again if a GPU is reset or removed (rare on single-GPU setups, more common in multi-GPU VMs).
- **Thread safety** — NVML is thread-safe, but calling `nvmlShutdown()` in one thread while another still queries will fail. Use a single NVML lifecycle.
- **Python 3.12+ compatibility** — older `nvidia-ml-py` versions (< 11.515) may not support Python 3.12. Upgrade to the latest version.
- **Memory leak warning** — in very long running processes (days/weeks), occasional `nvmlInit()` / `nvmlShutdown()` cycles can help avoid resource accumulation.

## Documentation Links

- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/)
- [pynvml / nvidia-ml-py GitHub](https://github.com/gpuopenanalytics/pynvml)
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/)
- [nvidia-smi equivalent commands](https://nvidia.custhelp.com/app/answers/detail/a_id/5482/)
- [NVML Developer Guide](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)
