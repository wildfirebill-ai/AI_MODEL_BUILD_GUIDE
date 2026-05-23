# psutil (Python System & Process Utilities)

**Version:** 5.9+

**Purpose:** Cross-platform library for retrieving system information — CPU utilization, memory usage, disk/network I/O, and process metrics. psutil is essential for building training dashboards, detecting memory leaks, monitoring GPU host memory pressure, and implementing resource-aware scheduling.

## Installation

```powershell
pip install psutil
```

Pin version:
```powershell
pip install "psutil>=5.9,<6"
```

Platform notes: Fully supported on Windows, Linux, and macOS. On Linux, some metrics (e.g., per-process disk I/O) require kernel 2.6.20+. On Windows, requires Windows 7+.

## Basic Usage

```python
import psutil

# CPU
print(f"CPU cores: {psutil.cpu_count()}, Physical: {psutil.cpu_count(logical=False)}")
print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")

# Memory
mem = psutil.virtual_memory()
print(f"Memory: {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB ({mem.percent}%)")

# Process metrics
proc = psutil.Process()
print(f"Process CPU: {proc.cpu_percent()}%")
print(f"Process memory: {proc.memory_info().rss / 1e6:.0f} MB")

# Disk
disk = psutil.disk_io_counters()
print(f"Disk read: {disk.read_bytes / 1e9:.1f} GB, write: {disk.write_bytes / 1e9:.1f} GB")
```

## Advanced Usage / Configuration

### Memory leak detection
```python
import psutil
proc = psutil.Process()
rss_before = proc.memory_info().rss
# Run training step
rss_after = proc.memory_info().rss
if rss_after - rss_before > 100 * 1024 * 1024:  # >100MB growth
    print(f"Potential memory leak: +{(rss_after - rss_before) / 1e6:.0f} MB")
```

### GPU host memory monitoring
```python
# Monitor host memory during training
import psutil
while training_running:
    mem = psutil.virtual_memory()
    if mem.percent > 95:
        print(f"WARNING: Host memory at {mem.percent}% — risk of OOM kill")
    time.sleep(30)
```

### Network monitoring
```python
net = psutil.net_io_counters()
print(f"Bytes sent: {net.bytes_sent / 1e9:.2f} GB, recv: {net.bytes_recv / 1e9:.2f} GB")
```

## Integration with WFB Model
psutil is used in WFB model training infrastructure for:
- `monitor.py` — training resource monitoring script that logs CPU, memory, and disk metrics alongside GPU metrics from `nvidia-smi`
- OOM (out-of-memory) telemetry — detecting and reporting when host memory pressure causes training crashes
- `wfb_model/utils/resources.py` — resource-aware scheduling helpers

## Common Pitfalls / Troubleshooting
- **`cpu_percent()` returns 0 on first call:** First call is a measurement point — subsequent calls return meaningful data. Call once to warm up.
- **Permission errors on Linux:** Some process metrics require root (`/proc/<pid>/io`). Run with `sudo` or check `/proc` permissions.
- **Missing disk I/O per-process on Windows:** Windows does not expose per-process disk I/O counters — use global `disk_io_counters()` instead.
- **CPU percent > 100%:** This is normal — `cpu_percent()` reports across all cores; 800% means 8 cores at full utilization.
- **Swapped memory not reported:** `psutil.swap_memory()` gives swap info; `virtual_memory().available` is the best indicator of usable memory.

## Documentation
- https://github.com/giampaolo/psutil
- https://psutil.readthedocs.io/
