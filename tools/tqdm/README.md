# tqdm

**Version:** 4.66+

**Purpose:** Fast, extensible progress bar utility for Python loops, iterable processing, and training progress tracking. tqdm provides real-time ETA, throughput (it/s), and customizable postfix metrics with minimal overhead (<0.1% performance impact). Essential for monitoring long-running training and data processing pipelines.

## Installation

```powershell
pip install tqdm
```

Pin version:
```powershell
pip install "tqdm>=4.66,<5"
```

Platform notes: Cross-platform. On Jupyter notebooks, use `tqdm.notebook` for rich HTML progress bars; on IPython without notebooks, `tqdm.auto` detects the best backend automatically.

## Basic Usage

```python
from tqdm import tqdm
import time

# Simple loop
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)

# Manual update with dynamic metrics
pbar = tqdm(total=1000, desc="Training")
for step in range(1000):
    loss = compute_loss()
    pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")
    pbar.update(1)
pbar.close()

# Truncate long descriptions
pbar = tqdm(total=100, desc="A very long description that will be truncated",
            bar_format="{l_bar}{bar:30}{r_bar}")
```

## Advanced Usage / Configuration

### Nested progress bars
```python
from tqdm import tqdm
for epoch in tqdm(range(10), desc="Epochs"):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        train_step(batch)
```
Use `leave=False` on inner bars to keep output clean.

### Trainer integration
```python
from transformers import Trainer
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            tqdm.write(str(logs))  # prevents log line interference

# Use tqdm.write() instead of print() to avoid corrupting progress bars
```

### Redirecting to a file/log
```python
import sys
from tqdm import tqdm
tqdm(file=sys.stdout)  # default; use sys.stderr for log files
```

## Integration with WFB Model
tqdm is used in WFB model training scripts for:
- Training loop progress with loss/lr postfix
- Data preprocessing / tokenization
- Evaluation and inference loops
- Dataset downloading with `tqdm(urllib.urlopen(...))`

Replace `print()` with `tqdm.write()` in any callbacks that log during tqdm-iterated loops.

## Common Pitfalls / Troubleshooting
- **Multiple progress bars overlapping:** Use `leave=False` on inner loops; use `position=0` for single-bars to lock position
- **Slow performance with many updates:** Increase `mininterval` (default 0.1s) or `miniters` (default 1) — set `mininterval=1.0` for very fast loops
- **Not displaying in Jupyter:** Use `from tqdm.notebook import tqdm` or `from tqdm.auto import tqdm` for automatic environment detection
- **Output mixed with logging:** Use `tqdm.write()` instead of `print()` to avoid bar corruption
- **`total=None` causes indeterminate mode:** Provide `total` for accurate ETA; for unknown-length iterators, set `total=None` and tqdm shows throughput instead

## Documentation
- https://github.com/tqdm/tqdm
- https://tqdm.github.io/
