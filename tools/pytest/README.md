# pytest — Testing Framework for Python

**Version:** 8.3.x (stable)

## Purpose

pytest is a mature, full-featured testing framework for Python. It enables writing simple unit tests as well as complex functional and integration tests with minimal boilerplate. Key features:

- **Auto-discovery** — tests are detected by filename (`test_*.py` / `*_test.py`) and function name prefix (`test_`).
- **Fixtures** — reusable setup / teardown logic with scoping (function, class, module, session).
- **Parameterization** — run the same test multiple times with different inputs (Cartesian product via `@pytest.mark.parametrize`).
- **Assertions** — plain `assert` statements with rich introspection (no need for `self.assertEqual(...)`).
- **Plugins** — `pytest-cov` (coverage), `pytest-xdist` (parallel execution), `pytest-mock` (mocking), `pytest-benchmark`, and hundreds more.
- **Marks** — `@pytest.mark.skip`, `@pytest.mark.filterwarnings`, `@pytest.mark.slow`, etc.
- **Hooks** — customise test collection, execution, and reporting via conftest.py hooks.

In the WFB model project, pytest validates every component — data pipelines, model export, inference correctness, and integration with MLflow / DVC.

## Installation

```bash
pip install pytest==8.3.4
# Recommended plugins:
pip install pytest-cov==5.0.0 pytest-xdist==3.6.1 pytest-mock==3.14.0 pytest-benchmark==4.0.0

# Conda:
conda install -c conda-forge pytest=8.3.4
```

**Platform notes:**
- All platforms work identically.
- On Windows, use `pytest -n auto` with `pytest-xdist` cautiously — forking may not work; use thread-based parallelism (`-n auto --dist=worksteal`).

## Basic Usage

**File: `tests/test_inference.py`**

```python
import pytest
import numpy as np
from src.model import WFBInference

@pytest.fixture
def model():
    return WFBInference("models/test_checkpoint.pt")

def test_output_shape(model):
    input_ids = np.ones((1, 128), dtype=np.int64)
    logits = model.predict(input_ids)
    assert logits.shape == (1, 2), "Expected binary classification output"

@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_varying_sequence_lengths(model, seq_len):
    input_ids = np.ones((1, seq_len), dtype=np.int64)
    logits = model.predict(input_ids)
    assert logits.shape == (1, 2)

@pytest.mark.slow
def test_end_to_end_pipeline():
    from src.pipeline import run_pipeline
    metrics = run_pipeline("data/sample.parquet")
    assert metrics["accuracy"] > 0.8
```

**Running tests:**

```bash
pytest tests/ -v
pytest tests/ -v -k "output_shape"       # filter by keyword
pytest tests/ -v -m "not slow"           # exclude slow tests
pytest tests/ --cov=src --cov-report=term-missing
pytest tests/ -n auto                    # parallel execution
```

## Advanced Usage / Configuration

| Feature | Description |
|---|---|
| `conftest.py` | Shared fixtures and hooks at each directory level. |
| `@pytest.mark.parametrize("arg", values)` | Run a test once per value (or product of values). |
| `@pytest.fixture(scope="session")` | Create once per test session (e.g., model load). |
| `mocker` (`pytest-mock`) | Mock objects / functions via `mocker.patch()`. |
| `tmp_path` fixture | Built-in temporary directory (pathlib.Path), unique per test. |
| `--durations=N` | Show the N slowest tests. |
| `-s` | Disable output capturing (print statements visible). |
| `--lf` (last-failed) | Re-run only tests that failed last time. |
| `--sw` (stepwise) | Stop after first failure; resume from there on next run. |

**Configuration file `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --strict-markers --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: requires CUDA GPU",
]
```

**Mocking MLflow / external services:**

```python
def test_tracking_logs_params(mocker):
    mock_log_params = mocker.patch("mlflow.log_params")
    import src.train
    src.train.run_training()
    mock_log_params.assert_called_once_with({"lr": 3e-5, "batch_size": 16})
```

## Integration with the WFB Model Project

pytest is the quality gate for all WFB model components:

1. **Unit tests** (`tests/unit/`) — test individual functions: tokenization, data loading, metric computation, ONNX export correctness.
2. **Integration tests** (`tests/integration/`) — test end-to-end pipeline with small sample data: training loop, MLflow logging, DVC tracking.
3. **Regression tests** (`tests/regression/`) — compare outputs against known-good snapshots to detect regressions.
4. **Coverage** — `pytest-cov` enforces a minimum coverage threshold (e.g., 85%) in CI.
5. **Performance benchmarks** — `pytest-benchmark` tracks inference latency across commits to prevent regressions.

## Common Pitfalls / Troubleshooting

- **Tests not discovered** — ensure filenames match `test_*.py` or `*_test.py`. Check `testpaths` in `pyproject.toml`.
- **Global state leaking** — use fixtures (especially `scope="function"`) rather than module-level setup. Clear state in `yield` teardown.
- **Mocking the wrong path** — always mock where the object is *used* (e.g., `mocker.patch("src.train.mlflow.log_params")`), not where it is defined.
- **Slow tests in CI** — mark slow tests with `@pytest.mark.slow` and split into separate job steps or use `pytest-xdist`.
- **GPU memory** — GPU-heavy tests must run sequentially. Use `@pytest.mark.order(1)` or a custom `pytest_collection_modifyitems` hook.
- **Flaky tests** — use `@pytest.mark.flaky(reruns=3)` from `pytest-rerunfailures` for network-dependent tests.

## Documentation Links

- [pytest Documentation](https://docs.pytest.org/en/stable/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest Parametrization](https://docs.pytest.org/en/stable/parametrize.html)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest-xdist](https://github.com/pytest-dev/pytest-xdist)
- [pytest-mock](https://github.com/pytest-dev/pytest-mock)
