# Python

**Version:** 3.10+ (recommended: 3.11 or 3.12)

**Purpose:** Primary programming language for AI/ML development. Python provides the ecosystem (PyTorch, Transformers, NumPy) that everything in this project builds on. Python 3.11+ offers significant performance improvements over 3.10 (faster startup, better bytecode).

## Installation

### Windows
1. Download from https://www.python.org/downloads/
2. **Check "Add Python to PATH"** during installation — critical for command-line usage
3. Open PowerShell and verify:
```powershell
python --version
pip --version
```
4. Alternative: Install via **Miniconda** (recommended for ML): https://docs.conda.io/en/latest/miniconda.html

### Linux (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip
python3.11 --version
```

### macOS
```bash
brew install python@3.12
python3.12 --version
```

## Post-Install: Virtual Environment

Always use a venv for project isolation:
```powershell
# Create
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1     # Windows PowerShell
source venv/bin/activate          # Linux/macOS

# Deactivate
deactivate
```

## Key Python Packages for ML
```powershell
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib jupyter ipykernel
```

## Advanced Usage / Configuration
- **Multiple Python versions:** Use `py -3.11` (Windows Launcher) to switch between installed versions
- **`PYTHONPATH`:** Add project root to `$env:PYTHONPATH` to enable module imports without install: `$env:PYTHONPATH = "G:\zed\wfb_model"`
- **pip config:** Set global index: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` (China mirror)
- **UV (faster pip alternative):** `pip install uv` then `uv pip install ...` for 10-100x faster installs

## Integration with WFB Model
All training scripts in this project require Python 3.10+. The project assumes a virtual environment at `wfb_model/venv/`. Key scripts:
- `scripts/train.py` — training entry point
- `scripts/evaluate.py` — evaluation
- `wfb_model/` — package source
- Activate the venv before running any script: `.\venv\Scripts\Activate.ps1`

## Common Pitfalls / Troubleshooting
- **`pip` is not recognized:** Python not on PATH — reinstall with "Add Python to PATH" checked, or add manually to system PATH
- **`Fatal error in launcher: Unable to create process using...`:** Path mismatch — recreate venv or reinstall Python
- **Long path errors on Windows:** Enable long path support: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD`
- **Python 3.13+ not yet supported:** Core ML libraries (PyTorch, TensorFlow) may lack wheels — stick to 3.11 or 3.12
- **SSL certificate errors on pip:** `pip config set global.trusted-host pypi.org files.pythonhosted.org`

## Python Version Decision Matrix
| Python Version | ML Library Support | Recommended For |
|---------------|-------------------|-----------------|
| 3.10 | All libraries supported | Legacy compatibility |
| 3.11 | All libraries supported | **Best balance** — 10-25% faster than 3.10 |
| 3.12 | Most libraries supported | Latest features — check PyTorch wheel availability |
| 3.13 | Limited support | Not recommended — most ML libraries lack wheels |

**Recommendation for WFB model: Python 3.11** — best performance without compatibility issues.

## Configuration Tips
```powershell
# Set project PYTHONPATH permanently
$env:PYTHONPATH = "G:\zed\wfb_model"

# Use a .pth file instead (persists across shells)
New-Item -ItemType File -Path "$(python -c 'import site; print(site.getsitepackages()[0])')\wfb_model.pth" -Value "G:\zed\wfb_model"

# Install UV for faster package management
pip install uv
uv pip install -r requirements.txt  # 10-100x faster than pip
```

## Integration with WFB Model
All training scripts require Python 3.10+. The project assumes a virtual environment at `wfb_model/venv/`. Key scripts:
- `scripts/train.py` — training entry point
- `scripts/evaluate.py` — evaluation
- `wfb_model/` — package source
- Activate before running: `.\venv\Scripts\Activate.ps1`
- The project's `pyproject.toml` specifies `requires-python = ">=3.10"`

## Common Pitfalls / Troubleshooting
- **`pip` is not recognized:** Python not on PATH — reinstall with "Add Python to PATH" checked, or add manually
- **`Fatal error in launcher: Unable to create process`:** Path mismatch — recreate venv: `deactivate; Remove-Item -Recurse venv; python -m venv venv`
- **Long path errors on Windows:** Enable long path support via Group Policy or registry
- **Python 3.13+ not yet supported:** Core ML libraries (PyTorch, TensorFlow) may lack wheels — stick to 3.11 or 3.12
- **SSL certificate errors on pip:** `pip config set global.trusted-host pypi.org files.pythonhosted.org`
- **`ModuleNotFoundError: No module named 'wfb_model'`:** Activate the venv or set PYTHONPATH

## Documentation
- https://docs.python.org/3/
- https://pypi.org/
