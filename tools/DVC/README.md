# DVC — Data and Model Version Control

**Version:** 3.48.x (stable)

## Purpose

DVC (Data Version Control) brings Git-like versioning to large files — datasets, model weights, checkpoints — that do not belong in a Git repository. It works on top of Git by storing file hashes in Git (`.dvc` files) while the actual data lives in remote storage (S3, GCS, HDFS, local). DVC also provides:

- **Pipeline management** — define reproducible stages (`dvc.yaml`) with inputs, outputs, and commands.
- **Experiments** — run and compare ML experiments with `dvc exp`.
- **CI/CD integration** — pull/push data in CI pipelines, enabling end-to-end reproducibility.

In the WFB model project, DVC version-controls training data, preprocessed datasets, and model checkpoints, forming the data layer of the CI/CD pipeline (Section 51).

## Installation

```bash
pip install dvc==3.48.0
# Remote storage support (pick one or more):
pip install "dvc[s3]==3.48.0"     # AWS S3
pip install "dvc[gs]==3.48.0"     # Google Cloud Storage
pip install "dvc[azure]==3.48.0"  # Azure Blob Storage
pip install "dvc[ssh]==3.48.0"    # SSH remote
pip install "dvc[gdrive]==3.48.0" # Google Drive

# Conda:
conda install -c conda-forge dvc=3.48.0
```

**Platform notes:**
- Windows: ensure Git is on PATH. Use `dvc init` inside a Git repository.
- Linux / macOS: no special configuration needed.
- For large datasets (>10 GB), consider `dvc gc` regularly and use a remote with versioning enabled.

## Basic Usage

```bash
# Initialize (inside a Git repo)
git init
dvc init

# Track a dataset
dvc add data/raw/train_dataset.parquet
git add data/raw/train_dataset.parquet.dvc .gitignore
git commit -m "track training dataset"

# Set up a remote storage
dvc remote add -d myremote s3://wfb-dvc-storage
dvc push

# Pull data on another machine
dvc pull
# Or a specific file / directory
dvc checkout data/processed/
```

**Tracking model checkpoints:**

```bash
dvc add checkpoints/epoch_10.pt
git add checkpoints/epoch_10.pt.dvc
git commit -m "model checkpoint epoch 10"
```

**Basic pipeline (`dvc.yaml`):**

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/
      - src/preprocess.py
    outs:
      - data/processed/
  train:
    cmd: python src/train.py
    deps:
      - data/processed/
      - src/train.py
    outs:
      - checkpoints/
    metrics:
      - metrics.json:
          cache: false
```

## Advanced Usage / Configuration

| Command / Option | Description |
|---|---|
| `dvc init` | Initialize DVC in the repository. Creates `.dvc/` directory. |
| `dvc add <path>` | Start tracking a file or directory. Replaces it with a `.dvc` file. |
| `dvc push / pull` | Sync data with the default remote. |
| `dvc checkout` | Restore tracked files from the cache. |
| `dvc repro` | Reproduce the pipeline — runs only stages with changed dependencies. |
| `dvc metrics diff` | Compare metrics across Git commits or experiments. |
| `dvc exp run` | Run an experiment; logs parameters and metrics. |
| `dvc exp show` | Display a table of experiments with parameters and metrics. |
| `dvc gc` | Garbage collect — remove unused files from cache. |
| `dvc remote add / modify` | Configure remote storage (S3, GCS, etc.). |

**Configuration file `.dvc/config`:**

```ini
[core]
    remote = myremote
['remote "myremote"']
    url = s3://wfb-dvc-storage
    access_key_id = AKIA...
    secret_access_key = ...
```

**Using with MLflow:** DVC version-controls the raw data and checkpoints, while MLflow tracks the experiments that consume them.

## Integration with the WFB Model Project

DVC is the data versioning pillar, feeding into **Section 51 (CI/CD)**:

1. **Data versioning** — raw EHR data, preprocessed parquet files, and augmentation corpora are tracked with `dvc add`. Each dataset version is pinned to a Git commit.
2. **Model checkpoints** — fine-tuned BERT checkpoints are stored via DVC, not Git, keeping the repo lightweight.
3. **CI/CD pipelines** — the CI workflow (GitHub Actions) runs `dvc pull` to restore the exact dataset version before training, ensuring reproducibility.
4. **Experiments** — `dvc exp run` together with `dvc.yaml` enables parameter sweeps with full provenance.

## Common Pitfalls / Troubleshooting

- **`dvc init` outside a Git repo** — DVC requires a Git repository. Run `git init` first.
- **Missing remote credentials** — use environment variables (`AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`) or `dvc remote modify` with `--local` flag to store secrets outside Git.
- **Large `.dvc/cache` directory** — run `dvc gc -w` regularly to remove unused cached files.
- **Windows symlink issues** — DVC uses symlinks by default on POSIX. On Windows you may need `cache.type = hardlink` or `cache.type = copy` in `.dvc/config`.
- **`dvc pull` doesn't update files** — run `dvc checkout` after `dvc pull` if the files are not automatically restored.

## Documentation Links

- [DVC Documentation](https://dvc.org/doc)
- [DVC Get Started](https://dvc.org/doc/start)
- [DVC Pipelines (dvc.yaml)](https://dvc.org/doc/user-guide/pipelines)
- [DVC Experiments](https://dvc.org/doc/user-guide/experiments)
- [DVC Remotes](https://dvc.org/doc/user-guide/data-management/remote-storage)
