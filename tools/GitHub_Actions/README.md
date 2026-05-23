# GitHub Actions — CI/CD for ML Pipelines

**Version:** Workflow syntax v2 (latest)

## Purpose

GitHub Actions is a CI/CD (Continuous Integration / Continuous Deployment) platform integrated into GitHub repositories. It automates software workflows — testing, linting, building, and deploying — in response to Git events (push, PR, tag, schedule). For ML projects, GitHub Actions provides:

- **Automated testing** — run pytest, linting, and type checks on every PR.
- **Model training & validation** — trigger training pipelines on code or data changes.
- **Model deployment** — deploy to staging / production after successful validation.
- **Self-hosted runners** — run jobs on your own GPU-equipped hardware.
- **Matrix builds** — test across multiple Python versions, OSes, or configurations.
- **Scheduled jobs** — periodic model retraining or data drift monitoring.

In the WFB model project, GitHub Actions orchestrates the CI/CD pipeline (Section 51), tying together testing, data versioning (DVC), experiment tracking (MLflow), and deployment.

## Installation (Setup)

No installation required — GitHub Actions is a service. To use it:

1. Create `.github/workflows/*.yml` files in your repository.
2. GitHub automatically picks up valid workflow files and runs them on triggers.

**Prerequisites:**
- A GitHub repository with your code pushed.
- For GPU runners / self-hosted, you need to add and configure a runner (see below).

## Basic Usage

**`.github/workflows/ci.yml` — simple CI:**

```yaml
name: CI Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Lint
        run: |
          ruff check .
          black --check .
      - name: Test
        run: |
          pytest tests/ -v --cov=src
```

**`.github/workflows/train.yml` — data + model pipeline:**

```yaml
name: Model Training
on:
  workflow_dispatch:
  push:
    paths:
      - "data/**"
      - "src/**"

jobs:
  train:
    runs-on: self-hosted-gpu
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Pull data from DVC
        run: |
          pip install dvc[s3]
          dvc pull
      - name: Train & log to MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python src/train.py
```

## Advanced Usage / Configuration

| Feature | Description |
|---|---|
| `runs-on: self-hosted` | Run jobs on your own hardware (GPU machines). |
| `services:` | Spin up Docker sidecars (e.g., PostgreSQL, MLflow server). |
| `strategy.matrix` | Run jobs across combinations (Python 3.10 / 3.11 / 3.12, OS, etc.). |
| `workflow_dispatch` | Manually trigger a workflow from the GitHub UI. |
| `schedule:` | Cron-based triggers (e.g., nightly retraining). |
| `environment:` | Deploy to GitHub Environments with approval gates and secrets. |
| `concurrency` | Cancel duplicate runs (e.g., skip queued runs for the same PR). |

**Self-hosted GPU runner setup:**

```bash
# On your GPU machine:
mkdir actions-runner && cd actions-runner
curl -o actions-runner-win-x64-2.311.0.zip -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-win-x64-2.311.0.zip
# Extract and configure:
./config.cmd --url https://github.com/<org>/<repo> --token <REG_TOKEN>
./run.cmd
```

**Caching dependencies and data:**

```yaml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

## Integration with the WFB Model Project

GitHub Actions is the orchestration layer for **Section 51 (CI/CD)**:

1. **On every PR** — the `ci.yml` workflow runs linting (Ruff, Black), type checking, and pytest. PRs are blocked if any step fails.
2. **On pushes to `main`** — the `train.yml` workflow triggers full training: pulls the latest data from DVC, runs `src/train.py`, logs results to MLflow, and registers the best model.
3. **Scheduled retraining** — a cron job runs weekly to check for data drift and trigger retraining if needed.
4. **Deployment** — a `deploy.yml` workflow deploys the production model to the inference endpoint after manual approval.

## Common Pitfalls / Troubleshooting

- **Self-hosted runner offline** — ensure the runner machine is running and the `run.sh` / `run.cmd` process stays alive. Use `sudo ./svc.sh install && sudo ./svc.sh start` on Linux to run as a service.
- **Secrets not available** — secrets are not passed to workflows triggered by `pull_request` from forked repositories. Use `pull_request_target` cautiously.
- **Matrix explosion** — keep matrix combinations manageable. Use `include` / `exclude` to avoid redundant jobs.
- **DVC + CI** — DVC pull requires cloud credentials. Store them in GitHub Secrets and pass as environment variables.
- **Job timeouts** — default timeout is 360 minutes. Set `timeout-minutes: <N>` on the job for long training runs.
- **Disk space on runners** — GitHub-hosted runners have ~14 GB free. Use `actions/cache` and clean up with `rm -rf` or `docker system prune -f`.

## Documentation Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners)
- [Caching Dependencies](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [GitHub Actions for MLOps](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-machine-learning-models)
