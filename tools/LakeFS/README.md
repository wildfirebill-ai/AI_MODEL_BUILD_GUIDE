# LakeFS — Data Version Control on Object Storage

LakeFS brings Git-like branch, commit, merge, and revert semantics to data lakes on S3, GCS, and Azure Blob, enabling versioned data management for ML pipelines.

## Key Concepts

- **Repository** — a versioned data lake backed by object storage
- **Branch** — an isolated copy of data for development
- **Commit** — an immutable snapshot of data on a branch
- **Merge** — combine changes from one branch into another
- **Hook** — automated checks triggered on commit/merge
- **GC Rule** — garbage collection policy for unused objects

## Python Client Setup

```python
import lakefs

client = lakefs.Client(
    host="http://localhost:8000",
    username="lakefs-user",
    password="lakefs-password",
)

repo = client.create_repository(
    name="ml-data",
    storage_namespace="s3://my-bucket/lakefs/ml-data",
    default_branch="main",
)
```

## Branch Operations

```python
# List branches
for branch in repo.branches():
    print(branch.id)

# Create a feature branch
feature = repo.create_branch(name="feature/exp-v2", source_branch="main")

# List objects
for obj in feature.objects():
    print(obj.path)

# Upload file
feature.object("data/train.parquet").upload("local_file.parquet")
```

## Committing and Merging

```python
# Commit
commit = feature.commit(
    message="Add experiment-v2 features",
    metadata={"author": "ml-team"},
)

# View commit history
for c in repo.log(branch="feature/exp-v2", max_amount=5):
    print(f"{c.id}: {c.message}")

# Merge
feature.merge_into("main", message="Merge exp-v2 into main")

# Diff between branches
for change in repo.diff(left_branch="feature/exp-v2", right_branch="main"):
    print(f"{change.type}: {change.path}")
```

## Reverting

```python
# Revert a specific commit
repo.revert(branch="main", reference=commit.id, parent_number=1)

# Reset branch to a previous state
repo.reset_branch(branch="main", ref="main~1")
```

## Hooks

```yaml
# lakefs.yaml (repository root)
hooks:
  pre-commit:
    - id: validate_schema
      type: webhook
      properties:
        url: "http://hooks:5000/validate"
        timeout: 30s
  post-merge:
    - id: trigger_pipeline
      type: webhook
      properties:
        url: "http://ml-pipeline:8000/trigger"
```

## Integration: ML Pipeline

```python
import lakefs
import pandas as pd

class DataVersioner:
    def __init__(self, repo_name: str):
        self.client = lakefs.Client(host="http://localhost:8000")
        self.repo = self.client.get_repository(repo_name)

    def create_branch(self, name: str) -> str:
        self.repo.create_branch(name=name, source_branch="main")
        return name

    def commit_data(self, branch: str, df: pd.DataFrame, path: str, msg: str):
        local = "/tmp/staging.parquet"
        df.to_parquet(local)
        self.repo.get_branch(branch).object(path).upload(local)
        self.repo.get_branch(branch).commit(message=msg)

    def load_data(self, branch: str, path: str) -> pd.DataFrame:
        obj = self.repo.get_branch(branch).object(path)
        return pd.read_parquet(obj.reader(mode="rb"))

    def merge(self, branch: str, msg: str):
        self.repo.get_branch(branch).merge_into("main", message=msg)

v = DataVersioner("ml-data")
branch = v.create_branch("ablation-v3")
v.commit_data(branch, pd.DataFrame({"x": [1, 2], "y": [0, 1]}),
              "data/train.parquet", "Initial features")
df = v.load_data(branch, "data/train.parquet")
v.merge(branch, "Merge ablation-v3")
```

## S3 Gateway

```python
# LakeFS is transparent to S3 clients
import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:8000",
                  aws_access_key_id="key", aws_secret_access_key="secret")

# List files on a branch
for obj in s3.list_objects_v2(Bucket="ml-data", Prefix="main/data/").get("Contents", []):
    print(obj["Key"])
```

## Best Practices

- Use branches per experiment, feature, or team member for data isolation.
- Commit before merging — commits are the atomic unit of reproducibility.
- Configure pre-commit hooks to validate data schema and quality.
- Set GC rules per branch pattern (feature branches expire faster).
- Never write directly to main — use branches for all data changes.
