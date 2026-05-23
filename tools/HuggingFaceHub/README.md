# HuggingFace Hub

**Version:** 0.24+

**Purpose:** Access and share models, datasets, and spaces on the HuggingFace Hub.

## Installation

```powershell
pip install huggingface-hub
```

## Usage

```python
from huggingface_hub import HfApi, snapshot_download, create_repo, upload_folder

api = HfApi()

# Download a model
model_path = snapshot_download(repo_id="mistralai/Mistral-7B-v0.1")

# Upload trained model
create_repo(repo_id="username/my-model", private=True)
upload_folder(
    repo_id="username/my-model",
    folder_path="./checkpoints/final",
)

# Search models
models = api.list_models(
    task="text-generation",
    library="transformers",
    sort="downloads",
    direction=-1,
    limit=10,
)

# Upload dataset
api.create_repo(repo_id="username/my-dataset", repo_type="dataset")
api.upload_file(
    repo_id="username/my-dataset",
    path_in_repo="train.jsonl",
    path_or_fileobj="./data/train.jsonl",
)

# Authentication
from huggingface_hub import login
login(token="hf_...")  # Or set HF_TOKEN env var
```

## CLI

```powershell
# Login
huggingface-cli login

# Upload
huggingface-cli upload my-model ./checkpoints/final

# Download
huggingface-cli download mistralai/Mistral-7B-v0.1
```

## Documentation

- https://huggingface.co/docs/hub
