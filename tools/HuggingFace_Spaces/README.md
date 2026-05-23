# HuggingFace Spaces — Free ML Demo Hosting

[HuggingFace Spaces](https://huggingface.co/spaces) provides free hosting for ML demos using Gradio, Streamlit, or Docker. Options for CPU, T4, and A10G GPUs.

## Quick Start — Create a Space via CLI

```bash
pip install huggingface-hub

# Login
huggingface-cli login

# Create a new Space
huggingface-cli repo create my-demo --type space -s streamlit
```

## Manual Setup

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose SDK: **Streamlit**, **Gradio**, or **Docker**
3. Clone locally, add your code, push

## Streamlit Space Example

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("My Model Demo")
st.write("Hosted on HuggingFace Spaces for free!")

df = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
st.line_chart(df)

value = st.slider("Pick a number", 0, 100, 50)
st.metric("Your value", value)
```

```yaml
# requirements.txt
streamlit==1.35.0
pandas==2.2.0
numpy==1.26.0
```

## Gradio Space Example

```python
# app.py
import gradio as gr
import numpy as np
from PIL import Image

def classify_image(img):
    return np.random.rand(10).tolist()

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Image Classifier Demo",
).launch()
```

```yaml
# requirements.txt
gradio==4.29.0
numpy==1.26.0
Pillow==10.2.0
```

## Hardware Options

Configure in `README.md`:

```yaml
---
title: My Demo
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
hardware:
  accelerator: T4  # CPU (default), T4, A10G, A100
---
```

GPU tiers: **T4** (16GB VRAM, free tier), **A10G** (24GB, paid), **A100** (80GB, paid).

## Secrets Management

```bash
huggingface-cli secret set OPENAI_API_KEY sk-...
```

Access in code:

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

## Persistent Storage

Spaces have ephemeral storage. For persistence, mount a HuggingFace Dataset:

```python
from datasets import load_dataset

ds = load_dataset("your-username/your-dataset", split="train")
```

Or use the Space's `/data` directory for semi-persistent storage during uptime.

## Zero-Downtime Deploys

Enable in Space settings → **Zero-downtime deploys**. Updates roll without dropping active connections.

## Using the HuggingFace Hub SDK

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="model.pt",
    path_in_repo="model.pt",
    repo_id="username/my-demo",
    repo_type="space",
)
```

## Multiple Files & Subdirectories

Spaces support arbitrary file structures. Place helper modules in subdirectories and reference them from `app.py`.

```
my-space/
├── app.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── model.py
└── assets/
    └── logo.png
```

## Use Cases

- **Model demos** — showcase your trained models
- **Community showcase** — share with the HuggingFace community
- **Internal tools** — private Spaces for team use
- **LLM playgrounds** — chat interfaces with GPU acceleration
