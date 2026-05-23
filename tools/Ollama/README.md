# Ollama

**Version:** 0.3+

**Purpose:** Run LLMs locally with a simple CLI. Great for testing and personal use.

## Download

- https://ollama.com/download

## Installation

1. Download and run the Windows installer
2. Verify:
```powershell
ollama --version
```

## Usage with Custom Models

### Convert to GGUF

```powershell
# Install llama.cpp
pip install llama-cpp-python

# Convert PyTorch checkpoint to GGUF
# (requires llama.cpp conversion script)
python llama.cpp/convert.py ./checkpoints/final --outfile model.gguf
```

### Create Modelfile

```
FROM model.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER num_ctx 131072
PARAMETER stop </s>

TEMPLATE """{{ .Prompt }}"""
```

### Create and Run

```powershell
ollama create wfb-model -f Modelfile
ollama run wfb-model
```

### API Access

```powershell
# By default, Ollama serves at http://localhost:11434
curl http://localhost:11434/api/generate -d '{"model": "wfb-model", "prompt": "Hello"}'
```

## Key Features

- **Simple CLI** — one command to run
- **GGUF format** — CPU + GPU inference
- **OpenAI-compatible API** — set `OPENAI_BASE_URL=http://localhost:11434/v1`
- **Model library** — download pre-trained models
