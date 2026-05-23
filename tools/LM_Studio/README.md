# LM Studio

**Version:** 0.3.10 (desktop app)

## Purpose

LM Studio is a desktop application for downloading, managing, and running large language models locally on consumer hardware. It provides a graphical interface for model discovery (via Hugging Face integration), local inference with GPU acceleration (CUDA, Metal, Vulkan), and an OpenAI-compatible API server for programmatic access. It is ideal for prototyping, local testing, and evaluating models before production deployment.

## Installation

Download the installer from [lmstudio.ai](https://lmstudio.ai/):

```bash
# Windows: Run LM-Studio-Setup-0.3.10.exe
# macOS: Open LM-Studio-0.3.10.dmg
# Linux: Unpack LM-Studio-0.3.10.AppImage
```

After installation, start the app and download models from the built-in Model Hub (e.g., Llama 3, Mistral, Phi-3).

## Basic Usage

### Local Inference via GUI

1. Open LM Studio, select a downloaded model from the sidebar.
2. Load the model with your preferred quantization (Q4_K_M, Q5_K_M, Q8_0).
3. Enter a prompt in the chat interface and click "Send".

### Programmatic Inference via API

Start the local inference server from the LM Studio UI (Settings > Developer > Start Server, default port 1234).

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # LM Studio ignores the key
)

response = client.chat.completions.create(
    model="lm-studio-model",
    messages=[
        {"role": "user", "content": "Explain wind farm optimization"}
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Advanced Usage / Configuration

### Server Configuration

```python
# Use streaming for real-time responses
stream = client.chat.completions.create(
    model="lm-studio-model",
    messages=[{"role": "user", "content": "Write a report"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Hardware Acceleration Settings

In LM Studio UI, configure:
- GPU Offload: Set to max (layers offloaded to GPU)
- Context Length: Adjust based on VRAM (2048-8192 tokens)
- Thread Count: Match CPU core count for prompt processing

### Model Management via CLI

```bash
# Models are stored in:
# Windows: %USERPROFILE%\.lmstudio\models\
# macOS: ~/.lmstudio/models/
# Linux: ~/.lmstudio/models/
```

## Integration with WFB Model Project

Use LM Studio for local testing of LLM components in the WFB pipeline before deploying to cloud APIs. Evaluate model responses for report generation, query interpretation, and tool-calling behavior. The OpenAI-compatible API allows drop-in replacement of cloud endpoints with local models during development.

## Common Pitfalls / Troubleshooting

- **Out of memory:** Reduce context length or use a smaller quantized model (e.g., Q4_K_M instead of Q8_0).
- **Slow inference:** Enable GPU offloading in settings. For CPU-only, reduce thread count to match physical cores.
- **Server not starting:** Check port availability (1234). Disable VPN/firewall if blocking localhost connections.
- **Model not loading:** Verify the model format is GGUF (LM Studio does not support raw PyTorch checkpoints).

## Documentation Links

- [LM Studio Docs](https://lmstudio.ai/docs)
- [Model Hub](https://lmstudio.ai/models)
- [API Reference](https://lmstudio.ai/docs/api)
