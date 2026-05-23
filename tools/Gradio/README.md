# Gradio

**Version:** 5.0+

**Purpose:** Rapidly build interactive web demos for ML models with Python. Gradio provides a high-level `gr.Interface` for single-input/single-output demos, `gr.ChatInterface` for conversational AI, and `gr.Blocks` for fully custom UIs. It supports streaming, queuing, authentication, and sharing via public links. In the WFB model pipeline, Gradio creates evaluation interfaces for internal testing, user studies for post-training alignment, and client-facing demos.

## Installation

```powershell
pip install "gradio==5.0.0"
```

## Basic Usage

### Simple Interface

```python
import gradio as gr

def generate(text: str, temperature: float) -> str:
    # Replace with actual model call
    return f"Generated response for: {text[:50]} (t={temperature})"

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(label="Prompt"), gr.Slider(0.0, 1.0, 0.7, label="Temperature")],
    outputs=gr.Textbox(label="Output"),
    title="WFB Model Demo",
)
demo.launch()
```

### Chat Interface

```python
import gradio as gr

def chat(message: str, history: list) -> str:
    history = history or []
    return f"You said: {message}"

demo = gr.ChatInterface(
    fn=chat,
    title="WFB Chat",
    description="Test the model's conversational ability",
)
demo.launch(share=True)  # Public link via Gradio tunnels
```

## Advanced Usage

### Custom Blocks with Streaming

```python
import gradio as gr
import time

def stream_response(prompt: str):
    tokens = [f"token_{i}" for i in range(20)]
    for t in tokens:
        yield t
        time.sleep(0.05)

with gr.Blocks(title="WFB Streaming Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WFB Model Streaming")
    with gr.Row():
        inp = gr.Textbox(label="Input", lines=3)
        out = gr.Textbox(label="Output", lines=10)
    btn = gr.Button("Generate")
    btn.click(stream_response, inp, out, queue=True)

demo.queue(max_size=10).launch(max_threads=4)
```

### Queuing and Authentication

```python
demo = gr.Interface(fn=generate, inputs="text", outputs="text")
demo.queue(default_concurrency_limit=5)
demo.launch(auth=("admin", "wfb_pass"), server_port=7860)
```

### Flagging for Data Collection

```python
demo = gr.Interface(
    fn=generate, inputs="text", outputs="text",
    flagging_mode="manual",
    flagging_dir="./flagged_outputs",
)
```

## Integration with WFB Model

Gradio is used during **Part 6** (Post-Training) and **Part 8** (Production) of the WFB pipeline. After SFT or DPO training, spin up a Gradio demo for qualitative evaluation — comparing base vs. aligned model outputs side-by-side. Use `gr.ChatInterface` for RLHF preference data collection: human raters converse with the model and flag poor responses. In production, Gradio serves as an internal staging environment before routing traffic to Ray Serve or vLLM.

## Common Pitfalls

- **Queue not enabled**: Streaming requires `demo.queue()`. Without it, outputs appear after full generation.
- **Share link timeout**: `share=True` links expire after 72h. For persistent access, deploy behind Nginx.
- **State management**: `gr.State()` is per-user. Do not store model weights in state — load once at module level.
- **CORS errors**: When embedding in a different domain, set `allowed_paths` and configure CORS headers.
- **Large model loading**: Load the model outside the `fn` and reuse it. Inside `fn`, inference only.

## Documentation

- https://www.gradio.app/docs
- https://www.gradio.app/guides/quickstart
