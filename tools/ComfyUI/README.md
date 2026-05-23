# ComfyUI — Stable Diffusion Node-Based Workflow Interface

**Version:** Latest (rolling) / `comfyui` (standalone application)

## Purpose

ComfyUI is an interactive node-based interface for Stable Diffusion that enables visual construction of image and video generation workflows. Users connect nodes (checkpoint loader, prompt encoder, KSampler, ControlNet, VAE decode) via edges to define every step of the diffusion process. It supports checkpoint management, custom nodes, LoRA, ControlNet, animatediff video generation, and model merging — all without writing code.

## Installation

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Launch
python main.py --listen 0.0.0.0 --port 8188
```

Open http://localhost:8188 in a browser. Place Stable Diffusion checkpoints (`.safetensors`) in `models/checkpoints/` and VAE models in `models/vae/`.

## Basic Usage Example

Workflows are serialized as JSON. Minimal text-to-image workflow (loadable via "Load" button in UI):

```json
{
  "3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
  "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
  "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat astronaut", "clip": ["4", 1]}},
  "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["4", 1]}},
  "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
  "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}}
}
```

The workflow can be executed programmatically:

```python
import json, requests

with open("workflow.json") as f:
    workflow = json.load(f)

resp = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
print(resp.json())  # {"prompt_id": "...", "number": 1}
```

## Advanced Usage / Configuration

- **Custom nodes**: Install via `ComfyUI-Manager` (GitHub: `ltdrdata/ComfyUI-Manager`) — 1,000+ nodes for ControlNet, AnimateDiff, IP-Adapter, InstantID, etc.
- **ControlNet**: Load a ControlNet model (`controlnet.safetensors`) in `models/controlnet/`, use the `ControlNetApply` node.
- **AnimateDiff**: Generate videos by adding `AnimateDiffLoader` and `AnimateDiffSampler` nodes; motion modules go in `models/animatediff_models/`.
- **Video generation**: Use `VideoCombine` or `FFMPEGSave` nodes to stitch frames into MP4.
- **Workflow sharing**: Export as "API format" (checkbox in Save dialog) for headless execution.
- **Extra models**: Place LoRAs in `models/loras/`, Embeddings in `models/embeddings/`, VAE in `models/vae/`.
- **Command-line arguments**: `--gpu-only`, `--highvram`, `--force-fp16`, `--multi-user` for server mode.

## Integration with the WFB Model Project

```python
# wfb_model/comfyui/generate.py
import json, requests, os

def generate_image(prompt: str, workflow_path: str = "workflows/txt2img.json"):
    with open(os.path.join("wfb_model", workflow_path)) as f:
        workflow = json.load(f)
    # Override prompt text
    workflow["6"]["inputs"]["text"] = prompt
    requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
```

WFB uses ComfyUI to generate synthetic training data (e.g., domain-specific images), augment datasets with ControlNet-guided variations, and produce visual outputs from diffusion model experiments. Generated images are stored under `wfb_model/data/generated/`.

## Common Pitfalls / Troubleshooting

- **Black output images**: Usually a VAE mismatch. Ensure the VAE matches the checkpoint (e.g., `vae-ft-mse-840000-ema-pruned.safetensors` for SD 1.5).
- **CUDA out of memory**: Reduce `EmptyLatentImage` dimensions or lower `batch_size`. Use `--lowvram` or `--normalvram` flags.
- **Custom nodes not loading**: Run `python ComfyUI-Manager/cm-cli.py fix` or reinstall node dependencies manually.
- **API prompt not executing**: Ensure the workflow was saved in "API format" (checkbox in ComfyUI). Regular workflow JSON includes UI metadata that breaks the API endpoint.
- **Slow generation**: Enable `--force-fp16` and use `xformers` or `--force-attention-split-opt`. For SDXL consider `--sdxl`.
- **Port already in use**: Change port with `--port 8282`.

## Documentation Links

- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- ComfyUI-Manager: https://github.com/ltdrdata/ComfyUI-Manager
- Workflow examples: https://comfyanonymous.github.io/ComfyUI_examples/
- Node reference: https://blobz.github.io/ComfyUI-Nodes-Docs/
