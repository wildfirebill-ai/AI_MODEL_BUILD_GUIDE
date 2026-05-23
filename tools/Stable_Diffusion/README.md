# Stable Diffusion — Text-to-Image with Diffusers

**Version:** 0.31 / `diffusers>=0.31.0`

## Purpose

The Hugging Face Diffusers library provides production-ready pipelines for Stable Diffusion models: text-to-image, image-to-image, inpainting, depth-to-image, and video generation. It integrates schedulers (DDIM, PNDM, Euler, DPM++, LCM), LoRA adapters, ControlNet conditioning, and support for SD 1.5/2.1, SDXL, SD3, and Flux. Used for generating synthetic datasets, augmenting training data, and experimenting with diffusion model customization.

## Installation

```bash
pip install "diffusers[torch]>=0.31.0" transformers accelerate peft
# For xformers (memory-efficient attention):
pip install xformers
```

Requires PyTorch 2.3+ and CUDA 12.1 (or CPU fallback). Use a Hugging Face token for gated models (`huggingface-cli login`).

## Basic Usage Example

```python
import torch
from diffusers import StableDiffusionPipeline

# Load the SD 1.5 pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable NSFW filter for research
)
pipe = pipe.to("cuda")

# Enable memory optimization
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

# Generate
image = pipe(
    prompt="a cat astronaut riding a rocket, digital art, 4k",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
).images[0]

image.save("cat_astronaut.png")
```

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("input.png").resize((512, 512))
result = pipe(prompt="make it a watercolor painting", image=init_image, strength=0.6).images[0]
result.save("output.png")
```

## Advanced Usage / Configuration

- **LoRA adapters**: `pipe.load_lora_weights("lora-weight-path")` for style or concept fine-tuning without merging.
- **ControlNet**: Combine `diffusers.StableDiffusionControlNetPipeline` with a `ControlNetModel` preprocessor (canny, depth, pose, scribble).
- **Scheduler swap**: `pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)` for 4–8 step generation.
- **Textual Inversion**: Load embeddings via `pipe.load_textual_inversion("sd-concepts-library/cat-toy")` to trigger custom concepts.
- **SDXL**: Use `StableDiffusionXLPipeline` with dual text encoders and 1024x1024 outputs.
- **FreeU**: Call `pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)` for improved image quality without additional cost.
- **Inpainting**: Use `StableDiffusionInpaintPipeline` with `image` and `mask_image` PIL inputs.

## Integration with the WFB Model Project

```python
# wfb_model/diffusion/generate_batch.py
from diffusers import StableDiffusionPipeline
import torch

class WFBImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate_batch(self, prompts: list[str], out_dir: str):
        for i, prompt in enumerate(prompts):
            img = self.pipe(prompt, num_inference_steps=25).images[0]
            img.save(f"{out_dir}/{i:04d}.png")
```

WFB uses this to create synthetic training images for domain adaptation and data augmentation. Generated datasets are stored in `wfb_model/data/diffusion/` and registered in the project's data catalog.

## Common Pitfalls / Troubleshooting

- **NSFW filter blocking output**: Disable with `safety_checker=None` during pipeline init (only for research/non-production).
- **CUDA OOM**: Use `pipe.enable_attention_slicing(slice_size="max")`, `pipe.enable_model_cpu_offload()`, or reduce resolution to 512×512.
- **Reproducibility**: Set `generator=torch.Generator(device="cuda").manual_seed(seed)` and `num_inference_steps` exactly.
- **Latent mismatch**: SD 1.5/2.1 need 512×512, SDXL needs 1024×1024. Wrong sizes produce artifacts.
- **Model download fails**: Use `snapshot_download("model-id", resume_download=True)` for large model files. Proxy: set `HF_ENDPOINT=https://hf-mirror.com`.
- **Slow generation**: Swap scheduler to `DPMSolverMultistepScheduler` (8–15 steps) or `LCMScheduler` (1–4 steps). Enable `torch.compile`.

## Documentation Links

- Diffusers docs: https://huggingface.co/docs/diffusers
- Pipeline API: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview
- LoRA fine-tuning: https://huggingface.co/docs/diffusers/main/en/training/lora
- ControlNet guide: https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlnet
