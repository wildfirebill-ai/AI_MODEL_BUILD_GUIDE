# Replicate — Cloud Platform for Running ML Models

[Replicate](https://replicate.com) provides serverless, pay-per-use inference for thousands of ML models. Deploy custom models or use community models via API.

## Installation

```bash
pip install replicate
```

Get your API token from [replicate.com/account](https://replicate.com/account).

## Quick Start — `replicate.run()`

```python
import replicate

output = replicate.run(
    "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    input={"prompt": "a cat wearing a hat"}
)
print(output)
# [URL to generated image]
```

## Replicate Client

```python
from replicate import Client

client = Client(api_token="your-token")
output = client.run(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"prompt": "Hello, how are you?"}
)
print("".join(output))
```

## Streaming Output

```python
import replicate

for event in replicate.stream(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"prompt": "Tell me a story"}
):
    print(event, end="")
```

## Predictions — Async & Webhooks

```python
import replicate

prediction = replicate.predictions.create(
    version="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"prompt": "Explain quantum computing"},
    webhook="https://example.com/webhook",
    webhook_events_filter=["completed"]
)

# Poll for result
prediction.wait()
print(prediction.output)
```

## Model Deployments — Custom Models

Cog is Replicate's deployment tool:

```bash
pip install cog

# Initialize cog project
cog init

# Build & push
cog build
cog push r8.im/your-username/your-model
```

```python
# predict.py (Cog format)
import cog
from diffusers import StableDiffusionPipeline

class Predictor(cog.Predictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    @cog.input("prompt", type=str)
    @cog.input("num_steps", type=int, default=50)
    def predict(self, prompt, num_steps):
        return self.pipe(prompt, num_inference_steps=num_steps).images[0]
```

## Fine-Tuning API

```python
import replicate

training = replicate.trainings.create(
    version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "input_images": "https://example.com/dataset.zip",
        "prompt": "a photo of TOK person",
    },
    destination="your-username/fine-tuned-model",
)
training.wait()
print(f"Fine-tuned model: {training.output['version']}")
```

## Listing Models & Versions

```python
from replicate import Client

client = Client()
models = client.models.list()
for model in models:
    print(f"{model.owner}/{model.name}")

# Get specific version
version = client.models.get_version("meta/llama-2-70b-chat", "02e509c...")
print(version)
```

## Webhook Events

```python
# Flask webhook receiver
from flask import Flask, request

app = Flask(__name__)

@app.post("/webhook")
def webhook():
    data = request.json
    prediction_id = data["id"]
    status = data["status"]
    print(f"Prediction {prediction_id}: {status}")
    return "OK", 200
```

## Environment Setup

```bash
export REPLICATE_API_TOKEN=r8_...
```

Or use `.env`:

```
REPLICATE_API_TOKEN=r8_...
```

## Use Cases

- **Serverless inference** — no infrastructure management
- **Model API generation** — turn any model into an API endpoint
- **Batch processing** — run predictions at scale
- **Fine-tuning** — customize models with your data
