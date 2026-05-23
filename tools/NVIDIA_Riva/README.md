# NVIDIA Riva — GPU-Accelerated Speech AI SDK

[Riva](https://developer.nvidia.com/riva) provides production-grade ASR (speech-to-text), TTS (text-to-speech), and NMT (translation) powered by NVIDIA GPUs.

## Installation

```bash
# Pull the Riva Docker image
docker pull nvcr.io/nvidia/riva/riva-speech:2.14.0

# Start the server (with GPU support)
docker run --gpus all -p 50051:50051 nvcr.io/nvidia/riva/riva-speech:2.14.0
```

Install the Python client:

```bash
pip install nvidia-riva-client
```

## Quick Start — ASR (Speech-to-Text)

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")
asr_service = riva.client.SpeechRecognitionService(auth)

with open("audio.wav", "rb") as f:
    audio_data = f.read()

config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="en-US",
    max_alternatives=1,
)

response = asr_service.recognize(audio_data, config)
print(response.results[0].alternatives[0].transcript)
```

## Streaming ASR

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")
asr_service = riva.client.SpeechRecognitionService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en-US",
    ),
    interim_results=True,
)

responses = asr_service.streaming_response_from_file("audio.wav", config)
for resp in responses:
    for result in resp.results:
        print(f"[{'final' if result.is_final else 'interim'}] {result.alternatives[0].transcript}")
```

## TTS (Text-to-Speech)

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")
tts_service = riva.client.SpeechSynthesisService(auth)

response = tts_service.synthesize(
    text="Hello, welcome to NVIDIA Riva.",
    voice_name="en-US-News-W",
    language_code="en-US",
)

with open("output.wav", "wb") as f:
    f.write(response.audio)
```

## Customizing Voice Parameters

```python
response = tts_service.synthesize(
    text="Speak faster with higher pitch.",
    voice_name="en-US-News-W",
    language_code="en-US",
    sample_rate_hertz=24000,
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    pitch_shift=1.2,       # pitch multiplier
    speaking_rate=1.3,     # speed multiplier
    volume_gain_db=3.0,    # volume boost
)
```

## NMT (Neural Machine Translation)

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")
nmt_service = riva.client.NeuralMachineTranslationClient(auth)

response = nmt_service.translate(
    text="How are you today?",
    source_language="en",
    target_language="es",
)
print(response.translations[0].translation)
# "¿Cómo estás hoy?"
```

## Custom Acoustic & Language Models

```bash
# Generate Riva-compatible models from NeMo
docker run --gpus all -v /path/to/models:/models nvcr.io/nvidia/riva/riva-speech:2.14.0 \
  --model-repo /models
```

Reference custom models:

```python
config = riva.client.RecognitionConfig(
    model="custom_asr_model",
    custom_config={
        "acoustic_model": "path/to/acoustic_model",
        "language_model": "path/to/lm.binary",
    },
)
```

## Riva API (gRPC vs HTTP)

Client uses **gRPC** by default (port 50051). HTTP REST also available:

```python
# gRPC (default, recommended for production)
auth = riva.client.Auth(uri="localhost:50051")

# HTTP REST
import requests
response = requests.post(
    "http://localhost:8001/v1/synthesize",
    json={"text": "Hello", "voice": "en-US-News-W"},
)
```

## Use Cases

- **Voice interfaces for LLMs** — STT input → LLM → TTS output
- **Call center analytics** — real-time transcription and sentiment
- **Accessibility tools** — screen readers, captioning systems
- **Multilingual applications** — translate and speak in any language
- **Custom wake-word detection** — train on domain-specific vocabulary
