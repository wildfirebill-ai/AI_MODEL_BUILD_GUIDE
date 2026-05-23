# FastAPI + Uvicorn

**Purpose:** High-performance Python web framework for LLM inference APIs.

---

## FastAPI

### Installation

```powershell
pip install fastapi uvicorn pydantic
```

### Usage

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/v1/completions")
async def generate(req: GenerateRequest):
    # Your inference code here
    return {"text": "Generated response"}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Uvicorn

```powershell
# Basic
uvicorn server:app --reload --port 8000

# Production
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop

# With HTTPS
uvicorn server:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

---

## Streaming with SSE

```python
from fastapi.responses import StreamingResponse
import asyncio

async def token_generator(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    for _ in range(100):
        logits = model(inputs)[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        yield tokenizer.decode(next_tok[0])
        inputs = torch.cat([inputs, next_tok], dim=-1)

@app.post("/v1/chat/completions")
async def chat(req: GenerateRequest):
    return StreamingResponse(
        token_generator(req.prompt),
        media_type="text/event-stream",
    )
```

## Documentation

- https://fastapi.tiangolo.com/
