# Pydantic

**Version:** 2.7+

**Purpose:** Data validation and settings management using Python type hints. Essential for FastAPI-based LLM serving.

## Installation

```powershell
pip install pydantic
```

## Usage for LLM API Schemas

```python
from pydantic import BaseModel, Field, PositiveInt, confloat
from typing import Optional, List, Literal

class GenerationRequest(BaseModel):
    model: str = Field(default="wfb-model", description="Model to use")
    prompt: str = Field(min_length=1, max_length=131072)
    max_tokens: PositiveInt = Field(default=256, le=8192)
    temperature: confloat(ge=0.0, le=2.0) = 0.7
    top_p: confloat(ge=0.0, le=1.0) = 0.95
    top_k: Optional[PositiveInt] = 50
    stop: Optional[List[str]] = None
    stream: bool = False

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class GenerationResponse(BaseModel):
    id: str
    object: str = "text_completion"
    choices: List[dict]
    usage: TokenUsage

# Auto-validation on request
@app.post("/v1/completions")
async def generate(req: GenerationRequest):
    # req is already validated
    pass
```

## Key Features

- **Runtime type validation** -- catch errors early
- **JSON Schema generation** -- auto OpenAPI docs
- **Serialization** -- `.model_dump()` and `.model_dump_json()`
- **Strict mode** -- `model_config = ConfigDict(strict=True)`

## Documentation

- https://docs.pydantic.dev/
