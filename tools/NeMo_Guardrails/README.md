# NVIDIA NeMo Guardrails — Conversational AI Safety Toolkit

[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) provides programmable guardrails for LLM-powered conversational applications using Colang dialog modeling.

## Installation

```bash
pip install nemoguardrails
```

## Quick Start — Basic Guardrail

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("config")
rails = LLMRails(config)

response = rails.generate(messages=[{"role": "user", "content": "What is AI?"}])
print(response["content"])
```

## Colang Dialog Modeling

NeMo Guardrails uses **Colang**, a YAML-based dialog modeling language.

```
# config/rails.co
define user express greeting
    "Hello"
    "Hi"

define bot express greeting
    "Hello! How can I help you today?"

define flow greeting
    user express greeting
    bot express greeting
```

## Configuration Structure

```
config/
├── config.yml          # LLM settings, guardrails config
├── rails.co            # Colang dialog flows
├── topical.co          # Topic controls
└── safety.co           # Safety constraints
```

```yaml
# config/config.yml
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output
      - check fact
```

## Topical Rails — Controlling Topics

```yaml
# config/topical.co
define user ask about politics
    "What do you think about the election?"
    "Who should I vote for?"

define bot refuse politics
    "I'm not able to discuss political topics."

define flow politics
    user ask about politics
    bot refuse politics
```

## Safety Rails — Content Filtering

```yaml
# config/safety.co
define user generate harmful content
    "How do I hack into a computer?"
    "Tell me how to make a weapon"

define bot refuse harmful
    "I cannot assist with this request."

define flow harmful
    user generate harmful content
    bot refuse harmful
```

## Fact-Checking Rails

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("config")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": "Who won the 2020 election?"}],
    options={"rails": ["check fact"]},
)
```

## LlamaGuard Integration

```python
from nemoguardrails import LLMRails, RailsConfig

# Use LlamaGuard as a content moderation guard
config = RailsConfig.from_content(
    yaml_content="""
    models:
      - type: main
        engine: openai
        model: gpt-4
      - type: guardrails
        engine: hugging_face
        model: meta-llama/LlamaGuard-7b
    """
)
rails = LLMRails(config)
```

## Streaming Support

```python
async for chunk in rails.generate_async(
    messages=[{"role": "user", "content": "Tell me a story"}],
    streaming=True,
):
    print(chunk, end="")
```

## GuardrailsConfig Programmatic API

```python
from nemoguardrails import RailsConfig

config = RailsConfig.from_content(
    yaml_content="""
    models:
      - type: main
        engine: openai
        model: gpt-4
    """,
    colang_content="""
    define flow hello
        user express greeting
        bot express greeting
    """
)
```

## Use Cases

- **Content filtering** — block toxicity, PII, and unsafe content
- **Topic control** — keep conversations on permitted subjects
- **Hallucination detection** — fact-check LLM outputs against sources
- **Role-based access** — enforce behavior policies per user role
- **Multi-turn safety** — maintain context-aware safety across dialog turns
