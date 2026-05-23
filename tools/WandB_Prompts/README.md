# W&B Prompts — LLM Prompt Engineering & Monitoring

Trace, visualize, and optimize LLM interactions with prompt/response logging, versioning, and chain visualization.

## Installation

```bash
pip install wandb
wandb login
```

## Quick Start — Automatic Tracing

```python
import wandb
from wandb.integration.openai import autolog

autolog({"project": "llm-monitoring"})

import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

## Manual Prompt Logging

```python
import wandb

wandb.init(project="prompt-engineering")

wandb.log({
    "prompt": wandb.Prompt(
        prompt="Translate to French: Hello",
        response="Bonjour",
        model="gpt-4",
        token_usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        latency=0.45,
    )
})
```

## Chain Visualization

```python
from wandb.sdk.data_types.trace_tree import Trace

wandb.init(project="chain-tracing")

root = Trace(name="rag_pipeline", kind="chain", status_code="success")

llm_span = Trace(
    name="openai_call", kind="llm", status_code="success",
    metadata={"model": "gpt-4", "usage": {"prompt_tokens": 150, "completion_tokens": 50}},
    inputs={"prompts": ["What is RAG?"]},
    outputs={"response": "RAG stands for Retrieval-Augmented Generation..."},
)

retrieval_span = Trace(
    name="vector_search", kind="tool", status_code="success",
    inputs={"query": "What is RAG?"},
    outputs={"results": ["doc1", "doc2"]},
)

llm_span.add_child(retrieval_span)
root.add_child(llm_span)
root.log(name="rag_trace")
```

## Prompt Versioning

```python
wandb.init(project="prompt-management", job_type="prompt_versioning")

templates = {"v1": "Answer: {q}", "v2": "You are helpful. Answer: {q}"}
for version, template in templates.items():
    artifact = wandb.Artifact(f"prompt-{version}", type="prompt")
    artifact.add_file(wandb.util.make_artifact(content=template, file_name="prompt.txt"))
    wandb.log_artifact(artifact)
```

## A/B Testing & Cost Monitoring

```python
wandb.init(project="ab-testing")

for version, template in {"concise": "Answer briefly: {q}", "detailed": "Explain in depth: {q}"}.items():
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": template.format(q="What is ML?")}],
    )
    wandb.log({
        f"latency_{version}": response.usage.completion_tokens,
        "cost": (response.usage.prompt_tokens / 1000 * 0.03 + response.usage.completion_tokens / 1000 * 0.06),
    })
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `wandb.Prompt` | Log prompt/response pairs |
| `Trace` (tree) | Span-based chain visualization |
| Autolog | Automatic OpenAI/LangChain capture |
| Prompt Versioning | Artifact-based prompt mgmt |
| Token Usage | Cost and consumption tracking |

## Integration

Enable `autolog()` for automatic capture. Use `wandb.Prompt` for detailed tracking. Build `Trace` trees for multi-step agents. W&B Reports turn metrics into team dashboards.
