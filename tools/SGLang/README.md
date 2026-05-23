# SGLang

**Version:** 0.3+

**Purpose:** Structured generation language and runtime for efficient LLM serving with constrained decoding.

## Installation

```powershell
pip install sglang
```

## Usage

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question, context):
    # Structured generation with constraints
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(f"Context: {context}\n\nQuestion: {question}")
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))

    # Constrained JSON output
    s += sgl.user("Extract the key information as JSON:")
    s += sgl.assistant(
        sgl.gen("json_output", max_tokens=256, regex=r'\{.*\}')
    )

# Run
state = multi_turn_qa.run(
    question="What is RLHF?",
    context="RLHF stands for Reinforcement Learning from Human Feedback...",
)
print(state["answer"])
print(state["json_output"])
```

## Key Features

- **RadixAttention** -- prefix caching for shared prompt prefixes
- **Structured generation** -- regex, JSON, grammar constraints
- **Automatic prefix caching** with LRU eviction
- **3-5x higher throughput** vs vanilla vLLM on chat workloads

## Documentation

- https://github.com/sgl-project/sglang
