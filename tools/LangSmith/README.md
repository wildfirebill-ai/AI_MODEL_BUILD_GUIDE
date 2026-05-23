# LangSmith — Debug, Test & Monitor LLM Applications

Platform for tracing, evaluating, and monitoring LLM applications. Integrates with LangChain and standalone usage.

## Installation

```bash
pip install langsmith langchain
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_TRACING=true
```

## Tracing

```python
from langsmith import traceable
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

@traceable(run_type="chain", name="my_chain")
def my_chain(question: str) -> str:
    return llm.invoke(question).content

my_chain("What is LangSmith?")

# Manual run tree
from langsmith.run_trees import RunTree

run = RunTree(name="parent", run_type="chain", inputs={"text": "Analyze this"})
child = run.create_child(name="llm_call", run_type="llm", inputs={"prompts": ["Analyze this"]})
child.end(outputs={"response": "Positive"})
run.end(outputs={"result": "done"})
run.post()
```

## Datasets & Evaluation

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(dataset_name="QA_Evaluation")
client.create_examples(
    dataset_id=dataset.id,
    inputs=[{"question": "What is the capital of France?"}],
    outputs=[{"answer": "Paris"}],
)

from langchain.smith import RunEvalConfig, run_on_dataset

eval_config = RunEvalConfig(evaluators=["qa", "coherence"], prediction_key="answer")
results = run_on_dataset(
    dataset_name="QA_Evaluation",
    llm_or_chain_factory=lambda: llm,
    evaluation=eval_config,
)
```

## Feedback

```python
client.create_feedback(
    run_id="your-run-id",
    key="user_score",
    score=0.85,
    comment="Good response",
)
```

## A/B Comparison

```python
run_on_dataset(
    dataset_name="QA_Evaluation",
    llm_or_chain_factory=lambda: ChatOpenAI(model="gpt-4"),
    project_name="gpt-4-eval",
)
run_on_dataset(
    dataset_name="QA_Evaluation",
    llm_or_chain_factory=lambda: ChatOpenAI(model="gpt-3.5-turbo"),
    project_name="gpt-3.5-eval",
)  # Compare in UI
```

## Prompt Hub

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
hub.push("my-org/custom-prompt", prompt)
```

## Key Features

| Feature | Description |
|---------|-------------|
| Traces | End-to-end request tracking |
| Runs | Individual span in a trace |
| Datasets | Versioned test examples |
| Evaluations | Automated scoring |
| Feedback | Human/LLM feedback |
| Hub | Version-controlled prompts |

## Integration

Works standalone via `traceable` decorator or with LangChain. Use datasets for regression testing before deploying prompt changes.
