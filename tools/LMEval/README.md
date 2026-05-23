# LM Evaluation Harness

**Version:** 0.4+

**Purpose:** Standardized evaluation framework for LLMs across 200+ benchmarks.

## Installation

```powershell
pip install lm-eval
```

## Usage

### CLI

```powershell
# Single benchmark
lm_eval --model hf --model_args pretrained=./model,tokenizer=./tokenizer --tasks mmlu --num_fewshot 5

# Multiple benchmarks
lm_eval --model hf --model_args pretrained=./model --tasks mmlu,gsm8k,humaneval,boolq

# With custom model
lm_eval --model local-completions --model_args model_path=./model --tasks mmlu
```

### Python API

```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=./model,tokenizer=./tokenizer",
    tasks=["mmlu", "gsm8k"],
    batch_size=4,
    num_fewshot=5,
)

for task, result in results["results"].items():
    print(f"{task}: {result.get('acc', result.get('exact_match', 'N/A')):.2%}")
```

## Key Benchmarks

| Benchmark | Type | Metrics |
|-----------|------|---------|
| MMLU / MMLU-Pro | 57 subjects | Accuracy |
| GSM8K / MATH | Math reasoning | Exact match |
| HumanEval / MBPP | Code generation | Pass@k |
| HellaSwag / ARC | Commonsense | Accuracy |
| TruthfulQA | Factuality | MC / generative |
| BBH | Big-Bench Hard | Accuracy |
| IFEval | Instruction following | Strict / loose |

## Documentation

- https://github.com/EleutherAI/lm-evaluation-harness
