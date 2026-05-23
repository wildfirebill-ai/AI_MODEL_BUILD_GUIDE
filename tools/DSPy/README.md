# DSPy

**Version:** 2.6+

**Purpose:** Programming framework for building LLM applications through declarative module composition and automatic prompt optimization. DSPy replaces manual prompt engineering with signatures, compiled modules (`dspy.ChainOfThought`, `dspy.ReAct`, `dspy.ProgramOfThought`), and teleprompters that optimize prompts for a given metric. In the WFB pipeline, DSPy optimizes evaluation prompts and builds structured LLM programs for benchmark scoring.

## Installation

```powershell
pip install "dspy==2.6.0"
```

## Basic Usage

### ChainOfThought Module

```python
import dspy

lm = dspy.LM("openai/gpt-4o", api_key="...", temperature=0.0)
dspy.configure(lm=lm)

class Summarize(dspy.Signature):
    """Summarize a document into 3 bullet points."""
    document: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="3 bullet points")

class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(Summarize)
    def forward(self, document):
        return self.summarize(document=document)

result = Summarizer()(document="WFB project uses JAX for pretraining...")
print(result.summary)
```

### Using Local Model via vLLM / TGI

```python
lm = dspy.LM("openai/meta-llama/Llama-3.1-70B",
    api_key="",
    base_url="http://localhost:8000/v1",
    model_type="chat",
)
dspy.configure(lm=lm)
```

## Advanced Usage

### ReAct Agent

```python
class SearchTool(dspy.Tool):
    def __call__(self, query: str) -> str:
        return f"Results for: {query}"

class WFBQA(dspy.Module):
    def __init__(self):
        self.react = dspy.ReAct(
            signature="question: str -> answer: str",
            tools=[SearchTool()],
        )
    def forward(self, question):
        return self.react(question=question)
```

### Prompt Optimization

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=lambda g, p, trace: g.answer == p.answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)
compiled = teleprompter.compile(AnswerVerifier(), trainset=trainset)
compiled.save("./optimized_pipeline.json")
```

### ProgramOfThought for Structured Output

```python
class ExtractMetrics(dspy.Signature):
    text: str = dspy.InputField()
    metrics: str = dspy.OutputField(desc="JSON: accuracy, latency, throughput")

class MetricExtractor(dspy.Module):
    def __init__(self):
        self.program = dspy.ProgramOfThought(ExtractMetrics)
    def forward(self, text):
        return self.program(text=text)
```

## Integration with WFB Model

DSPy is used in **Part 6** (Post-Training) and **Part 12** (Advanced Evaluation & Safety) of the WFB pipeline. After SFT, DSPy replaces hand-written eval prompts with compiled modules that consistently extract metrics (factuality, helpfulness, safety). Use DSPy optimizers to find the best prompt formulation for reward model scoring during RLHF. During alignment evaluation, `ReAct` agents run standardized safety benchmarks with tool-use for external knowledge retrieval.

## Common Pitfalls

- **Undefined LM**: Always call `dspy.configure(lm=lm)` before module calls.
- **OpenAI-only assumptions**: DSPy supports any OpenAI-compatible endpoint. Set `base_url` for local models.
- **Metric design**: Poor metrics produce poor optimizations. Design task-specific metrics.
- **Cache pollution**: Clear with `dspy.settings.cache.clear()` during development.
- **Cost control**: Optimization runs 100+ LM calls. Use local models or set log level to INFO.

## Documentation

- https://dspy.ai/
- https://github.com/stanfordnlp/dspy
