# Argilla — Data Annotation & Feedback for LLMs

Platform for collecting human feedback on LLM outputs—supports RLHF preference data, evaluation datasets, and LLM-as-judge workflows.

## Installation

```bash
pip install argilla
```

## Quick Start — Dataset Creation

```python
import argilla as rg

rg.init(api_url="http://localhost:6900", api_key="admin.apikey")

dataset = rg.Dataset(
    name="sentiment_analysis",
    settings=rg.DatasetSettings(
        guidelines="Classify as POSITIVE, NEGATIVE, or NEUTRAL.",
        fields=[rg.TextField(name="text")],
        questions=[rg.LabelQuestion(name="sentiment", labels=["POSITIVE", "NEGATIVE", "NEUTRAL"])],
    ),
)
dataset.create()

records = [
    rg.Record(fields={"text": "This product is amazing!"}),
    rg.Record(fields={"text": "Worst experience ever."}),
]
dataset.records.log(records)
```

## Feedback Dataset for RLHF

```python
feedback_ds = rg.FeedbackDataset(
    name="llm_preferences",
    fields=[
        rg.TextField(name="prompt"),
        rg.TextField(name="response_a"),
        rg.TextField(name="response_b"),
    ],
    questions=[
        rg.RankingQuestion(name="preference", values={"response_a": "A", "response_b": "B"}),
    ],
)
feedback_ds.create()

feedback_ds.add_records([
    rg.FeedbackRecord(fields={"prompt": "Explain AI", "response_a": "AI is...", "response_b": "It's..."}),
])
```

## LLM-as-Judge

```python
def llm_judge(prompt, response_a, response_b):
    import openai
    result = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Which is better? A: {response_a} B: {response_b}"}],
    )
    return result.choices[0].message.content.strip()

for record in feedback_ds.records():
    pref = llm_judge(record.fields["prompt"], record.fields["response_a"], record.fields["response_b"])
    record.annotations = {"preference": pref}
    record.update()
```

## Export

```python
df = dataset.records().to_pandas()
hf_dataset = dataset.to_datasets()
dataset.records(query=rg.Query(query="status:completed")).to_json("annotations.json")
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `Dataset` | Records with fields/questions |
| `FeedbackDataset` | Preference/reward data |
| `Record` | Fields + annotations |
| `Question` | Label, ranking, or text question |

## Integration

Use Argilla for RLHF preference collection and model evaluation. LLM-as-judge enables auto-annotation. Export to Hugging Face Datasets for training.
