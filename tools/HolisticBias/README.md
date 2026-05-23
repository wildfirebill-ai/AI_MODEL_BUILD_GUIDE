# HolisticBias

**Purpose:** Evaluate and measure bias in language model outputs across demographic axes (gender, race, religion, age, nationality, etc.).

## Installation

```powershell
pip install holistic-bias
```

## Usage

```python
from holistic_bias import HolisticBias

# Evaluate a model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

hb = HolisticBias(model=model, tokenizer=tokenizer)

# Run full evaluation
results = hb.evaluate_all()

for axis, data in results.items():
    print(f"{axis}: score = {data['score']:.3f}")

# Individual axis evaluation
gender_results = hb.evaluate("gender")
race_results = hb.evaluate("race")

# Custom evaluation setup
results = hb.evaluate_all(
    axes=["gender", "race", "religion", "age"],
    num_samples=100,
)
```

## Supported Axes

| Axis | Description |
|------|-------------|
| gender | Male/female associations |
| race | Racial/ethnic stereotypes |
| religion | Religious bias |
| age | Age discrimination |
| nationality | Nationality stereotypes |
| sexuality | Sexual orientation bias |
| disability | Disability representation |

## Documentation

- https://github.com/nyu-mll/holistic-bias
