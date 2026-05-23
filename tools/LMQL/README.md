# LMQL (Language Model Query Language)

**Version:** 0.7+

**Purpose:** Programming language for LLM interaction with constraints, control flow, and structured output.

## Installation

```powershell
pip install lmql
```

## Usage

```python
import lmql

# Constrained generation
@lmql.query
async def classify_sentiment(text):
    '''lmql
    "Review: {text}\nSentiment:" [SENTIMENT]
    where SENTIMENT in ["positive", "negative", "neutral"]
    '''

result = await classify_sentiment("This movie was amazing!")
print(result.variables["SENTIMENT"])  # "positive"

# Multi-step with constraints
@lmql.query
async def extract_info(text):
    '''lmql
    "Text: {text}\n"
    "Name:" [NAME] "\n"
    "Age:" [AGE]
    where AGE > 0 and AGE < 150
      and len(NAME) > 0
    '''
```

## Key Features

- **Constraint-aware generation** (type, regex, custom)
- **Multi-variable extraction**
- **Control flow** (loops, conditions)
- **Model-agnostic** (OpenAI, HuggingFace, local)

## Documentation

- https://lmql.ai/
