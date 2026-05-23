# Outlines / LMQL / Guidance

**Purpose:** Libraries for structured generation -- JSON, regex, grammar-constrained output.

---

## Outlines

### Installation

```powershell
pip install outlines
```

### Usage

```python
import outlines

# JSON Schema generation
schema = """{
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
}"""
generator = outlines.generate.json(model, schema)
result = generator("Create a person")

# Regex generation
regex_generator = outlines.generate.regex(model, r"\d{3}-\d{3}-\d{4}")
phone = regex_generator("Generate a phone number")

# Choice selection
choice_generator = outlines.generate.choice(model, ["positive", "negative", "neutral"])
sentiment = choice_generator("This movie was great!")
```

---

## LMQL

### Installation

```powershell
pip install lmql
```

### Usage

```python
import lmql

@lmql.query
async def colors():
    '''lmql
    "List 3 colors:" [COLORS]
    where len(COLORS.split(",")) == 3
      and all(c in ["red","green","blue","yellow"] for c in COLORS.split(","))
    '''
```

---

## Guidance

### Installation

```powershell
pip install guidance
```

### Usage

```python
import guidance

program = guidance("""Answer: {{#select "answer"}}yes{{/select}}""")
result = program()
```
