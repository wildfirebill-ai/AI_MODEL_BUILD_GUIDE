# Guardrails AI — Validation & Guardrails for LLM Outputs

[Guardrails AI](https://www.guardrailsai.com) is a Python framework for adding structure, validation, and safety checks to LLM outputs. Define a RAIL spec and Guardrails handles reasking, fixing, and enforcing constraints.

## Installation

```bash
pip install guardrails-ai
```

## Quick Start — Simple Regex Validation

```python
import guardrails as gd

rail_spec = """
<rail version="0.1">
<output>
    <string name="greeting" format="valid-choices">
        <choices>["Hello", "Hi", "Hey"]</choices>
    </string>
</output>
<prompt>
Generate a greeting for the user.
</prompt>
</rail>
"""

guard = gd.Guard.from_rail_string(rail_spec)
raw, validated = guard(llm_api=openai.chat.completions.create, model="gpt-4")
print(validated)  # {"greeting": "Hello"}
```

## Guard Object

The `Guard` wraps any LLM call. It validates both the prompt and the output.

```python
import guardrails as gd
from openai import OpenAI

client = OpenAI()

guard = gd.Guard.from_rail_string("""
<rail version="0.1">
<output>
    <integer name="age" format="valid-range" min="0" max="150" />
</output>
<prompt>
Tell me the age of Einstein when he died.
</prompt>
</rail>
""")

raw, validated = guard(llm_api=client.chat.completions.create, model="gpt-4")
print(validated)  # {"age": 76}
```

## RAIL Spec XML Schema

Define expected output structure and validators:

```xml
<rail version="0.1">
<output>
    <object name="person">
        <string name="name" format="valid-choices">
            <choices>["Alice", "Bob", "Carol"]</choices>
        </string>
        <integer name="age" format="valid-range" min="0" max="120" />
        <string name="email" format="valid-regex" regex="^[\w\.-]+@[\w\.-]+\.\w+$" />
    </object>
</output>
<prompt>
Extract person info: {{output_schema}}
{{user_input}}
</prompt>
</rail>
```

## Reask — Automatic Retry on Validation Failure

When output fails validation, Guardrails can re-prompt the LLM with the error message:

```python
guard = gd.Guard.from_rail_string(rail_spec)
raw, validated = guard(
    llm_api=client.chat.completions.create,
    model="gpt-4",
    num_reasks=2,  # retry up to 2 times
)
```

## Fix — Correcting Outputs Automatically

Some validators support automatic fixes:

```python
raw, validated = guard(
    llm_api=client.chat.completions.create,
    model="gpt-4",
    on_fail="fix",  # try to fix instead of reasking
)
```

Available policies: `reask` (default), `fix`, `noop`, `exception`.

## Built-in Validators

```python
from guardrails.validators import (
    ValidRange,
    ValidChoices,
    RegexMatch,
    IsLowerCase,
    IsHighQuality,  # quality scoring
    reading_time,   # safe reading level
)

# Use in RAIL spec via format attribute:
# format="valid-range"  → ValidRange
# format="valid-choices" → ValidChoices
# format="is-lower-case" → IsLowerCase
```

## Safety & Quality Validators

```python
rail_spec = """
<rail version="0.1">
<output>
    <string name="response" format="is-high-quality" />
</output>
<prompt>
{{user_input}}
</prompt>
</rail>
"""

guard = gd.Guard.from_rail_string(rail_spec)
_, validated = guard(
    llm_api=client.chat.completions.create,
    model="gpt-4",
    prompt_params={"user_input": "Tell me about quantum physics"},
)
```

## Structured Output Enforcement

Combine with Pydantic-style schemas:

```python
import guardrails as gd

rail_spec = """
<rail version="0.1">
<output>
    <list name="items">
        <object>
            <string name="product" />
            <float name="price" format="valid-range" min="0" max="10000" />
            <integer name="quantity" format="valid-range" min="1" max="100" />
        </object>
    </list>
</output>
<prompt>
Extract order items from: {{user_text}}
</prompt>
</rail>
"""
```

## Integration with LangChain

```python
from guardrails import Guard
from langchain.chat_models import ChatOpenAI

guard = Guard.from_rail_string(rail_spec)
llm = ChatOpenAI(model="gpt-4")

raw, validated = guard(llm_api=llm.predict, prompt="...")
```

## Use Cases

- **LLM output validation** — ensure JSON matches schema
- **Safety constraints** — block toxic or unsafe responses
- **Structured extraction** — reliably parse entities from text
- **Quality gates** — reject low-quality or off-topic responses
