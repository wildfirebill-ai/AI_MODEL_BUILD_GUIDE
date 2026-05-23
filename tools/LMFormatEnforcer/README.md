# LM Format Enforcer — Structured Output from LLMs

**Version:** 0.10 / `lm-format-enforcer>=0.10.0`

## Purpose

LM Format Enforcer enforces structured output from text-generation LLMs at the token level. It works by intercepting the model's logits before sampling and zeroing out any tokens that would violate a specified grammar, JSON schema, or regular expression. Supports character-level enforcement for JSON, YAML, XML, Python code, and arbitrary context-free grammars. Integrates with Transformers, vLLM, LlamaCpp, and OpenAI API (via logit bias trick).

## Installation

```bash
pip install "lm-format-enforcer>=0.10.0"
# For OpenAI integration:
pip install "lm-format-enforcer[openai]>=0.10.0"
```

No GPU requirement for the enforcer itself — it runs on CPU. Only the LLM inference needs GPU.

## Basic Usage Example

### JSON Schema enforcement with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_format_enforcer import DecoderRuntimeContainer
from pydantic import BaseModel

# Define the schema
class Person(BaseModel):
    name: str
    age: int
    email: str

# Load model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
)

# Create enforcer
from lm_format_enforcer import CharacterLevelParser
from lm_format_enforcer.constraint import JsonSchemaConstraint

schema_str = Person.model_json_schema()
constraint = JsonSchemaConstraint(schema_str)
parser = CharacterLevelParser(constraint)

# Enforce during generation
from transformers import LogitsProcessorList
from lm_format_enforcer import LogitsEnforcer

enforcer = LogitsEnforcer(tokenizer=tokenizer, character_level_parser=parser)

inputs = tokenizer("Generate a JSON for John, 30, john@example.com:", return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")

for _ in range(128):
    outputs = model(input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    next_token_logits = enforcer(input_ids[0], next_token_logits)
    next_token = next_token_logits.argmax()
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
    if next_token == tokenizer.eos_token_id:
        break

print(tokenizer.decode(input_ids[0]))
```

### Regex enforcement

```python
from lm_format_enforcer.constraint import RegexConstraint

regex = r"(\d{3})-(\d{3})-(\d{4})"  # Phone number
parser = CharacterLevelParser(RegexConstraint(regex))
enforcer = LogitsEnforcer(tokenizer=tokenizer, character_level_parser=parser)
```

## Advanced Usage / Configuration

- **`JsonSchemaConstraint`**: Accepts a JSON schema dict or string. Handles `anyOf`, `allOf`, `$ref`, `enum`, nested objects, and arrays.
- **`GrammarConstraint`**: Define a context-free grammar in EBNF format. Can enforce Python code, SQL, or custom DSLs.
- **`RegexConstraint`**: Character-level enforcement of any Python `re` pattern. Supports `[a-zA-Z]`, `\d`, `{min,max}`, groups, etc.
- **`WhitespaceConstraint`**: Enforce specific whitespace rules (e.g., exactly 2-space indentation for YAML).
- **`LogitsEnforcer` modes**: `max_tokens` (limit generated length), `tokenizer` (pass full tokenizer for multi-byte character awareness).
- **Batch enforcement**: `enforcer.reshape_for_batch(batch_size)` to apply the same constraint to multiple simultaneous generations.
- **OpenAI integration**: Use `lm_format_enforcer.openai.OpenAIEnforcer` with `response_model` parameter to get structured responses from chat completion API.
- **vLLM integration**: Pass `guided_decoder="json"` and `guided_decoder_backend="lm-format-enforcer"` to vLLM's `SamplingParams`.

## Integration with the WFB Model Project

```python
# wfb_model/enforce/structure_output.py
from lm_format_enforcer.constraint import JsonSchemaConstraint
from lm_format_enforcer import LogitsEnforcer, CharacterLevelParser

WFB_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
    },
    "required": ["prediction", "confidence"],
}

def enforce_wfb_output(model, tokenizer, prompt: str) -> dict:
    constraint = JsonSchemaConstraint(WFB_OUTPUT_SCHEMA)
    parser = CharacterLevelParser(constraint)
    enforcer = LogitsEnforcer(tokenizer, parser)
    # ... generation loop ...
```

WFB uses LM Format Enforcer to guarantee that every LLM response (for model prediction, explanation, or hyperparameter recommendations) conforms to a specified JSON schema. This eliminates parsing errors downstream and ensures type safety in automated pipelines. Configurations and schemas are stored in `wfb_model/configs/enforcer/`.

## Common Pitfalls / Troubleshooting

- **Slow generation with complex schemas**: Each token requires a character-level validation pass. For deep schemas ($ref chains), batch constraining beforehand. Use `cache_size=10000` on `LogitsEnforcer` to cache token-level prefix masks.
- **Multi-byte characters (Chinese, emoji)**: The enforcer works byte-by-byte internally. Ensure `tokenizer` is passed to `LogitsEnforcer` so it can decode single-byte sequences correctly. Some multi-byte UTF-8 chars may temporarily appear as incomplete.
- **EOS token never generated**: The enforcer may block EOS if the schema hasn't been completed. Add a stop condition: `if enforcer.can_end(): break`.
- **OpenAI API `response_model` fails**: Ensure the model parameter is `gpt-4o` or `gpt-4-turbo` (function-calling models). Set `enforcer.timeout=30` for long outputs.
- **Schema too restrictive**: Test the schema validity first with `JsonSchemaConstraint.is_valid(...)`. A schema that cannot be satisfied (e.g., two exclusive required fields) will loop forever.
- **Regex constraint very slow on long sequences**: Compile the regex with `re.compile(pattern)` and pass it. For very long matches, enforce in chunks.

## Documentation Links

- LM Format Enforcer GitHub: https://github.com/noamgat/lm-format-enforcer
- PyPI: https://pypi.org/project/lm-format-enforcer/
- JSON Schema enforcement guide: https://github.com/noamgat/lm-format-enforcer/blob/main/documentation/JsonSchema.md
- Integration with vLLM: https://github.com/noamgat/lm-format-enforcer/blob/main/documentation/vLLM.md
