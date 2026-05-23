# Guidance

**Version:** 0.1+

**Purpose:** Template-based LLM program with constraint enforcement and variable binding.

## Installation

```powershell
pip install guidance
```

## Usage

```python
import guidance

# Simple template
program = guidance("""What is {{question}}?
Answer: {{#select "answer"}}yes{{/select}}no{{/select}}""")
result = program(question="Is the sky blue?")
print(result["answer"])  # "yes"

# With generation
program = guidance("""Write a poem about {{topic}}:
{{~gen "poem" max_tokens=100 temperature=0.8}}""")
result = program(topic="AI")
print(result["poem"])

# Tool call format
program = guidance("""{{#system~}}
You have access to tools.
{{~/system}}
{{#user~}}
What's the weather?
{{~/user}}
{{#assistant~}}
{{#tool "get_weather"}}
{"city": "Tokyo"}
{{~/tool}}
{{~/assistant}}""")
```

## Key Features

- **Role-based templates** (system/user/assistant/tool)
- **Variable binding** with type constraints
- **Generation directives** (gen, select, each)
- **Grammar enforcement** at token level

## Documentation

- https://github.com/guidance-ai/guidance
