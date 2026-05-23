# Semantic Kernel — SDK for LLM Integration

**Version:** 1.22 / `semantic-kernel>=1.22.0`

## Purpose

Semantic Kernel (Microsoft) is an SDK that bridges LLMs with traditional programming. It provides AI orchestration with plugins (native functions), semantic functions (prompt templates with variable injection), memory (vector store for RAG), planners (auto function chaining), and connectors for OpenAI/Azure/Claude/Llama. Kernel is the central DI container that coordinates model services, plugins, memory, and logging.

## Installation

```bash
pip install "semantic-kernel[openai,memory]==1.22.0"
# For Azure OpenAI:
pip install "semantic-kernel[azure]==1.22.0"
```

Set environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`.

## Basic Usage Example

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# Create kernel
kernel = Kernel()
service = OpenAIChatCompletion(service_id="gpt4", ai_model_id="gpt-4o")
kernel.add_service(service)

# Define a native plugin
class WeatherPlugin:
    @kernel_function(description="Get current weather for a location.")
    def get_weather(self, location: str) -> str:
        return f"The weather in {location} is 22°C and sunny."

kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

# Define a semantic function (prompt template)
from semantic_kernel.functions import KernelFunctionFromPrompt

greeting = KernelFunctionFromPrompt(
    function_name="greet",
    plugin_name="hello",
    prompt="Hello {{$name}}, today's weather: {{weather.get_weather $name}}",
)

async def main():
    result = await kernel.invoke(greeting, name="Paris")
    print(result)

asyncio.run(main())
```

### Chat history

```python
from semantic_kernel.contents import ChatHistory

chat = ChatHistory()
chat.add_user_message("Explain what a transformer is.")
reply = await kernel.invoke_prompt("{{$chat}}", chat=chat)
print(reply)
```

## Advanced Usage / Configuration

- **Planner**: `semantic_kernel.planners.SequentialPlanner` auto-generates a step-by-step plan by connecting registered plugins. Use `kernel.plan()` to execute.
- **Memory**: Add vector stores via `semantic_kernel.memory.MemoryBuilder()` with backends (Chroma, Qdrant, Azure AI Search). Store text with embeddings and retrieve with `kernel.memory.search_async()`.
- **Filters**: Register callbacks on `kernel.add_filter("function_invocation")` for logging, safety checks, or latency tracking.
- **Auto function calling**: When a model supports tool calls, SK automatically invokes the correct plugin function and returns results to the model.
- **Multiple model backends**: Register multiple services; select by `service_id` when creating functions. Supports OpenAI, Azure, HuggingFace, Vertex AI.
- **YAML prompt templates**: Store prompts in `.yaml` files with `template_format`, `execution_settings`, and `input_variables`.

## Integration with the WFB Model Project

```python
# wfb_model/semantic_kernel/wfb_plugin.py
from semantic_kernel.functions import kernel_function

class WFBModelPlugin:
    @kernel_function(description="Train the WFB model with given hyperparameters.")
    def train(self, learning_rate: float, epochs: int) -> str:
        # Calls wfb_model/train.py internally
        return f"Training started with lr={learning_rate}, epochs={epochs}"

# Kernel is initialized once in wfb_model/kernel_init.py
# and injected into dependent services.
```

WFB uses Semantic Kernel as the LLM orchestration layer: RAG queries on model documentation, natural-language-driven training triggers, and automated experiment reporting via plugin chains.

## Common Pitfalls / Troubleshooting

- **Planner fails to find functions**: Ensure all plugins are registered via `kernel.add_plugin()` *before* calling `kernel.plan()`. Plugin names must match the prompt template's `{{plugin_name.function_name}}` syntax.
- **Memory search returns empty**: Verify the memory store has data after `kernel.memory.save_information_async()`. The default memory store (volatile) resets on restart.
- **Prompt injection warnings**: Use `semantic_kernel.functions.KernelArguments` with strict `InputVariable` types and `template_format="handlebars"` to auto-escape user inputs.
- **Token limit exceeded**: Set `execution_settings={"max_tokens": 4000}` on `KernelFunctionFromPrompt` or use `OpenAIPromptExecutionSettings(max_tokens=... )`.
- **Async deadlock in Jupyter**: Ensure you run `asyncio.run(main())` only once per cell. Use `await` in notebooks after `import nest_asyncio; nest_asyncio.apply()`.
- **Model not found**: The `ai_model_id` must match a deployed model name. For Azure, set `deployment_name` in `AzureChatCompletion`.

## Documentation Links

- Semantic Kernel docs: https://learn.microsoft.com/en-us/semantic-kernel/
- Python SDK reference: https://github.com/microsoft/semantic-kernel/tree/main/python
- Planners: https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning
- Memory: https://learn.microsoft.com/en-us/semantic-kernel/memories/
