# AutoGen — Multi-Agent Conversation Framework

**Version:** 0.8 / `pyautogen>=0.8.0`

## Purpose

AutoGen (by Microsoft) enables building multi-agent LLM applications where specialized agents converse, delegate tasks, and execute tools collaboratively. Agents can be assistant, user proxy, or group chat roles, each with distinct system prompts and tool sets. Supports GPT-4, Claude, Llama, and local models via vLLM or Ollama.

## Installation

```bash
pip install "pyautogen[autobuild,graph,long-context,retrieve,teach,websurfer]==0.8.0"
# Or minimal:
pip install "pyautogen>=0.8.0"
```

Requires an LLM endpoint (OpenAI API key or local endpoint). Set `OPENAI_API_KEY` or configure a local model endpoint.

## Basic Usage Example

```python
import autogen

config_list = [
    {"model": "gpt-4o", "api_key": "YOUR_KEY"},
]

# Create a coding assistant
assistant = autogen.AssistantAgent(
    name="coder",
    llm_config={"config_list": config_list},
    system_message="You are a Python expert. Write and execute code.",
)

# Create a user proxy that can execute code
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

# Initiate a chat
user_proxy.initiate_chat(
    assistant,
    message="Write a Python script that trains a logistic regression on iris data and prints accuracy.",
)
```

### Group Chat

```python
from autogen import GroupChat, GroupChatManager

planner = autogen.AssistantAgent(name="planner", llm_config={"config_list": config_list})
critic = autogen.AssistantAgent(name="critic", llm_config={"config_list": config_list})

group_chat = GroupChat(
    agents=[user_proxy, assistant, planner, critic],
    messages=[],
    max_round=10,
)
manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list})

user_proxy.initiate_chat(manager, message="Design and implement a WFB training pipeline step.")
```

## Advanced Usage / Configuration

- **Tool execution agents**: Register Python functions with `@user_proxy.register_for_execution()` and `@assistant.register_for_llm()`.
- **Custom reply functions**: Override `generate_reply` for deterministic tool orchestration.
- **WebSurferAgent**: Built-in agent that browses the web, extracts content, and answers questions from live URLs.
- **Teachability**: Agents that learn from user feedback and store knowledge in vector DB (Chroma).
- **RetrieveAssistantAgent**: RAG-powered agent that queries a document collection (`retrieve_config={"task": "qa", "docs_path": "./docs"}`).
- **Nested chats**: Use `register_nested_chats()` to spawn sub-dialogues for subtasks.

## Integration with the WFB Model Project

```python
# wfb_model/agents/train_agent.py
from autogen import AssistantAgent, UserProxyAgent

trainer = AssistantAgent(
    name="trainer",
    system_message="You manage the WFB model training pipeline.",
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]},
)
executor = UserProxyAgent(
    name="executor",
    code_execution_config={"work_dir": "/wfb_model", "use_docker": False},
)

executor.initiate_chat(
    trainer,
    message="Run `python train.py --config configs/default.yaml` and report the loss curve.",
)
```

AutoGen orchestrates multi-step WFB workflows: data loading, preprocessing, hyperparameter search, and evaluation — all through agent collaboration logged to `wfb_model/logs/agent_runs/`.

## Common Pitfalls / Troubleshooting

- **Agent loops forever**: Set `max_consecutive_auto_reply=3` on UserProxyAgent to limit turn count.
- **Code execution errors**: Ensure `code_execution_config["work_dir"]` is writable and `use_docker=False` if Docker is unavailable.
- **LLM rate limits**: Use `config_list` with multiple API keys and enable retry via `llm_config={"config_list": [...], "timeout": 120}`.
- **Group chat deadlock**: Increase `max_round` or add a `speaker_selection_method="random"` to break repetition.
- **Serialization fails with custom objects**: Use `autogen.runtime_logging.start(logger_type="file")` and pass only JSON-serializable data between agents.
- **Local model support**: Point `config_list` at `"api_type": "ollama"` and set `"model": "llama3"`. Disable function calling if the model does not support tools.

## Documentation Links

- AutoGen docs: https://microsoft.github.io/autogen/
- GitHub repo: https://github.com/microsoft/autogen
- GroupChat patterns: https://microsoft.github.io/autogen/docs/tutorial/group-chat
- Agent function registration: https://microsoft.github.io/autogen/docs/tutorial/tool-use
