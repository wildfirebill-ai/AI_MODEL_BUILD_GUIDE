# LangChain / AutoGen / CrewAI

**Purpose:** Agent frameworks for building LLM-powered applications with tool use, multi-agent coordination, and RAG.

---

## LangChain

### Installation

```powershell
pip install langchain langchain-core langchain-community
```

### Usage

```python
from langchain.llms import VLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

llm = VLLM(model="./model", trust_remote_code=True)

@tool
def search(query: str) -> str:
    """Search the knowledge base"""
    return "Results for: " + query

agent = create_react_agent(llm, [search], ...)
executor = AgentExecutor(agent=agent, tools=[search])
result = executor.invoke({"input": "What is the weather?"})
```

---

## AutoGen (Microsoft)

### Installation

```powershell
pip install pyautogen
```

### Usage

```python
import autogen

assistant = autogen.AssistantAgent(name="assistant", llm_config={"model": "./model"})
user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="NEVER")
user_proxy.initiate_chat(assistant, message="Write a Python script for Fibonacci")
```

---

## CrewAI

### Installation

```powershell
pip install crewai
```

---

## Comparison

| Framework | Agents | Tool Use | Memory | Multi-agent |
|-----------|--------|----------|--------|-------------|
| LangChain | Yes | Yes | Yes | Yes |
| AutoGen | Yes | Yes | Yes | Yes (built-in) |
| CrewAI | Yes (roles) | Yes | Yes | Yes (role-based) |
