# LangGraph

**Version:** 0.3.0

## Purpose

LangGraph is a framework for building stateful, multi-actor LLM applications using directed graph structures. It extends LangChain with cyclic graphs, checkpointing, human-in-the-loop workflows, and persistent state management. LangGraph enables complex agentic behaviors: multi-step reasoning, tool-use orchestration, conditional branching, and parallel node execution.

## Installation

```bash
pip install langgraph==0.3.0 langchain-openai==0.3.0
```

## Basic Usage

### Simple Agent Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import random

class AgentState(TypedDict):
    input: str
    reasoning: str
    output: str

def reason(state: AgentState) -> dict:
    return {"reasoning": f"Thinking about: {state['input']}"}

def decide_action(state: AgentState) -> Literal["answer", "tool"]:
    return "answer" if "weather" not in state["input"] else "tool"

def answer(state: AgentState) -> dict:
    return {"output": f"Based on reasoning: {state['reasoning']}"}

def use_tool(state: AgentState) -> dict:
    return {"output": f"Tool result for: {state['input']}"}

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("reason", reason)
builder.add_node("answer", answer)
builder.add_node("tool", use_tool)

builder.set_entry_point("reason")
builder.add_conditional_edges("reason", decide_action)
builder.add_edge("answer", END)
builder.add_edge("tool", END)

graph = builder.compile()

# Run
result = graph.invoke({"input": "What is the weather in Berlin?"})
print(result["output"])
```

### With Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke({"input": "Hello"}, config=thread)
result = graph.invoke({"input": "Continue"}, config=thread)
```

## Advanced Usage / Configuration

### Human-in-the-Loop

```python
from langgraph.pregel import interrupt

def human_review(state):
    response = interrupt({"question": "Approve step?", "data": state})
    if response["approved"]:
        return {"output": "Proceeding..."}
    return {"output": "Cancelled"}
```

### Parallel Execution

```python
builder.add_node("research", research_fn)
builder.add_node("summarize", summarize_fn)
builder.add_edge("reason", "research")
builder.add_edge("reason", "summarize")
```

## Integration with WFB Model Project

LangGraph orchestrates multi-step WFB workflows: given a user query, the graph reasons about which WFB model components to invoke (turbine prediction, farm optimization, report generation), routes to the appropriate tool, and formats the response with checkpointing for conversation persistence.

## Common Pitfalls / Troubleshooting

- **State not updating:** Return a dict from node functions with only the keys you want to update. State is immutable by default.
- **Infinite loops:** Ensure conditional edges have exhaustive `Literal` return types and terminal edges to `END`.
- **Checkpoint not persisting:** Use `MemorySaver` for in-memory or `SqliteSaver` for persistent checkpoints across restarts.
- **Graph visualization:** Call `graph.get_graph().print_ascii()` or `graph.get_graph().draw_mermaid_png()` to debug graph structure.

## Documentation Links

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [API Reference](https://langchain-ai.github.io/langgraph/reference/)
