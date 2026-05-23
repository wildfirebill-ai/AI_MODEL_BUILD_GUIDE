# CrewAI

**Version:** 0.105.0

## Purpose

CrewAI is a multi-agent orchestration framework for coordinating role-based AI agents. It defines `Agent` instances with specific roles, goals, and backstories, assigns them `Task` objects, and executes them within a `Crew` using a configurable `Process` (sequential or hierarchical). Agents can use custom tools, delegate subtasks, and collaborate to solve complex workflows.

## Installation

```bash
pip install crewai==0.105.0 crewai-tools==0.36.0
```

## Basic Usage

### Define Agents and Execute a Crew

```python
from crewai import Agent, Task, Crew, Process

# Define agents
analyst = Agent(
    role="Data Analyst",
    goal="Analyze wind farm efficiency data",
    backstory="Expert in renewable energy analytics",
    verbose=True,
)

writer = Agent(
    role="Report Writer",
    goal="Write clear reports from analysis",
    backstory="Technical writer with energy domain knowledge",
)

# Define tasks
analysis_task = Task(
    description="Analyze efficiency metrics for the wind farm",
    expected_output="List of key metrics and anomalies",
    agent=analyst,
)

report_task = Task(
    description="Write a summary report based on analysis",
    expected_output="A concise executive summary",
    agent=writer,
)

# Create and run the crew
crew = Crew(
    agents=[analyst, writer],
    tasks=[analysis_task, report_task],
    process=Process.sequential,
)

result = crew.kickoff()
print(result)
```

### Using Custom Tools

```python
from crewai_tools import tool

@tool("WFB Predictor")
def wfb_predictor(farm_id: str) -> str:
    """Predicts wind farm output for a given farm ID."""
    return f"Predicted output for farm {farm_id}: 142.3 MWh"

analyst = Agent(
    role="Data Analyst",
    goal="Predict wind farm output",
    tools=[wfb_predictor],
)
```

## Advanced Usage / Configuration

### Hierarchical Process with Manager

```python
manager = Agent(
    role="Project Manager",
    goal="Coordinate analysis and reporting",
    backstory="Senior project manager with ML expertise",
    allow_delegation=True,
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[analysis_task, report_task],
    process=Process.hierarchical,
    manager_agent=manager,
)
```

### Step Callbacks and Memory

```python
def step_callback(agent, task):
    print(f"{agent.role} completed: {task.description}")

crew = Crew(
    agents=[analyst, writer],
    tasks=[analysis_task, report_task],
    memory=True,
    step_callback=step_callback,
)
```

## Integration with WFB Model Project

CrewAI coordinates multi-agent WFB workflows: an Analyst agent queries model predictions, a Validator agent checks output quality, and a Reporter agent generates dashboards. Agents share context and delegate tasks, enabling automated model monitoring and report generation without human intervention.

## Common Pitfalls / Troubleshooting

- **Task never completes:** Set `max_iter=25` on agents to prevent infinite loops. Verify `expected_output` is specific enough.
- **API key errors:** CrewAI uses LangChain under the hood. Set `OPENAI_API_KEY` (or other provider env vars) before running.
- **Agent hallucinates tools:** Restrict tool access via `tools` parameter and use `@tool` with clear docstrings.
- **Memory issues:** Disable `memory=True` for one-shot tasks. Use `cache=False` for fresh reasoning each time.

## Documentation Links

- [CrewAI Docs](https://docs.crewai.com/)
- [Core Concepts](https://docs.crewai.com/core-concepts/Agents/)
- [Tools Guide](https://docs.crewai.com/core-concepts/Tools/)
