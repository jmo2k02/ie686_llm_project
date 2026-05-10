# Execution Agent Design Notes

These notes capture the current design direction for the execution agent. The goal is to continue from the existing flow:

```text
constraint_iteration_agent -> planner_agent -> execution_agent
```

The planner already returns a validated `task_list`. The execution agent should consume that list, start the right search agents, retain their typed artifacts, and later compose a draft itinerary.

## Recommended Approach

Use a LangGraph execution graph as the orchestrator.

The execution graph should not be just one LLM agent with many agent-tools. It should own the deterministic control flow:

- Read the approved `TaskModel` list.
- Filter to valid tasks.
- Route each task by `task.type`.
- Spawn only the search agent needed for that task.
- Collect typed artifacts from each search agent.
- Merge artifacts back into `StateContractModel.agent_artifacts`.
- Compose a draft itinerary from retained artifacts.
- Later, pass the draft itinerary to a final reviewer and retry selectively if needed.

This matches the architecture in `docs/workflow.md`: spawn agents on demand, return typed artifacts, retain artifacts, compose itinerary, then review/refine.

## Why Not Only Agent Tools

The simple example in `tools.py` is useful for learning the API:

```python
subagent = create_agent(model="google_genai:gemini-3.1-pro-preview", tools=[...])

@tool("research", description="Research a topic and return findings")
def call_generic_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

execution_agent = create_agent(
    model="google_genai:gemini-3.1-pro-preview",
    tools=[call_generic_agent],
)
```

But this should not be the main execution design because the LLM could skip tasks, call tools in the wrong order, hide orchestration decisions in chat, and return anonymous text instead of stable typed artifacts.

Use tools inside search agents for API calls. Use LangGraph for orchestration.

## Spawn Concept

In this project, "spawn" can be simple. It does not need to mean a long-lived process or a background worker.

One spawn event means:

```text
one valid TaskModel -> one matching search graph invocation -> one or more typed artifacts
```

For example, a flight task spawns the flight search graph with a state containing only that task. The result is merged back into the execution state.

## Task Routing

Start with a static routing map from task type to search graph factory:

```python
from travelplanner.agents.attraction_search_agent import make_graph as make_attraction_graph
from travelplanner.agents.flight_search_agent import make_graph as make_flight_graph
from travelplanner.agents.general_web_search_agent import make_graph as make_general_web_search_graph
from travelplanner.agents.hotel_search_agent import make_graph as make_hotel_graph
from travelplanner.agents.restaurant_search_agent import make_graph as make_restaurant_graph


TASK_AGENT_MAP = {
    "flight": ("flight_search_agent", make_flight_graph),
    "hotel": ("hotel_search_agent", make_hotel_graph),
    "restaurant": ("restaurant_search_agent", make_restaurant_graph),
    "attraction": ("attraction_search_agent", make_attraction_graph),
    "general-web-search": ("general_web_search_agent", make_general_web_search_graph),
}
```

The first version can skip unsupported task types such as `opening_times` and `routing-check`, or route them to `general-web-search` until dedicated agents exist.

## Spawn Example

This is the core idea for one task:

```python
def spawn_search_agent_for_task(state, task):
    agent_key, make_agent_graph = TASK_AGENT_MAP[task.type]
    agent_graph = make_agent_graph()

    agent_input = {
        "query": state.query,
        "model_name": state.model_name,
        "temperature": state.temperature,
        "task_list": [task],
        "agent_artifacts": state.agent_artifacts,
        "message_history": None,
    }

    result = agent_graph.invoke(agent_input)

    return {
        "task": task,
        "agent_key": agent_key,
        "artifacts": result.get("agent_artifacts", {}),
        "message_history": result.get("message_history"),
    }
```

The important detail is `"task_list": [task]`. Each spawned search agent sees only the task it should execute.

## Dispatch Node Example

The execution graph can have a node like this:

```python
def dispatch_search_tasks(state):
    valid_tasks = [task for task in state.task_list if task.is_valid]
    agent_artifacts = dict(state.agent_artifacts)

    for task in valid_tasks:
        if task.type not in TASK_AGENT_MAP:
            continue

        spawn_result = spawn_search_agent_for_task(state, task)

        for key, artifacts in spawn_result["artifacts"].items():
            agent_artifacts.setdefault(key, []).extend(artifacts)

    return {
        "agent_artifacts": agent_artifacts,
    }
```

This should be enough for a minimal execution agent: run task-specific search graphs and collect artifacts.

## Possible Execution Graph

A later version can grow into this shape:

```text
validate_tasks
  -> dispatch_search_tasks
  -> collect_artifacts
  -> compose_itinerary
  -> review_itinerary
  -> route_after_review
```

For now, implement only the minimal path:

```text
dispatch_search_tasks -> END
```

Then add itinerary composition and final review once artifact collection works reliably.

## Spawn Event Shape

A spawn event can be represented for logging/debugging like this:

```json
{
  "spawn_id": "spawn-flight-001",
  "task_id": "find_flights",
  "task_type": "flight",
  "agent": "flight_search_agent",
  "input": {
    "query": "Plan a 5 day trip from Frankfurt to Barcelona",
    "task_list": [
      {
        "name": "Find flights",
        "type": "flight",
        "text": "Find round-trip flights from Frankfurt to Barcelona for June 15 to June 20",
        "is_valid": true
      }
    ]
  },
  "output": {
    "artifact_type": "flight_search_result",
    "artifact_name": "flight-search-find-flights"
  }
}
```

This does not need to be persisted in the first implementation. It is useful for debugging and future retry logic.

## Parallelism Later

The first implementation can run tasks sequentially. This is easier to debug.

Later, independent tasks can run in parallel. LangGraph supports dynamic fan-out patterns, but do not start there unless needed. Add dependencies to `TaskModel` first if the planner needs to express ordering.

Potential future dependency shape:

```json
{
  "name": "Find hotels near event venue",
  "type": "hotel",
  "text": "Find hotels near the conference venue",
  "is_valid": true,
  "depends_on": ["find_event_venue"]
}
```

Do not add this until there is a concrete need, because changing `TaskModel` affects prompts and downstream consumers.

## Implementation Steps

1. Create `execution/graph.py`.
2. Define an execution state or reuse `StateContractModel` if no extra fields are needed.
3. Add `TASK_AGENT_MAP`.
4. Add `spawn_search_agent_for_task`.
5. Add `dispatch_search_tasks`.
6. Build a small `StateGraph` with `dispatch_search_tasks -> END`.
7. Wire the workflow as `constraint_agent -> planner_agent -> execution_agent`.
8. Run `uv run python -m compileall src`.

## Design Rule Of Thumb

Use this split:

- LangGraph handles orchestration, routing, retries, and artifact collection.
- Search agents handle specialized retrieval and normalization.
- Tools handle external API calls inside search agents.
- LLMs help with extraction, ranking, conflict resolution, and final composition, but should not silently own the whole execution control flow.
