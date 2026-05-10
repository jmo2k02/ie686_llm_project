from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from travelplanner.agents.execution.prompts import SYSTEM_PROMPT
from travelplanner.agents.tools import make_subagent_tools
from travelplanner.config import get_setting
from travelplanner.schema.system_state import StateContractModel, TodoItem
from travelplanner.travelplan import TravelPlan, make_travelplan_tools
from travelplanner.utils.llm import make_chat_model
from travelplanner.utils.runtime_monitor import update_todos, update_travelplan


logger = logging.getLogger("execution_agent")

MODEL_NAME = get_setting("agents.execution.model_name")


def make_graph(
    plan: TravelPlan,
    *,
    model: str | BaseChatModel = "openai:gpt-4o-mini",
    temperature: float = 0.0,
) -> CompiledStateGraph:
    """Build the TravelPlanner execution agent.

    The given ``plan`` is closure-bound into the tool layer; every tool call
    mutates that single instance. After ``agent.invoke({"messages": [...]})``
    returns, inspect ``plan`` to read the final state.

    Args:
        plan: The TravelPlan instance the agent will edit. Pass an empty
            ``TravelPlan()`` for a fresh session, or a pre-seeded one to
            continue editing.
        model: Either a provider-aware model name (e.g. ``"openai:gpt-4o-mini"``
            — same convention as the rest of the agents in this repo) or a
            pre-built ``BaseChatModel`` (useful for tests with a fake model).
        temperature: Sampling temperature; ignored when ``model`` is already
            a ``BaseChatModel``.

    Returns:
        A compiled LangGraph state graph. Invoke via
        ``graph.invoke({"messages": [HumanMessage(...)]})``.
    """
    if isinstance(model, BaseChatModel):
        chat_model: BaseChatModel = model
    else:
        chat_model = make_chat_model(model_name=model, temperature=temperature)

    return create_deep_agent(
        model=chat_model,
        tools=[*make_subagent_tools(), *make_travelplan_tools(plan)],
        system_prompt=SYSTEM_PROMPT,
    )


def _to_todo_items(raw_todos: Any) -> list[TodoItem]:
    """Map deepagents' Todo TypedDicts ({content, status}) onto TodoItem."""
    if not isinstance(raw_todos, list):
        return []
    items: list[TodoItem] = []
    for entry in raw_todos:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content") or ""
        status = entry.get("status", "pending")
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        items.append(TodoItem(title=content, status=status, description=""))
    return items


def _compose_user_prompt(state: StateContractModel) -> str:
    sections: list[str] = [
        "# Original user request",
        state.query.strip(),
    ]

    if state.normalized_constraints is not None:
        nc_dump = state.normalized_constraints.model_dump(exclude_none=True)
        if nc_dump:
            sections.append("\n# Normalized constraints (MUST honor)")
            sections.append(json.dumps(nc_dump, indent=2, ensure_ascii=False))

    if state.task_list:
        task_lines = [
            f"- ({task.type}) {task.name}: {task.text}"
            for task in state.task_list
        ]
        sections.append(
            "\n# Planner suggestions (helpful guidelines, NOT a strict checklist)"
        )
        sections.append("\n".join(task_lines))
        sections.append(
            "Treat the planner's tasks as inspiration. You may merge, reorder, "
            "split, drop, or add tasks based on the constraints and your own "
            "judgement — what matters is honouring the constraints and producing "
            "a coherent itinerary."
        )

    sections.append("\nNow build the travel plan with the available tools.")
    return "\n".join(sections)


def make_node(
    *,
    model_name: str,
    temperature: float = 0.0,
) -> Callable[[StateContractModel], Awaitable[dict[str, Any]]]:
    """Build a langgraph-compatible async node that runs the deepagent.

    The returned coroutine:
      - Closure-binds ``state.travelplan`` into the deepagent's tools.
      - Streams the deepagent (``stream_mode="values"``) so it can mirror
        deepagents' internal todos out to the dashboard live.
      - Pushes both the live travelplan reference and the latest todos into
        the active ``RunMonitor`` (no-op if none is set).
      - Returns ``{"travelplan": plan, "todos": latest_todos}`` once the
        deepagent terminates.
    """

    async def execution_node(state: StateContractModel) -> dict[str, Any]:
        plan: TravelPlan = state.travelplan
        agent = make_graph(plan, model=model_name, temperature=temperature)
        prompt = _compose_user_prompt(state)

        latest_todos: list[TodoItem] = []
        async for event in agent.astream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="values",
        ):
            if isinstance(event, dict):
                latest_todos = _to_todo_items(event.get("todos"))
            update_travelplan(plan)
            update_todos(latest_todos)
        with open("tp.json", "w") as f_json, open("tp.md", "w") as f_md, open("tp.ics", "w", encoding="utf-8") as f_ics:
            f_json.write(plan.model_dump_json(indent=2))
            f_md.write(plan.to_markdown())
            f_ics.write(plan.to_ical())
            
        return {"travelplan": plan, "todos": latest_todos}

    return execution_node
