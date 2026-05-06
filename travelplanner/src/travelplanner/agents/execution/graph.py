from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from travelplanner.agents.execution.prompts import SYSTEM_PROMPT
from travelplanner.travelplan import TravelPlan, make_travelplan_tools
from travelplanner.utils.llm import make_chat_model


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
        tools=make_travelplan_tools(plan),
        system_prompt=SYSTEM_PROMPT,
    )
import logging
from pydantic import BaseModel, Field

from travelplanner.agents.execution.subagents import (
    TASK_AGENT_MAP,
    spawn_search_agent_for_task,
)
from travelplanner.schema.system_state import TaskModel
from travelplanner.config import get_setting

logger = logging.getLogger("execution_agent")

MODEL_NAME = get_setting("agents.execution.model_name")


class ExecutionAgentState(BaseModel):
    query: str
    task_list: list[TaskModel] = Field(default_factory=list)


def dispatch_search_tasks(state):
    """"""
    for task in state.task_list:
        if task.type not in TASK_AGENT_MAP:
            msg = f"Task type not available in TASK_AGENT_MAP: '{task.type}'. Skipping task."
            logger.warning(msg)
            continue

        spawn_result = spawn_search_agent_for_task(state, task)

        print(spawn_result)
