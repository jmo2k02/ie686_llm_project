from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from travelplanner.config import get_setting
from travelplanner.agents.constraint_iteration_agent import (
    ConstraintIterationState,
    get_constraint_list,
    get_message_history,
    make_graph as make_constraint_graph,
)
from travelplanner.agents.planner import (
    make_graph as make_planner_graph,
)
from travelplanner.agents.general_web_search_agent import (
    GeneralWebSearchAgentState,
    make_graph as make_general_web_search_graph,
)
from travelplanner.schema.system_state import StateContractModel, TaskModel


CONSTRAINT_HISTORY_KEY = "constraint_agent"
PLANNER_HISTORY_KEY = "planner_agent"
GENERAL_WEB_SEARCH_HISTORY_KEY = "general_web_search_agent"


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    effective_model_name = model_name or str(
        get_setting(
            "models.workflows.task_planning.model_name", "gpt-5.4-nano-2026-03-17"
        )
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )
    constraint_graph = make_constraint_graph().compile()
    planner_graph = make_planner_graph(
        model_name=effective_model_name,
        temperature=effective_temperature,
    ).compile()
    

    graph = StateGraph(StateContractModel)
    graph.add_node("constraint_agent", constraint_graph)
    graph.add_node("planner_agent", planner_graph)


    graph.set_entry_point("constraint_agent")
    graph.add_edge("constraint_agent", "planner_agent")
    graph.add_edge("planner_agent", END)
    
    return graph


def run(query: str, model_name: str, temperature: float = 0.0) -> StateContractModel:
    graph = make_graph(model_name=model_name, temperature=temperature)
    initial_state = StateContractModel(query=query)
    result = graph.invoke(initial_state)
    return StateContractModel.model_validate(result)


def get_reviewed_task_list(
    query: str,
    model_name: str,
    temperature: float = 0.0,
) -> list[TaskModel]:
    return run(query=query, model_name=model_name, temperature=temperature).task_list
