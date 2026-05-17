from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, StateGraph

from travelplanner.config import get_setting
from travelplanner.agents.constraint_iteration_agent import (
    make_graph as make_constraint_graph,
)
from travelplanner.agents.planner import (
    make_graph as make_planner_graph,
)
from travelplanner.agents.itinerary_validator_agent import (
    make_graph as make_validator_graph,
)
from travelplanner.agents.execution import make_node as make_execution_node
from travelplanner.schema.system_state import StateContractModel, TaskModel


CONSTRAINT_HISTORY_KEY = "constraint_agent"
PLANNER_HISTORY_KEY = "planner_agent"
GENERAL_WEB_SEARCH_HISTORY_KEY = "general_web_search_agent"
ROUTING_CHECK_HISTORY_KEY = "routing_check_agent"


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    """This is the main application graph. This graph orchestrates the whole system."""
    effective_model_name = model_name or str(
        get_setting("models.workflows.task_planning.model_name")
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
    execution_node = make_execution_node(
        model_name=get_setting("agents.execution.model_name"),
        temperature=effective_temperature,
    )
    validator_graph = make_validator_graph().compile()

    MAX_VALIDATION_RETRIES: int = int(
        os.getenv("TRAVELPLANNER_MAX_VALIDATION_RETRIES", "3")
    )

    def route_after_validator(state: StateContractModel) -> str:
        if state.validation_passed:
            return END
        if state.validation_attempts >= MAX_VALIDATION_RETRIES:
            return END
        return "execution_agent"

    graph = StateGraph(StateContractModel)
    graph.add_node("constraint_agent", constraint_graph)
    graph.add_node("planner_agent", planner_graph)
    graph.add_node("execution_agent", execution_node)
    graph.add_node("itinerary_validator", validator_graph)

    graph.set_entry_point("constraint_agent")
    graph.add_edge("constraint_agent", "planner_agent")
    graph.add_edge("planner_agent", "execution_agent")
    graph.add_edge("execution_agent", "itinerary_validator")
    # graph.add_conditional_edges("planner_agent", route_after_planner)
    # graph.add_edge("general_web_search_agent", "routing_check_agent")
    # graph.add_edge("routing_check_agent", "search_orchestrator")
    # graph.add_edge("search_orchestrator", "timetable_builder")
    # graph.add_edge("timetable_builder", "itinerary_validator")
    graph.add_conditional_edges("itinerary_validator", route_after_validator)
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
