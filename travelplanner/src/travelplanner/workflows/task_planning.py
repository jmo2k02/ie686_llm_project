from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from travelplanner.config import get_setting
from travelplanner.agents.constraint_agent import (
    ConstraintAgentState,
    make_graph as make_constraint_graph,
)
from travelplanner.agents.planner_agent import (
    PlannerAgentState,
    make_graph as make_planner_graph,
)
from travelplanner.agents.reviewer_agent import (
    ReviewerAgentState,
    make_graph as make_reviewer_graph,
)
from travelplanner.agents.general_web_search_agent import (
    GeneralWebSearchAgentState,
    make_graph as make_general_web_search_graph,
)
from travelplanner.schema.system_state import StateContractModel, TaskModel


CONSTRAINT_HISTORY_KEY = "constraint_agent"
PLANNER_HISTORY_KEY = "planner_agent"
REVIEWER_HISTORY_KEY = "reviewer_agent"
GENERAL_WEB_SEARCH_HISTORY_KEY = "general_web_search_agent"


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
):
    effective_model_name = model_name or str(
        get_setting("models.workflows.task_planning.model_name", "gpt-5.4-nano-2026-03-17")
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )
    constraint_graph = make_constraint_graph()
    planner_graph = make_planner_graph()
    reviewer_graph = make_reviewer_graph()
    web_search_graph = make_general_web_search_graph()

    def constraint_node(state: StateContractModel) -> dict[str, Any]:
        agent_state = ConstraintAgentState(
            query=state.query,
            model_name=effective_model_name,
            temperature=effective_temperature,
        )
        result = constraint_graph.invoke(agent_state)

        message_histories = dict(state.message_histories)
        message_histories[CONSTRAINT_HISTORY_KEY] = result["message_history"]
        return {
            "constraint_list": result["constraint_list"],
            "message_histories": message_histories,
        }

    def planner_node(state: StateContractModel) -> dict[str, Any]:
        agent_state = PlannerAgentState(
            query=state.query,
            model_name=effective_model_name,
            temperature=effective_temperature,
            constraint_list=state.constraint_list,
        )
        result = planner_graph.invoke(agent_state)

        message_histories = dict(state.message_histories)
        message_histories[PLANNER_HISTORY_KEY] = result["message_history"]
        return {
            "task_list": result["task_list"],
            "message_histories": message_histories,
        }

    def reviewer_node(state: StateContractModel) -> dict[str, Any]:
        agent_state = ReviewerAgentState(
            query=state.query,
            model_name=effective_model_name,
            temperature=effective_temperature,
            constraint_list=state.constraint_list,
            proposed_task_list=state.task_list,
        )
        result = reviewer_graph.invoke(agent_state)

        message_histories = dict(state.message_histories)
        message_histories[REVIEWER_HISTORY_KEY] = result["message_history"]
        return {
            "task_list": result["approved_task_list"],
            "message_histories": message_histories,
        }

    def general_web_search_node(state: StateContractModel) -> dict[str, Any]:
        agent_state = GeneralWebSearchAgentState(
            query=state.query,
            task_list=state.task_list,
            agent_artifacts=state.agent_artifacts,
        )
        result = web_search_graph.invoke(agent_state)

        message_histories = dict(state.message_histories)
        message_histories[GENERAL_WEB_SEARCH_HISTORY_KEY] = result["message_history"]
        return {
            "agent_artifacts": result["agent_artifacts"],
            "message_histories": message_histories,
        }

    graph = StateGraph(StateContractModel)
    graph.add_node("constraint_agent", constraint_node)
    graph.add_node("planner_agent", planner_node)
    graph.add_node("reviewer_agent", reviewer_node)
    graph.add_node("general_web_search_agent", general_web_search_node)
    graph.set_entry_point("constraint_agent")
    graph.add_edge("constraint_agent", "planner_agent")
    graph.add_edge("planner_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", "general_web_search_agent")
    graph.add_edge("general_web_search_agent", END)
    return graph.compile()


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
