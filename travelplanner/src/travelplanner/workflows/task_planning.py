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
from travelplanner.integrations.routing_check_agent import (
    RoutingCheckAgentState,
    make_graph as make_routing_check_graph,
)
from travelplanner.integrations.routing_contracts import (
    PlaceGraphFileTaskPayload,
    SingleOdTaskPayload,
    parse_routing_check_task_text,
)
from travelplanner.schema.system_state import StateContractModel, TaskModel


CONSTRAINT_HISTORY_KEY = "constraint_agent"
PLANNER_HISTORY_KEY = "planner_agent"
GENERAL_WEB_SEARCH_HISTORY_KEY = "general_web_search_agent"
ROUTING_CHECK_HISTORY_KEY = "routing_check_agent"


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
    web_search_graph = make_general_web_search_graph().compile()
    routing_check_graph = make_routing_check_graph()

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

    def routing_check_node(state: StateContractModel) -> dict[str, Any]:
        routing_tasks = [
            t for t in state.task_list if t.type == "routing-check" and t.is_valid
        ]
        if not routing_tasks:
            return {"message_histories": dict(state.message_histories)}

        task = routing_tasks[0]

        try:
            parsed = parse_routing_check_task_text(task.text)
        except (ValueError, Exception):
            return {"message_histories": dict(state.message_histories)}

        if isinstance(parsed, SingleOdTaskPayload):
            agent_state = RoutingCheckAgentState(
                task_ref=task.name,
                origin_address=parsed.origin_address,
                destination_address=parsed.destination_address,
                travel_mode=parsed.travel_mode,
                departure_time_rfc3339=parsed.departure_time_rfc3339,
                detail_level=parsed.detail_level,
                include_transit_alternatives=parsed.include_transit_alternatives,
            )
        elif isinstance(parsed, PlaceGraphFileTaskPayload):
            agent_state = RoutingCheckAgentState(
                task_ref=task.name,
                places_json_path=parsed.places_json_path,
                cluster_context=parsed.cluster_context,
            )
        else:
            return {"message_histories": dict(state.message_histories)}

        result = routing_check_graph.invoke(agent_state)

        message_histories = dict(state.message_histories)
        message_histories[ROUTING_CHECK_HISTORY_KEY] = result["message_history"]
        existing = dict(state.agent_artifacts)
        key = "routing_check_agent"
        artifact = result.get("artifact")
        if artifact is not None:
            existing[key] = existing.get(key, []) + [artifact]
        return {
            "agent_artifacts": existing,
            "message_histories": message_histories,
        }

    def route_after_planner(state: StateContractModel) -> str:
        """Match main latency when there's nothing to execute after planning.

        On ``main``, the workflow stops at ``planner_agent`` → END. Keep that path
        when there are no valid general-web-search or routing-check tasks. When
        both apply, general web search still runs before routing-check.
        """
        tasks = state.task_list
        if any(t.type == "general-web-search" and t.is_valid for t in tasks):
            return "general_web_search_agent"
        if any(t.type == "routing-check" and t.is_valid for t in tasks):
            return "routing_check_agent"
        return END

    graph = StateGraph(StateContractModel)
    graph.add_node("constraint_agent", constraint_graph)
    graph.add_node("planner_agent", planner_graph)
    graph.add_node("general_web_search_agent", general_web_search_node)
    graph.add_node("routing_check_agent", routing_check_node)

    graph.set_entry_point("constraint_agent")
    graph.add_edge("constraint_agent", "planner_agent")
    graph.add_conditional_edges("planner_agent", route_after_planner)
    graph.add_edge("general_web_search_agent", "routing_check_agent")
    graph.add_edge("routing_check_agent", END)
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
