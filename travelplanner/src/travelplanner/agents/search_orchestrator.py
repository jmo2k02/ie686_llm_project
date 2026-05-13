"""Search Orchestrator — dynamically dispatches search agents per task type.

One monolithic function ``run_searches`` that runs all searches in priority order
(flight → hotel → restaurant → attraction), applies retry logic + sanity checks,
and stores results in ``agent_artifacts``.
"""

from __future__ import annotations

from typing import Any

from travelplanner.config import get_setting
from travelplanner.schema.system_state import AgentArtifactModel, StateContractModel, TaskModel

_DEFAULT_MODEL = get_setting("models.workflows.task_planning.model_name")
_MAX_RETRIES_PER_AGENT = 3


def _model_name(state: StateContractModel) -> str:
    return state.message_histories.get("execution_agent", {}).get("model", _DEFAULT_MODEL)


def _is_result_sane(artifacts: list[AgentArtifactModel]) -> tuple[bool, str]:
    """Sanity check: at least one artifact exists and none indicate failure."""
    if not artifacts:
        return False, "No artifact produced"

    for a in artifacts:
        content = a.content if isinstance(a.content, dict) else {}
        status = content.get("status", "ok")
        if status == "failed":
            errors = content.get("errors", [])
            err = errors[0] if errors else "Unknown error"
            return False, err

        if a.type == "attraction-search-result":
            finalists = content.get("finalists", [])
            if not finalists:
                return False, "No attraction finalists selected"

        if a.type == "flight-search-result":
            best = content.get("best_flights", [])
            if not best:
                return False, "No flight options found"

    return True, ""


def _run_with_retry(
    task: TaskModel,
    state: StateContractModel,
    handler: Any,
) -> list[AgentArtifactModel]:
    """Run a search handler with up to ``_MAX_RETRIES_PER_AGENT`` retries + sanity checks."""
    attempt = 0
    artifacts: list[AgentArtifactModel] = []
    sane = False

    while attempt < _MAX_RETRIES_PER_AGENT and not sane:
        attempt += 1
        artifacts = handler(task, state)
        sane, _ = _is_result_sane(artifacts)
        if sane:
            break

    return artifacts


def run_searches(state: StateContractModel) -> dict[str, Any]:
    """Run all search agents dynamically per task type and store results.

    Priority order: flight → hotel → restaurant → attraction.
    """
    from travelplanner.agents.flight_search_agent import (
        FlightSearchAgentState,
        make_graph as make_flight_graph,
    )
    from travelplanner.agents.hotel_search_agent import (
        IntelligentHotelSearchState,
        make_intelligent_hotel_graph,
    )
    from travelplanner.agents.restaurant_search_agent import (
        RestaurantSearchAgentState,
        make_graph as make_restaurant_graph,
    )
    from travelplanner.agents.attraction_search_agent import (
        AttractionSearchAgentState,
        make_graph as make_attraction_graph,
    )

    task_by_type: dict[str, list[TaskModel]] = {}
    for t in state.task_list:
        if t.is_valid and t.type in ("flight", "hotel", "restaurant", "attraction"):
            task_by_type.setdefault(t.type, []).append(t)

    merged_artifacts = dict(state.agent_artifacts)

    for task_type in ("flight", "hotel", "restaurant", "attraction"):
        tasks = task_by_type.get(task_type, [])
        if not tasks:
            continue

        for task in tasks:
            if task_type == "flight":

                def handler(task: TaskModel, state: StateContractModel) -> list[AgentArtifactModel]:
                    try:
                        graph = make_flight_graph()
                        agent_state = FlightSearchAgentState(
                            query=state.query,
                            model_name=_model_name(state),
                            temperature=0.0,
                            task_list=[task],
                            agent_artifacts=state.agent_artifacts,
                        )
                        result = graph.invoke(agent_state)
                        return list(result.get("agent_artifacts", {}).get("flight_search_agent", []))
                    except Exception as exc:
                        return [
                            AgentArtifactModel(
                                name=task.name,
                                type="flight-search-result",
                                content={"status": "failed", "errors": [str(exc)]},
                                description=f"Flight search failed: {exc}",
                            )
                        ]

                agent_key = "flight_search_agent"

            elif task_type == "hotel":

                def handler(task: TaskModel, state: StateContractModel) -> list[AgentArtifactModel]:
                    try:
                        graph = make_intelligent_hotel_graph()
                        agent_state = IntelligentHotelSearchState(
                            query=task.text,
                            system_state=state,
                            agent_key="hotel_search",
                            model_name=_model_name(state),
                        )
                        result = graph.invoke(agent_state)
                        return list(
                            result.get("system_state", {}).get("agent_artifacts", {}).get("hotel_search", [])
                        )
                    except Exception as exc:
                        return [
                            AgentArtifactModel(
                                name=task.name,
                                type="hotel-search-result",
                                content={"status": "failed", "errors": [str(exc)]},
                                description=f"Hotel search failed: {exc}",
                            )
                        ]

                agent_key = "hotel_search_agent"

            elif task_type == "restaurant":

                def handler(task: TaskModel, state: StateContractModel) -> list[AgentArtifactModel]:
                    try:
                        graph = make_restaurant_graph()
                        agent_state = RestaurantSearchAgentState(
                            query=task.text,
                            model_name=_model_name(state),
                            system_state=state,
                            agent_key="restaurant_search",
                            temperature=0.0,
                            task_list=[task],
                        )
                        result = graph.invoke(agent_state)
                        return list(
                            result.get("system_state", {})
                            .get("agent_artifacts", {})
                            .get("restaurant_search", [])
                        )
                    except Exception as exc:
                        return [
                            AgentArtifactModel(
                                name=task.name,
                                type="restaurant-search-result",
                                content={"status": "failed", "errors": [str(exc)]},
                                description=f"Restaurant search failed: {exc}",
                            )
                        ]

                agent_key = "restaurant_search_agent"

            elif task_type == "attraction":

                def handler(task: TaskModel, state: StateContractModel) -> list[AgentArtifactModel]:
                    try:
                        graph = make_attraction_graph()
                        agent_state = AttractionSearchAgentState(
                            query=state.query,
                            model_name=_model_name(state),
                            temperature=0.0,
                            task_list=[task],
                            agent_artifacts=state.agent_artifacts,
                            attraction_params=None,
                            experience_pool=None,
                            raw_search_results=None,
                            search_result=None,
                            selected_candidates=[],
                            archetype=None,
                            generated_experiences=[],
                            selected_experiences=[],
                            activity_options=[],
                            final_candidates=[],
                            status="pending",
                            errors=[],
                            retry_count=0,
                        )
                        result = graph.invoke(agent_state)
                        return list(result.get("agent_artifacts", {}).get("attraction_search_agent", []))
                    except Exception as exc:
                        return [
                            AgentArtifactModel(
                                name=task.name,
                                type="attraction-search-result",
                                content={"status": "failed", "errors": [str(exc)]},
                                description=f"Attraction search failed: {exc}",
                            )
                        ]

                agent_key = "attraction_search_agent"

            task_artifacts = _run_with_retry(task, state, handler)
            if task_artifacts:
                merged_artifacts.setdefault(agent_key, [])
                merged_artifacts[agent_key].extend(task_artifacts)

    return {"agent_artifacts": merged_artifacts}
