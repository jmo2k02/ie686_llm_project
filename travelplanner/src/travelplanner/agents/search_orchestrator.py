"""Search Orchestrator — per-type search nodes for the main workflow graph.

Each search type (flight, hotel, restaurant, attraction) gets its own
LangGraph node so the dashboard can show real-time progress.

Nodes:
- ``search_flight``     → runs flight search agent for valid flight tasks
- ``search_hotel``      → runs hotel search agent for valid hotel tasks
- ``search_restaurant`` → runs restaurant search agent for valid restaurant tasks
- ``search_attraction`` → runs attraction search agent for valid attraction tasks

Each node filters the task_list, invokes the matching agent, and stores
results in ``state.agent_artifacts``.  If no matching tasks exist, the node
passes through state unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    StateContractModel,
    TaskModel,
)


_DEFAULT_MODEL = "openai:gpt-4o-mini"
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
    agent_key: str,
) -> list[AgentArtifactModel]:
    """Run a search handler with up to MAX_RETRIES retries + sanity checks."""
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


# ── Flight Search ─────────────────────────────────────────────────────────


def search_flight_node(state: StateContractModel) -> dict[str, Any]:
    """Run flight search for all valid flight tasks."""
    from travelplanner.agents.flight_search_agent import (
        FlightSearchAgentState,
        make_graph as make_flight_graph,
    )

    tasks = [t for t in state.task_list if t.type == "flight" and t.is_valid]
    if not tasks:
        return {}

    artifacts: list[AgentArtifactModel] = []
    messages: list[str] = []

    for task in tasks:
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
                return [AgentArtifactModel(
                    name=task.name, type="flight-search-result",
                    content={"status": "failed", "errors": [str(exc)]},
                    description=f"Flight search failed: {exc}",
                )]

        task_artifacts = _run_with_retry(task, state, handler, "flight_search_agent")
        artifacts.extend(task_artifacts)
        messages.append(
            f"Flight '{task.name}': {len(task_artifacts)} artifact(s), "
            f"sane={_is_result_sane(task_artifacts)[0]}"
        )

    merged = dict(state.agent_artifacts)
    if artifacts:
        merged.setdefault("flight_search_agent", [])
        merged["flight_search_agent"].extend(artifacts)

    return {
        "agent_artifacts": merged,
        "message_histories": {
            **state.message_histories,
            "search_flight": MessageHistoryModel(
                user_agent="search_flight",
                model=_model_name(state),
                agent_ref="travelplanner.agents.search_orchestrator",
                messages=[
                    {"role": "user", "content": json.dumps([t.model_dump() for t in tasks])},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
            ),
        },
    }


# ── Hotel Search ────────────────────────────────────────────────────────


def search_hotel_node(state: StateContractModel) -> dict[str, Any]:
    """Run hotel search for all valid hotel tasks."""
    from travelplanner.agents.hotel_search_agent import (
        IntelligentHotelSearchState,
        make_intelligent_hotel_graph,
    )

    tasks = [t for t in state.task_list if t.type == "hotel" and t.is_valid]
    if not tasks:
        return {}

    artifacts: list[AgentArtifactModel] = []
    messages: list[str] = []

    for task in tasks:
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
                return list(result.get("system_state", {}).get("agent_artifacts", {}).get("hotel_search", []))
            except Exception as exc:
                return [AgentArtifactModel(
                    name=task.name, type="hotel-search-result",
                    content={"status": "failed", "errors": [str(exc)]},
                    description=f"Hotel search failed: {exc}",
                )]

        task_artifacts = _run_with_retry(task, state, handler, "hotel_search_agent")
        artifacts.extend(task_artifacts)
        messages.append(
            f"Hotel '{task.name}': {len(task_artifacts)} artifact(s), "
            f"sane={_is_result_sane(task_artifacts)[0]}"
        )

    merged = dict(state.agent_artifacts)
    if artifacts:
        merged.setdefault("hotel_search_agent", [])
        merged["hotel_search_agent"].extend(artifacts)

    return {
        "agent_artifacts": merged,
        "message_histories": {
            **state.message_histories,
            "search_hotel": MessageHistoryModel(
                user_agent="search_hotel",
                model=_model_name(state),
                agent_ref="travelplanner.agents.search_orchestrator",
                messages=[
                    {"role": "user", "content": json.dumps([t.model_dump() for t in tasks])},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
            ),
        },
    }


# ── Restaurant Search ───────────────────────────────────────────────────


def search_restaurant_node(state: StateContractModel) -> dict[str, Any]:
    """Run restaurant search for all valid restaurant tasks."""
    from travelplanner.agents.restaurant_search_agent import (
        RestaurantSearchAgentState,
        make_graph as make_restaurant_graph,
    )

    tasks = [t for t in state.task_list if t.type == "restaurant" and t.is_valid]
    if not tasks:
        return {}

    artifacts: list[AgentArtifactModel] = []
    messages: list[str] = []

    for task in tasks:
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
                return list(result.get("system_state", {}).get("agent_artifacts", {}).get("restaurant_search", []))
            except Exception as exc:
                return [AgentArtifactModel(
                    name=task.name, type="restaurant-search-result",
                    content={"status": "failed", "errors": [str(exc)]},
                    description=f"Restaurant search failed: {exc}",
                )]

        task_artifacts = _run_with_retry(task, state, handler, "restaurant_search_agent")
        artifacts.extend(task_artifacts)
        messages.append(
            f"Restaurant '{task.name}': {len(task_artifacts)} artifact(s), "
            f"sane={_is_result_sane(task_artifacts)[0]}"
        )

    merged = dict(state.agent_artifacts)
    if artifacts:
        merged.setdefault("restaurant_search_agent", [])
        merged["restaurant_search_agent"].extend(artifacts)

    return {
        "agent_artifacts": merged,
        "message_histories": {
            **state.message_histories,
            "search_restaurant": MessageHistoryModel(
                user_agent="search_restaurant",
                model=_model_name(state),
                agent_ref="travelplanner.agents.search_orchestrator",
                messages=[
                    {"role": "user", "content": json.dumps([t.model_dump() for t in tasks])},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
            ),
        },
    }


# ── Attraction Search ───────────────────────────────────────────────────


def search_attraction_node(state: StateContractModel) -> dict[str, Any]:
    """Run attraction search for all valid attraction tasks."""
    from travelplanner.agents.attraction_search_agent import (
        AttractionSearchAgentState,
        make_graph as make_attraction_graph,
    )

    tasks = [t for t in state.task_list if t.type == "attraction" and t.is_valid]
    if not tasks:
        return {}

    artifacts: list[AgentArtifactModel] = []
    messages: list[str] = []

    for task in tasks:
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
                return [AgentArtifactModel(
                    name=task.name, type="attraction-search-result",
                    content={"status": "failed", "errors": [str(exc)]},
                    description=f"Attraction search failed: {exc}",
                )]

        task_artifacts = _run_with_retry(task, state, handler, "attraction_search_agent")
        artifacts.extend(task_artifacts)
        messages.append(
            f"Attraction '{task.name}': {len(task_artifacts)} artifact(s), "
            f"sane={_is_result_sane(task_artifacts)[0]}"
        )

    merged = dict(state.agent_artifacts)
    if artifacts:
        merged.setdefault("attraction_search_agent", [])
        merged["attraction_search_agent"].extend(artifacts)

    return {
        "agent_artifacts": merged,
        "message_histories": {
            **state.message_histories,
            "search_attraction": MessageHistoryModel(
                user_agent="search_attraction",
                model=_model_name(state),
                agent_ref="travelplanner.agents.search_orchestrator",
                messages=[
                    {"role": "user", "content": json.dumps([t.model_dump() for t in tasks])},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
            ),
        },
    }
