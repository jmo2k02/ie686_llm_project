"""Routing agent with LLM-driven cluster context selection.

Use case: given a list of stops (addresses), build a distance graph with walkable clusters
and smart routing (bike/transit/drive). The LLM decides which cluster preset to use based
on the trip characteristics when ``cluster_context`` is omitted.

Example::

    from travelplanner.agents.routing_agent import build_routing_graph

    graph = build_routing_graph()
    result = graph.invoke({
        "stops": [
            {"address": "Dam 1, Amsterdam", "name": "Start"},
            {"address": "Museumplein 6, Amsterdam", "name": "Rijksmuseum"},
        ],
        "api_key": "YOUR_KEY",
    })
    # result["artifact"] — AgentArtifactModel with place_distance_graph content
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.integrations.place_distance_graph import (
    build_place_distance_graph,
    parse_places_input_payload,
    place_distance_graph_config_for_context,
)
from travelplanner.integrations.routing_contracts import (
    ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
)
from travelplanner.schema.place_distance_graph import ClusterContext
from travelplanner.schema.system_state import AgentArtifactModel, MessageHistoryModel
from travelplanner.utils.llm import invoke_structured_model


_DEFAULT_MODEL = "openai:gpt-4o-mini"


@dataclass(frozen=True)
class RoutingAgentConfig:
    model_name: str = _DEFAULT_MODEL
    temperature: float = 0.0


def load_config_from_env() -> RoutingAgentConfig:
    return RoutingAgentConfig(
        model_name=os.getenv("TRAVELPLANNER_ROUTING_MODEL", _DEFAULT_MODEL),
        temperature=float(os.getenv("TRAVELPLANNER_ROUTING_TEMPERATURE", "0.0")),
    )


class ClusterPresetResponse(BaseModel):
    cluster_context: Literal["dense_urban", "mixed", "sparse"] = Field(
        description="Clustering mode: dense_urban for city centers with tight walking distances, mixed for balanced urban/suburban, sparse for road-trip style with far-apart stops"
    )


class RoutingAgentState(BaseModel):
    stops: list[dict[str, str]] = Field(
        default_factory=list,
        description="Each stop: address (required), optional name / category keys.",
    )
    cluster_context: ClusterContext | None = Field(
        default=None,
        description="dense_urban | mixed | sparse; omit to let the LLM infer from stops.",
    )
    api_key: str = Field(
        default="",
        description="Google Maps API key; falls back to GOOGLE_MAPS_API_KEY.",
    )
    model_name: str = Field(default=_DEFAULT_MODEL)
    temperature: float = Field(default=0.0)
    decided_cluster_context: ClusterContext | None = None
    artifact: AgentArtifactModel | None = None
    message_history: MessageHistoryModel | None = None
    error: str | None = None


ROUTING_SYSTEM_PROMPT = """You are a routing expert. Given a list of stops with addresses, decide the cluster preset."""

ROUTING_USER_PROMPT_TEMPLATE = """Stops:
{stop_summary}

Return JSON with the chosen cluster_context (dense_urban | mixed | sparse)."""


def _decide_cluster_context_llm(state: RoutingAgentState) -> dict[str, Any]:
    if not state.stops:
        return {"error": "No input data"}

    if state.cluster_context:
        return {"decided_cluster_context": state.cluster_context}

    stop_summary = "\n".join(
        f"- {s.get('name', 'unnamed')}: {s.get('address', 'no address')}"
        for s in state.stops
    )
    user_prompt = ROUTING_USER_PROMPT_TEMPLATE.format(stop_summary=stop_summary)

    try:
        structured_output, _user_prompt, raw_response = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ClusterPresetResponse,
        )
        history = MessageHistoryModel(
            user_agent="routing_agent",
            model=state.model_name,
            agent_ref="travelplanner.agents.routing_agent",
            messages=[
                {"role": "user", "content": _user_prompt},
                {"role": "assistant", "content": raw_response},
            ],
        )
        return {
            "decided_cluster_context": structured_output.cluster_context,
            "message_history": history,
        }
    except Exception as exc:
        return {
            "decided_cluster_context": "mixed",
            "error": f"LLM decision failed: {exc}",
        }


def _build_graph_node(state: RoutingAgentState) -> dict[str, Any]:
    if not state.stops:
        return {"error": "No input data"}

    api_key = state.api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return {"error": "Missing Google Maps API key"}

    cluster_ctx = state.decided_cluster_context or "mixed"

    try:
        raw = {"stops": state.stops}
        _, places, _ = parse_places_input_payload(raw)
        cfg = place_distance_graph_config_for_context(cluster_ctx)
        graph_model = build_place_distance_graph(places, api_key, config=cfg)
        st = graph_model.stats
        desc = (
            f"{st.place_count if st else 0} places, "
            f"{st.cluster_count if st else 0} clusters, "
            f"{st.edges_stored if st else 0} directed edges"
        )

        artifact = AgentArtifactModel(
            name="routing_place_graph",
            type=ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
            content=graph_model.model_dump(mode="json"),
            description=desc,
        )

        return {"artifact": artifact}

    except Exception as exc:
        return {"error": f"Graph build failed: {exc}"}


def build_routing_graph() -> Any:
    graph = StateGraph(RoutingAgentState)
    graph.add_node("decide_cluster", _decide_cluster_context_llm)
    graph.add_node("build_graph", _build_graph_node)
    graph.set_entry_point("decide_cluster")
    graph.add_edge("decide_cluster", "build_graph")
    graph.add_edge("build_graph", END)
    return graph.compile()


def run_routing_agent(
    stops: list[dict[str, str]],
    *,
    cluster_context: ClusterContext | None = None,
    api_key: str = "",
    model_name: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
) -> AgentArtifactModel:
    g = build_routing_graph()
    result = g.invoke(
        RoutingAgentState(
            stops=stops,
            cluster_context=cluster_context,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        ).model_dump(mode="json")
    )
    cluster_error = result.get("decide_cluster", {}).get("error")
    if cluster_error:
        raise RuntimeError(f"[decide_cluster] {cluster_error}")
    if result.get("error"):
        raise RuntimeError(result["error"])
    art = result.get("artifact")
    if art is None:
        raise RuntimeError("routing agent produced no artifact")
    if isinstance(art, dict):
        return AgentArtifactModel.model_validate(art)
    return art
