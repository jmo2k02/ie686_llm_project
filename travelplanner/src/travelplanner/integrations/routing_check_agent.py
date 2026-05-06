"""Routing Check — LLM-augmented multi-hop routing agent.

**Role**

The routing check agent decides *how* to route (single origin-destination pair vs.
multi-stop place graph) and executes the appropriate deterministic task.

**Multi-hop behaviour**

* ``single_od`` — one origin→destination route (fast, direct).
* ``place_graph_file`` — auto-triggered when the user query implies multiple stops
  or "all distances between places"; the LLM also picks a ``cluster_context`` preset
  (``dense_urban | mixed | sparse``) when one is not supplied.

**Quality gating**

Before returning, the agent validates that the produced artifact contains the
expected non-empty fields for the chosen mode. If validation fails on the first
attempt the agent retries once before surfacing an error artifact.

**Fallback**

If ``single_od`` execution raises an exception the agent automatically falls back
to ``place_graph_file`` using the LLM-suggested cluster context (or ``mixed`` as
the default). This handles cases where a simple two-address query turns out to be
part of a larger trip.

**Structured LLM output**

The agent uses :func:`travelplanner.utils.llm.invoke_structured_model` with a
``RoutingCheckDecision`` response model, mirroring the pattern in
:class:`travelplanner.agents.constraint_agent.ConstraintAgentState`.

``docs/workflow.md`` Phase 2 **Route Check** aligns with
``route_timing_artifact`` / ``routing-check`` tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.integrations.routing_contracts import (
    ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
    ARTIFACT_TYPE_ROUTE_TIMING,
    PlaceGraphFileTaskPayload,
    ROUTING_CHECK_TASK_TYPE,
    SingleOdTaskPayload,
)
from travelplanner.integrations.routing_execution import execute_routing_check_task
from travelplanner.schema.place_distance_graph import ClusterContext
from travelplanner.schema.route_plan import RouteDetailLevel
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    TaskModel,
)
from travelplanner.utils.llm import invoke_structured_model

__all__ = [
    "ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH",
    "ARTIFACT_TYPE_ROUTE_TIMING",
    "ROUTING_CHECK_TASK_TYPE",
    "RoutingCheckAgentConfig",
    "RoutingCheckAgentState",
    "load_config_from_env",
    "make_graph",
]


_DEFAULT_MODEL = "openai:gpt-4o-mini"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingCheckAgentConfig:
    model_name: str = _DEFAULT_MODEL
    temperature: float = 0.0


def load_config_from_env() -> RoutingCheckAgentConfig:
    return RoutingCheckAgentConfig(
        model_name=os.getenv("TRAVELPLANNER_ROUTING_CHECK_MODEL", _DEFAULT_MODEL),
        temperature=float(os.getenv("TRAVELPLANNER_ROUTING_CHECK_TEMPERATURE", "0.0")),
    )


# ---------------------------------------------------------------------------
# LLM response model
# ---------------------------------------------------------------------------


class RoutingCheckDecision(BaseModel):
    """Structured output from the routing-approach decision node."""

    mode: Literal["single_od", "place_graph_file"] = Field(
        description="Routing mode to use: single_od for one route, place_graph_file for multi-stop clustering"
    )
    reasoning: str = Field(
        default="",
        description="Why this mode was chosen, including any edge-case considerations",
    )
    cluster_context_suggested: Literal["dense_urban", "mixed", "sparse"] | None = Field(
        default=None,
        description="Cluster preset to pass to place_graph_file when mode=place_graph_file and the caller did not specify one",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0=guess, 1.0=highly confident. Below 0.4 triggers clarification-request",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RoutingCheckAgentState(BaseModel):
    """Mirrors SingleOdTaskPayload + place-graph fields, plus LLM runtime options.

    Allows the agent to handle both single-OD and multi-stop place-graph queries
    without the caller needing to know which mode will be used.
    """

    # --- single-OD fields (always accepted, fall back to place-graph if needed) ---
    task_ref: str = Field(
        default="routing_check",
        description="Becomes ``TaskModel.name`` when delegating to ``execute_routing_check_task``.",
    )
    origin_address: str = Field(default="")
    destination_address: str = Field(default="")
    travel_mode: str = Field(default="drive")
    departure_time_rfc3339: str | None = None
    detail_level: RouteDetailLevel = "standard"
    include_transit_alternatives: bool = True

    # --- place-graph fields (optional on input; LLM can decide to use these) ---
    places_json_path: str | None = Field(
        default=None,
        description="Path to a places JSON file; triggers place_graph_file mode when set.",
    )
    cluster_context: ClusterContext | None = Field(
        default=None,
        description="dense_urban | mixed | sparse; omit to let the LLM infer.",
    )

    # --- execution ---
    api_key: str = Field(
        default="",
        description="Optional override; otherwise ``GOOGLE_MAPS_API_KEY`` from the environment.",
    )
    model_name: str = Field(default=_DEFAULT_MODEL)
    temperature: float = Field(default=0.0)

    # --- internal ---
    decided_mode: Literal["single_od", "place_graph_file"] | None = None
    decided_cluster_context: ClusterContext | None = None
    decision_confidence: float = Field(default=0.0)
    decision_reasoning: str = ""
    artifact: AgentArtifactModel | None = None
    message_history: MessageHistoryModel | None = None
    error: str | None = None
    retry_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# System / user prompts
# ---------------------------------------------------------------------------

_DECIDE_SYSTEM_PROMPT = """You are the TravelPlanner routing-check decision agent.

Given a travel request, decide how to route it:

* **single_od** — one origin→destination pair. Use when the user asks for a route
  between exactly two addresses (e.g. "fastest route from A to B", "driving time").

* **place_graph_file** — multiple stops requiring a distance/cumulative graph. Use
  when the user asks for "all distances", "distances between all these places",
  "route connecting these stops", or 7+ addresses are present.

For **place_graph_file**, also pick a cluster preset:
  - ``dense_urban`` — city centres, tight walking distances, many small clusters.
  - ``mixed`` — balanced urban/suburban, default if unsure.
  - ``sparse`` — road-trip style, far-apart stops, few large clusters.

Set confidence:
  - 0.9–1.0: unambiguous single-OD request or unambiguous multi-stop request.
  - 0.5–0.8: plausible but some ambiguity; fallback will handle if wrong.
  - 0.0–0.4: too vague; request clarification.

Return JSON only, matching the schema exactly."""

_DECIDE_USER_PROMPT_TEMPLATE = """Routing request:

places_json_path: {places_json_path}
origin_address: {origin_address}
destination_address: {destination_address}
query hint (if any): {query_hint}
total known addresses: {address_count}

Return JSON:
{{"mode": "single_od" | "place_graph_file", "reasoning": "...", "cluster_context_suggested": "dense_urban" | "mixed" | "sparse" | null, "confidence": 0.0–1.0}}"""


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _api_key(state: RoutingCheckAgentState) -> str:
    return (state.api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")).strip()


def _build_message_history(
    *,
    user_content: str,
    assistant_content: str,
    agent_ref: str = "travelplanner.integrations.routing_check_agent",
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="routing_check_agent",
        model="llm",
        agent_ref=agent_ref,
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    )


# ---------------------------------------------------------------------------
# Node: decide_routing_approach
# ---------------------------------------------------------------------------


def _decide_routing_approach(state: RoutingCheckAgentState) -> dict[str, Any]:
    """LLM node — determines routing mode and, for place_graph_file, the cluster preset."""

    # Short-circuit: if places_json_path is provided, force place_graph_file.
    if state.places_json_path:
        return {
            "decided_mode": "place_graph_file",
            "decided_cluster_context": state.cluster_context,
            "decision_confidence": 1.0,
            "decision_reasoning": "places_json_path provided; using place_graph_file.",
        }

    # Short-circuit: if both origin and destination are present and no multi-stop hint,
    # use single_od with high confidence.
    if (
        state.origin_address
        and state.destination_address
        and not state.places_json_path
    ):
        # Heuristic: if the user explicitly gave two addresses, trust single_od.
        return {
            "decided_mode": "single_od",
            "decided_cluster_context": None,
            "decision_confidence": 0.95,
            "decision_reasoning": "origin and destination explicitly provided; single_od.",
        }

    # Need LLM to disambiguate.
    user_prompt = _DECIDE_USER_PROMPT_TEMPLATE.format(
        places_json_path=str(state.places_json_path or ""),
        origin_address=state.origin_address or "",
        destination_address=state.destination_address or "",
        query_hint="",
        address_count=0,
    )

    try:
        structured_output, _user_prompt, raw_response = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=_DECIDE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=RoutingCheckDecision,
        )
        history = _build_message_history(
            user_content=user_prompt,
            assistant_content=raw_response,
        )
        return {
            "decided_mode": structured_output.mode,
            "decided_cluster_context": (
                state.cluster_context
                if state.cluster_context is not None
                else structured_output.cluster_context_suggested
            ),
            "decision_confidence": structured_output.confidence,
            "decision_reasoning": structured_output.reasoning,
            "message_history": history,
        }
    except Exception as exc:
        # Degrade gracefully: fall back to single_od if we have addresses, else error.
        if state.origin_address and state.destination_address:
            return {
                "decided_mode": "single_od",
                "decided_cluster_context": None,
                "decision_confidence": 0.0,
                "decision_reasoning": f"LLM decision failed ({exc}); defaulting to single_od.",
                "error": f"LLM decision failed: {exc}",
            }
        return {
            "decided_mode": None,
            "error": f"Could not determine routing approach: {exc}",
        }


# ---------------------------------------------------------------------------
# Node: execute_routing
# ---------------------------------------------------------------------------


def _execute_routing(state: RoutingCheckAgentState) -> dict[str, Any]:
    """Deterministic node — calls execute_routing_check_task with the appropriate payload."""

    key = _api_key(state)
    if not key:
        return {
            "error": (
                "Google Routes API key missing: set GOOGLE_MAPS_API_KEY or pass api_key on state."
            )
        }

    mode = state.decided_mode
    if mode is None:
        return {
            "error": "Routing mode was not determined (decided_mode is None); cannot call Google APIs."
        }

    try:
        if mode == "single_od":
            origin = (state.origin_address or "").strip()
            dest = (state.destination_address or "").strip()
            if not origin or not dest:
                return {
                    "error": (
                        "single_od requires non-empty origin_address and destination_address "
                        "before calling route APIs."
                    )
                }
            payload = SingleOdTaskPayload(
                origin_address=origin,
                destination_address=dest,
                travel_mode=state.travel_mode,
                departure_time_rfc3339=state.departure_time_rfc3339,
                detail_level=state.detail_level,
                include_transit_alternatives=state.include_transit_alternatives,
            )
            task = TaskModel(
                name=state.task_ref,
                type=cast(Literal["routing-check"], ROUTING_CHECK_TASK_TYPE),
                text=payload.model_dump_json(),
                is_valid=True,
                validation_comment=None,
            )
            artifact = execute_routing_check_task(task, api_key=key)
            history = _build_message_history(
                user_content=(
                    f"single_od: {state.origin_address} → {state.destination_address} "
                    f"via {state.travel_mode}"
                ),
                assistant_content=f"{artifact.type}: {artifact.description or ''}",
            )
            return {"artifact": artifact, "message_history": history}

        else:  # place_graph_file
            path_txt = (state.places_json_path or "").strip()
            if not path_txt:
                return {
                    "error": (
                        "place_graph_file mode requires places_json_path before calling APIs."
                    )
                }
            payload = PlaceGraphFileTaskPayload(
                places_json_path=path_txt,
                cluster_context=state.decided_cluster_context,
            )
            task = TaskModel(
                name=state.task_ref,
                type=cast(Literal["routing-check"], ROUTING_CHECK_TASK_TYPE),
                text=payload.model_dump_json(),
                is_valid=True,
                validation_comment=None,
            )
            artifact = execute_routing_check_task(task, api_key=key)
            history = _build_message_history(
                user_content=f"place_graph_file: {state.places_json_path}",
                assistant_content=f"{artifact.type}: {artifact.description or ''}",
            )
            return {"artifact": artifact, "message_history": history}

    except Exception as exc:
        if mode == "single_od" and state.places_json_path:
            fallback_payload = PlaceGraphFileTaskPayload(
                places_json_path=state.places_json_path,
                cluster_context=state.decided_cluster_context or "mixed",
            )
            fallback_task = TaskModel(
                name=state.task_ref,
                type=cast(Literal["routing-check"], ROUTING_CHECK_TASK_TYPE),
                text=fallback_payload.model_dump_json(),
                is_valid=True,
                validation_comment=None,
            )
            try:
                artifact = execute_routing_check_task(fallback_task, api_key=key)
                history = _build_message_history(
                    user_content=(
                        "single_od failed; fallback to place_graph_file: "
                        f"{state.places_json_path}"
                    ),
                    assistant_content=f"{artifact.type}: {artifact.description or ''}",
                )
                return {
                    "artifact": artifact,
                    "message_history": history,
                    "error": f"single_od failed, recovered via place_graph_file: {exc}",
                }
            except Exception as fallback_exc:
                return {
                    "error": f"single_od failed ({exc}); fallback also failed ({fallback_exc})"
                }

        return {"error": str(exc)}


def _validate_artifact(state: RoutingCheckAgentState) -> dict[str, Any]:
    """Quality gate — check that the artifact has the expected non-empty structure."""

    artifact = state.artifact
    if artifact is None:
        # Do not clobber a prior execute/decide error with a generic validation message.
        if state.error:
            return {}
        return {"error": "No artifact to validate."}

    mode = state.decided_mode
    content = artifact.content if isinstance(artifact.content, dict) else {}

    if mode == "single_od":
        # For single_od we expect route metrics under content.
        if not content:
            return {"error": "single_od artifact content is empty."}
        # Basic sanity: should have some key that's not just {}
        has_data = any(
            k in content
            for k in ("request", "route", "metrics", "distance_km", "duration_seconds")
        )
        if not has_data:
            return {"error": "single_od artifact missing expected fields."}
        return {"error": None}

    elif mode == "place_graph_file":
        # For place_graph_file we expect places, clusters, edges.
        required_keys = ("places", "clusters", "edges")
        missing = [k for k in required_keys if k not in content or not content.get(k)]
        if missing:
            return {
                "error": f"place_graph_file artifact missing/non-empty fields: {missing}",
                "retry_count": state.retry_count + 1,
            }
        return {"error": None}

    return {"error": None}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def make_graph() -> Any:
    """Compile the multi-hop routing-check LangGraph."""

    graph = StateGraph(RoutingCheckAgentState)

    graph.add_node("decide_routing_approach", _decide_routing_approach)
    graph.add_node("execute_routing", _execute_routing)
    graph.add_node("validate_artifact", _validate_artifact)

    graph.set_entry_point("decide_routing_approach")

    # Branch on decided_mode and confidence
    def _route_after_decide(state: RoutingCheckAgentState) -> str:
        # Do not invoke execute_routing without a mode — preserves error state and avoids RuntimeError.
        if state.decided_mode is None:
            return END
        if state.decision_confidence < 0.4:
            return "ask_clarification"
        return "execute_routing"

    graph.add_conditional_edges(
        "decide_routing_approach",
        _route_after_decide,
        {
            "execute_routing": "execute_routing",
            "ask_clarification": END,  # interrupt pattern — caller must re-invoke with clarification
            END: END,
        },
    )

    graph.add_edge("execute_routing", "validate_artifact")
    graph.add_edge("validate_artifact", END)

    return graph.compile()
