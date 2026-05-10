"""Stable, agent-callable routing helpers (no raises for expected failures).

**Orchestrator pattern (recommended)**

1. **Build once** — call :func:`build_place_graph_with_routing_agent` (wraps the
   LangGraph **routing agent**: optional LLM cluster preset, then Google graph build).
   Persist ``result["graph"]`` (``place_distance_graph`` dict) in session / ``agent_artifacts``.
2. **Query on demand** — call :func:`distance_between_places`, :func:`closest_places_to_target`,
   or lower-level :mod:`travelplanner.routing_lookup` on that dict. **No Google calls** for reads.

For a **fixed** cluster preset without the routing-agent LLM step, use
:func:`build_distance_graph_from_stops` instead. For a **single** A→B leg without a graph,
use :func:`route_one_leg`.

Tool schemas
------------
- :data:`ORCHESTRATOR_ROUTING_TOOL_SCHEMAS` — build + read tools for the pattern above.
- :data:`ROUTING_TOOL_SCHEMAS` — superset (includes deterministic graph build + single leg).

Each callable returns a JSON-friendly dict with ``ok: bool`` and ``stage`` on failure.
Inputs are validated **before** Google APIs where applicable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal, cast

from travelplanner.integrations.place_distance_graph import (
    build_place_distance_graph,
    parse_places_input_payload,
    place_distance_graph_config_for_context,
)
from travelplanner.integrations.routing_contracts import (
    ROUTING_CHECK_TASK_TYPE,
    SingleOdTaskPayload,
)
from travelplanner.integrations.routing_execution import execute_routing_check_task
from travelplanner.routing_lookup.queries import (
    ClosestResult,
    DistanceResult,
    PlaceResolutionError,
    closest_to,
    distance_between,
)
from travelplanner.schema.place_distance_graph import ClusterContext
from travelplanner.schema.route_plan import RouteDetailLevel
from travelplanner.schema.system_state import AgentArtifactModel, TaskModel


def _normalize_maps_api_key(api_key: str | None) -> tuple[str | None, str | None]:
    key = (api_key if api_key is not None else os.getenv("GOOGLE_MAPS_API_KEY", "")).strip()
    if not key:
        return None, (
            "Missing Google Maps API key: pass api_key=... or set GOOGLE_MAPS_API_KEY."
        )
    return key, None


def build_place_graph_with_routing_agent(
    stops: list[dict[str, str]],
    *,
    cluster_context: ClusterContext | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build a ``place_distance_graph`` via the **routing agent** (reliable multi-stop path).

    The orchestrator should call this **once** per trip segment that needs a graph, keep
    ``graph`` in context, then answer follow-ups with :func:`distance_between_places` /
    :func:`closest_places_to_target` only.

    When ``cluster_context`` is omitted, the agent may call an LLM to pick
    ``dense_urban`` / ``mixed`` / ``sparse`` before geocoding and matrix work.
    """
    if not stops:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "stops must be a non-empty list of dicts (address/name keys).",
        }

    key = (api_key if api_key is not None else os.getenv("GOOGLE_MAPS_API_KEY", "")).strip()
    if not key:
        return {
            "ok": False,
            "stage": "api_key",
            "error": (
                "Missing Google Maps API key: pass api_key=... or set GOOGLE_MAPS_API_KEY."
            ),
        }

    kwargs: dict[str, Any] = {
        "stops": stops,
        "cluster_context": cluster_context,
        "api_key": key,
        "temperature": temperature,
    }
    if model_name is not None:
        kwargs["model_name"] = model_name

    from travelplanner.agents.routing_agent import run_routing_graph_result

    raw = run_routing_graph_result(**kwargs)
    if not raw.get("ok"):
        return {
            "ok": False,
            "stage": "routing_agent",
            "error": raw.get("error") or "routing agent failed to produce a graph",
            "decided_cluster_context": raw.get("decided_cluster_context"),
        }

    art = raw.get("artifact") or {}
    content = art.get("content") if isinstance(art.get("content"), dict) else {}
    if not content.get("places"):
        return {
            "ok": False,
            "stage": "routing_agent",
            "error": raw.get("error") or "artifact missing place_distance_graph content",
            "decided_cluster_context": raw.get("decided_cluster_context"),
        }

    stats = content.get("stats") if isinstance(content.get("stats"), dict) else {}
    summary = art.get("description") or (
        f"{stats.get('place_count', '?')} places, "
        f"{stats.get('cluster_count', '?')} clusters, "
        f"{stats.get('edges_stored', '?')} directed edges"
    )
    out: dict[str, Any] = {
        "ok": True,
        "stage": "done",
        "summary": summary,
        "graph": content,
        "decided_cluster_context": raw.get("decided_cluster_context"),
        "stats": stats,
    }
    if raw.get("message_history") is not None:
        out["message_history"] = raw["message_history"]
    return out


def _artifact_to_json(artifact: AgentArtifactModel) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "type": artifact.type,
        "description": artifact.description,
        "content": artifact.content if isinstance(artifact.content, dict) else {},
    }


def route_one_leg(
    *,
    origin_address: str,
    destination_address: str,
    travel_mode: str = "drive",
    api_key: str | None = None,
    task_name: str = "routing_tool",
    departure_time_rfc3339: str | None = None,
    detail_level: RouteDetailLevel = "standard",
    include_transit_alternatives: bool = True,
) -> dict[str, Any]:
    """Compute one origin→destination route (live Google Routes API).

    Validates addresses before contacting Google. Never raises for user errors.
    """
    o = origin_address.strip()
    d = destination_address.strip()
    if not o or not d:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "origin_address and destination_address must both be non-empty.",
        }

    mk, mk_err = _normalize_maps_api_key(api_key)
    if mk_err:
        return {"ok": False, "stage": "api_key", "error": mk_err}

    payload = SingleOdTaskPayload(
        origin_address=o,
        destination_address=d,
        travel_mode=travel_mode,
        departure_time_rfc3339=departure_time_rfc3339,
        detail_level=detail_level,
        include_transit_alternatives=include_transit_alternatives,
    )
    task = TaskModel(
        name=task_name,
        type=cast(Literal["routing-check"], ROUTING_CHECK_TASK_TYPE),
        text=payload.model_dump_json(),
        is_valid=True,
        validation_comment=None,
    )
    try:
        artifact = execute_routing_check_task(task, api_key=cast(str, mk))
    except Exception as exc:
        return {
            "ok": False,
            "stage": "google_route",
            "error": f"Route computation failed: {exc}",
        }

    blob = artifact.content if isinstance(artifact.content, dict) else {}
    metrics = blob.get("metrics") if isinstance(blob, dict) else {}
    summary = artifact.description or (
        "Route computed"
        if metrics
        else "Route artefact produced (inspect content.metrics)"
    )
    return {
        "ok": True,
        "stage": "done",
        "summary": summary,
        "artifact": _artifact_to_json(artifact),
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def build_distance_graph_from_stops(
    stops: list[dict[str, str]],
    *,
    cluster_context: ClusterContext | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build a clustered place-distance graph from address stops (live Google APIs).

    Parses ``stops`` locally first; skips Google entirely if parsing yields nothing usable.
    """
    if not stops:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "stops must be a non-empty list of dicts (address/name keys).",
        }

    mk, mk_err = _normalize_maps_api_key(api_key)
    if mk_err:
        return {"ok": False, "stage": "api_key", "error": mk_err}

    raw = {"stops": stops}
    try:
        _, places, _ = parse_places_input_payload(raw)
    except Exception as exc:
        return {
            "ok": False,
            "stage": "parse_stops",
            "error": f"Could not interpret stops payload: {exc}",
        }

    if not places:
        return {
            "ok": False,
            "stage": "parse_stops",
            "error": "No usable places after parsing stops (need address or lat/lng per row).",
        }

    cluster: ClusterContext = cluster_context if cluster_context is not None else "mixed"
    cfg = place_distance_graph_config_for_context(cluster)

    try:
        graph_model = build_place_distance_graph(places, cast(str, mk), config=cfg)
    except Exception as exc:
        return {
            "ok": False,
            "stage": "build_graph",
            "error": f"Graph build failed: {exc}",
        }

    stats = graph_model.stats
    desc = (
        f"{stats.place_count} places, {stats.cluster_count} clusters, "
        f"{stats.edges_stored} directed edges"
        if stats
        else "graph built"
    )
    content = graph_model.model_dump(mode="json")
    return {
        "ok": True,
        "stage": "done",
        "summary": desc,
        "cluster_context": cluster,
        "stats": stats.model_dump(mode="json") if stats else {},
        "graph": content,
    }


def distance_between_places(
    graph: Mapping[str, Any] | dict[str, Any],
    from_place: str,
    to_place: str,
    *,
    preference: Literal[
        "duration",
        "distance",
        "fastest",
        "shortest",
        "walk",
        "bike",
        "bicycle",
        "transit",
        "drive",
        "cheapest",
    ] = "duration",
) -> dict[str, Any]:
    """Resolve two names/IDs on an existing ``place_distance_graph`` and return distances.

    **No external API.** Fuzzy name matching can return ambiguity — see ``candidates`` in failures.
    """
    a = from_place.strip()
    b = to_place.strip()
    if not a or not b:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "from_place and to_place must be non-empty.",
        }

    try:
        result: DistanceResult = distance_between(
            dict(graph),
            from_name=a,
            to_name=b,
            preference=preference,
        )
    except PlaceResolutionError as exc:
        alts = [
            {"place_id": c.place_id, "name": c.name, "score": c.score}
            for c in (exc.candidates or [])[:5]
        ]
        return {
            "ok": False,
            "stage": "resolve_place",
            "error": str(exc),
            "query": exc.query,
            "candidates": alts,
        }
    except Exception as exc:
        return {
            "ok": False,
            "stage": "lookup",
            "error": str(exc),
        }

    opt = result.option
    return {
        "ok": True,
        "stage": "done",
        "summary": (
            f"{result.from_place.matched.name} ({result.from_place.matched.place_id}) → "
            f"{result.to_place.matched.name} ({result.to_place.matched.place_id}): "
            f"{opt.travel_mode}, {opt.distance_km:.2f} km, {opt.duration_minutes:.1f} min"
        ),
        "from_place_id": result.from_place.matched.place_id,
        "to_place_id": result.to_place.matched.place_id,
        "travel_mode": opt.travel_mode,
        "distance_meters": opt.distance_meters,
        "distance_km": round(opt.distance_km, 3),
        "duration_seconds": opt.duration_seconds,
        "duration_minutes": round(opt.duration_minutes, 2),
        "quality": result.explanation,
        "preference": preference,
    }


def closest_places_to_target(
    graph: Mapping[str, Any] | dict[str, Any],
    target_name: str,
    candidate_names: list[str],
    *,
    preference: Literal[
        "duration",
        "distance",
        "fastest",
        "shortest",
        "walk",
        "bike",
        "bicycle",
        "transit",
        "drive",
        "cheapest",
    ] = "duration",
) -> dict[str, Any]:
    """Rank ``candidate_names`` by travel time/distance to ``target_name`` on an existing graph.

    **No external API.** Same fuzzy resolution rules as :func:`distance_between_places`.
    """
    t = target_name.strip()
    if not t:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "target_name must be non-empty.",
        }
    if not candidate_names:
        return {
            "ok": False,
            "stage": "validate_input",
            "error": "candidate_names must be a non-empty list.",
        }

    try:
        result: ClosestResult = closest_to(
            dict(graph),
            target_name=t,
            candidate_names=candidate_names,
            preference=preference,
        )
    except PlaceResolutionError as exc:
        alts = [
            {"place_id": c.place_id, "name": c.name, "score": c.score}
            for c in (exc.candidates or [])[:5]
        ]
        return {
            "ok": False,
            "stage": "resolve_place",
            "error": str(exc),
            "query": exc.query,
            "candidates": alts,
        }
    except ValueError as exc:
        return {"ok": False, "stage": "lookup", "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "stage": "lookup", "error": str(exc)}

    ranked = []
    for item in result.ranked:
        o = item.option
        ranked.append(
            {
                "candidate_query": item.candidate_query,
                "place_id": item.candidate.matched.place_id,
                "name": item.candidate.matched.name,
                "travel_mode": o.travel_mode,
                "distance_km": round(o.distance_km, 3),
                "duration_minutes": round(o.duration_minutes, 2),
            }
        )
    w = result.winner
    return {
        "ok": True,
        "stage": "done",
        "summary": (
            f"Closest to {result.target.matched.name}: "
            f"{w.candidate.matched.name} ({w.option.duration_minutes:.1f} min)"
        ),
        "target_place_id": result.target.matched.place_id,
        "winner": ranked[0] if ranked else None,
        "ranked": ranked,
        "preference": preference,
    }


ORCHESTRATOR_ROUTING_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "build_place_graph_with_routing_agent",
            "description": (
                "ONE-TIME build: run the routing agent to produce a place_distance_graph "
                "(clusters + edges + hub matrix). Call once per trip; retain `graph` for follow-ups. "
                "Omit cluster_context to let the agent infer dense_urban/mixed/sparse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "address": {"type": "string"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "cluster_context": {
                        "type": "string",
                        "enum": ["dense_urban", "mixed", "sparse"],
                    },
                },
                "required": ["stops"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distance_between_places",
            "description": (
                "READ-ONLY: distance/duration between two places on a graph you already built "
                "(pass the same graph dict). No API cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "object"},
                    "from_place": {"type": "string"},
                    "to_place": {"type": "string"},
                    "preference": {
                        "type": "string",
                        "enum": [
                            "duration",
                            "distance",
                            "fastest",
                            "shortest",
                            "transit",
                            "walk",
                            "drive",
                            "bike",
                            "cheapest",
                        ],
                        "default": "duration",
                    },
                },
                "required": ["graph", "from_place", "to_place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "closest_places_to_target",
            "description": (
                "READ-ONLY: rank candidates by travel time/distance to a target on an existing graph. "
                "No API cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "object"},
                    "target_name": {"type": "string"},
                    "candidate_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "preference": {
                        "type": "string",
                        "enum": [
                            "duration",
                            "distance",
                            "fastest",
                            "shortest",
                            "transit",
                            "walk",
                            "drive",
                            "bike",
                            "cheapest",
                        ],
                        "default": "duration",
                    },
                },
                "required": ["graph", "target_name", "candidate_names"],
            },
        },
    },
)


ROUTING_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    *ORCHESTRATOR_ROUTING_TOOL_SCHEMAS,
    {
        "type": "function",
        "function": {
            "name": "route_one_leg",
            "description": (
                "Get driving/transit/etc. metrics between two addresses (Google Routes). "
                "Use when no place_distance_graph exists yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_address": {"type": "string"},
                    "destination_address": {"type": "string"},
                    "travel_mode": {
                        "type": "string",
                        "enum": ["drive", "walking", "bicycling", "transit"],
                        "default": "drive",
                    },
                },
                "required": ["origin_address", "destination_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_distance_graph_from_stops",
            "description": (
                "Deterministic graph build: same geometry as the routing agent but **no** LLM "
                "cluster step — pass cluster_context or defaults to mixed. Prefer "
                "build_place_graph_with_routing_agent when the orchestrator wants automatic preset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "address": {"type": "string"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "cluster_context": {
                        "type": "string",
                        "enum": ["dense_urban", "mixed", "sparse"],
                    },
                },
                "required": ["stops"],
            },
        },
    },
)


__all__ = [
    "ORCHESTRATOR_ROUTING_TOOL_SCHEMAS",
    "ROUTING_TOOL_SCHEMAS",
    "build_distance_graph_from_stops",
    "build_place_graph_with_routing_agent",
    "closest_places_to_target",
    "distance_between_places",
    "route_one_leg",
]
