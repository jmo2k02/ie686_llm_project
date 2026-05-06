"""Deterministic execution of ``routing-check`` tasks → :class:`~travelplanner.schema.system_state.AgentArtifactModel`.

Use this from an **execution** layer or scripts. The task-planning LangGraph may also
invoke ``routing_check_agent`` when the reviewed task list includes valid ``routing-check``
tasks; this function stays the deterministic single-task entrypoint for CLIs/tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from travelplanner.integrations.google_routes import (
    compute_route_plan,
    resolve_travel_mode,
    route_plan_to_jsonable,
)
from travelplanner.integrations.place_distance_graph import (
    build_place_distance_graph,
    parse_places_input_payload,
    place_distance_graph_config_for_context,
)
from travelplanner.integrations.routing_contracts import (
    ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
    ARTIFACT_TYPE_ROUTE_TIMING,
    ROUTING_CHECK_TASK_TYPE,
    PlaceGraphFileTaskPayload,
    SingleOdTaskPayload,
    parse_routing_check_task_text,
    resolve_places_json_path,
)
from travelplanner.schema.route_plan import RouteDetailLevel
from travelplanner.schema.place_distance_graph import (
    MapPlaceInputModel,
    PlaceDistanceGraphBuildConfig,
)
from travelplanner.schema.system_state import AgentArtifactModel, TaskModel


def execute_routing_check_task(
    task: TaskModel,
    *,
    api_key: str,
    places_json_base_dir: Path | None = None,
    graph_config: PlaceDistanceGraphBuildConfig | None = None,
) -> AgentArtifactModel:
    """Run one ``routing-check`` task and return a versioned artifact (single OD or place graph)."""
    if task.type != ROUTING_CHECK_TASK_TYPE:
        msg = f"task.type must be {ROUTING_CHECK_TASK_TYPE!r}, got {task.type!r}"
        raise ValueError(msg)
    payload = parse_routing_check_task_text(task.text)
    key = api_key.strip()
    if not key:
        msg = "api_key is empty"
        raise ValueError(msg)

    if isinstance(payload, SingleOdTaskPayload):
        mode = resolve_travel_mode(payload.travel_mode)
        plan = compute_route_plan(
            origin=payload.origin_address,
            destination=payload.destination_address,
            api_key=key,
            travel_mode=mode,
            departure_time_rfc3339=payload.departure_time_rfc3339,
            detail_level=cast(RouteDetailLevel, payload.detail_level),
            include_transit_alternatives=payload.include_transit_alternatives,
        )
        blob = route_plan_to_jsonable(plan)
        desc = (
            f"{plan.request.travel_mode}: {plan.metrics.distance_km} km, "
            f"{plan.metrics.duration_seconds / 60:.0f} min"
        )
        return AgentArtifactModel(
            name=f"{task.name}_route_timing",
            type=ARTIFACT_TYPE_ROUTE_TIMING,
            content=blob,
            description=desc,
        )

    if isinstance(payload, PlaceGraphFileTaskPayload):
        path = resolve_places_json_path(
            payload.places_json_path, base_dir=places_json_base_dir
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        id_to_address: dict[str, str] = {}
        if isinstance(raw, dict) and "stops" in raw:
            file_ctx, places, id_to_address = parse_places_input_payload(raw)
            cluster_ctx = (
                payload.cluster_context
                if payload.cluster_context is not None
                else file_ctx
            )
            preset = place_distance_graph_config_for_context(cluster_ctx)
            if graph_config is None:
                cfg = preset
            else:
                cfg = preset.model_copy(
                    update=graph_config.model_dump(exclude_unset=True)
                )
        elif isinstance(raw, list):
            places = [MapPlaceInputModel.model_validate(r) for r in raw]
            cfg = graph_config or PlaceDistanceGraphBuildConfig()
            for p in places:
                if p.address and str(p.address).strip():
                    id_to_address[p.id] = str(p.address).strip()
        else:
            msg = (
                "places JSON must be an array of stops or an object with a stops array"
            )
            raise ValueError(msg)
        graph = build_place_distance_graph(places, key, config=cfg)
        desc = (
            f"{graph.stats.place_count} places, {graph.stats.cluster_count} clusters, "
            f"{graph.stats.edges_stored} directed edges"
        )
        return AgentArtifactModel(
            name=f"{task.name}_place_distance_graph",
            type=ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
            content=graph.model_dump(mode="json"),
            description=desc,
        )

    msg = f"unsupported routing payload: {type(payload).__name__}"
    raise TypeError(msg)
