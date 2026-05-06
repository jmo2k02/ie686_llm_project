"""Deterministic routing task execution (no HTTP when mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from travelplanner.integrations.routing_contracts import (
    ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
)
from travelplanner.integrations.routing_execution import execute_routing_check_task
from travelplanner.schema.route_plan import (
    RouteMetricModel,
    RoutePlanModel,
    RouteRequestModel,
)
from travelplanner.schema.place_distance_graph import (
    ClusterSummaryModel,
    EdgeEstimateModel,
    EdgeLegModel,
    GraphBuildStatsModel,
    PlaceDistanceGraphModel,
    PlaceNodeModel,
)
from travelplanner.schema.system_state import TaskModel


def _tiny_place_distance_graph() -> PlaceDistanceGraphModel:
    """Minimal valid graph: one walk cluster, two stops, no hub×hub matrix rows."""
    places = [
        PlaceNodeModel(
            id="stop_0",
            name="A",
            latitude=41.38,
            longitude=2.17,
            cluster_id=0,
            cluster_hub_id="stop_0",
        ),
        PlaceNodeModel(
            id="stop_1",
            name="B",
            latitude=41.381,
            longitude=2.171,
            cluster_id=0,
            cluster_hub_id="stop_0",
        ),
    ]
    clusters = [
        ClusterSummaryModel(
            cluster_id=0,
            member_place_ids=["stop_0", "stop_1"],
            hub_place_id="stop_0",
            centroid_latitude=41.3805,
            centroid_longitude=2.1705,
        )
    ]
    leg01 = EdgeLegModel(
        kind="walk_approx",
        from_place_id="stop_0",
        to_place_id="stop_1",
        travel_mode="WALK",
        distance_meters=150.0,
        duration_seconds=110.0,
        quality="haversine_walk",
    )
    leg10 = EdgeLegModel(
        kind="walk_approx",
        from_place_id="stop_1",
        to_place_id="stop_0",
        travel_mode="WALK",
        distance_meters=150.0,
        duration_seconds=110.0,
        quality="haversine_walk",
    )
    edges = [
        EdgeEstimateModel(
            from_place_id="stop_0",
            to_place_id="stop_1",
            distance_meters=150.0,
            duration_seconds=110.0,
            travel_mode_effective="WALK",
            quality="haversine_walk",
            legs=[leg01],
        ),
        EdgeEstimateModel(
            from_place_id="stop_1",
            to_place_id="stop_0",
            distance_meters=150.0,
            duration_seconds=110.0,
            travel_mode_effective="WALK",
            quality="haversine_walk",
            legs=[leg10],
        ),
    ]
    stats = GraphBuildStatsModel(
        place_count=2,
        cluster_count=1,
        hub_hub_matrix_elements=0,
        matrix_http_requests=0,
        edges_stored=2,
        edges_google_matrix=0,
        edges_haversine_walk=2,
        edges_hub_chain=0,
        edges_fallback=0,
    )
    return PlaceDistanceGraphModel(
        places=places,
        clusters=clusters,
        hub_hub_legs=[],
        edges=edges,
        stats=stats,
    )


def _minimal_plan() -> RoutePlanModel:
    return RoutePlanModel(
        request=RouteRequestModel(origin="X", destination="Y", travel_mode="DRIVE"),
        metrics=RouteMetricModel(
            distance_meters=100, distance_km=0.1, duration_seconds=60.0
        ),
    )


def test_place_graph_cluster_context_on_task_overrides_file(tmp_path: Path) -> None:
    places_file = tmp_path / "places.json"
    places_file.write_text(
        json.dumps(
            {
                "cluster_context": "sparse",
                "stops": [{"address": "Street A, City"}, {"address": "Street B, City"}],
            }
        ),
        encoding="utf-8",
    )
    preset_calls: list[str] = []

    def spy_preset(ctx: str):
        preset_calls.append(ctx)
        from travelplanner.integrations.place_distance_graph import (
            place_distance_graph_config_for_context as real_preset,
        )

        return real_preset(ctx)

    graph = MagicMock()
    graph.stats.place_count = 2
    graph.stats.cluster_count = 1
    graph.stats.edges_stored = 2
    graph.model_dump.return_value = {"schema_version": "1.5"}
    graph.model_copy.return_value = graph

    task = TaskModel(
        name="g",
        type="routing-check",
        text=json.dumps(
            {
                "kind": "place_graph_file",
                "places_json_path": "places.json",
                "cluster_context": "dense_urban",
            }
        ),
        is_valid=True,
        validation_comment=None,
    )
    with (
        patch(
            "travelplanner.integrations.routing_execution.place_distance_graph_config_for_context",
            side_effect=spy_preset,
        ),
        patch(
            "travelplanner.integrations.routing_execution.build_place_distance_graph",
            return_value=graph,
        ),
    ):
        execute_routing_check_task(task, api_key="k", places_json_base_dir=tmp_path)

    assert preset_calls == ["dense_urban"]


def test_execute_place_graph_file_returns_place_distance_graph_artifact(
    tmp_path: Path,
) -> None:
    """Many-place routing-check tasks yield a validated place_distance_graph (no HTTP when graph is mocked)."""
    places_file = tmp_path / "stops.json"
    places_file.write_text(
        json.dumps(
            {
                "stops": [
                    {"address": "Somewhere 1, Barcelona"},
                    {"address": "Somewhere 2, Barcelona"},
                ]
            }
        ),
        encoding="utf-8",
    )
    tiny = _tiny_place_distance_graph()
    task = TaskModel(
        name="graph_task",
        type="routing-check",
        text=json.dumps(
            {
                "kind": "place_graph_file",
                "places_json_path": "stops.json",
                "cluster_context": "mixed",
            }
        ),
        is_valid=True,
        validation_comment=None,
    )
    with patch(
        "travelplanner.integrations.routing_execution.build_place_distance_graph",
        return_value=tiny,
    ):
        art = execute_routing_check_task(
            task, api_key="k", places_json_base_dir=tmp_path
        )

    assert art.type == ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH
    assert art.name == "graph_task_place_distance_graph"
    round_trip = PlaceDistanceGraphModel.model_validate(art.content)
    assert round_trip.stats is not None
    assert round_trip.stats.place_count == 2
    assert round_trip.stats.cluster_count == 1
    assert len(round_trip.edges) == 2
    assert round_trip.hub_hub_leg("stop_0", "stop_0", "BICYCLE") is None


def test_execute_single_od_task() -> None:
    task = TaskModel(
        name="t1",
        type="routing-check",
        text='{"kind":"single_od","origin_address":"X","destination_address":"Y","travel_mode":"drive"}',
        is_valid=True,
        validation_comment=None,
    )
    with patch(
        "travelplanner.integrations.routing_execution.compute_route_plan",
        return_value=_minimal_plan(),
    ):
        art = execute_routing_check_task(task, api_key="k")
    assert art.type == "route_timing_artifact"
    assert art.name == "t1_route_timing"
