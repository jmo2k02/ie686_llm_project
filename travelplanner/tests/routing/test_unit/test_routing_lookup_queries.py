from __future__ import annotations

import pytest

from travelplanner.integrations.routing_contracts import ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH
from travelplanner.routing_lookup import closest_to, distance_between, resolve_place_id
from travelplanner.routing_lookup.queries import PlaceResolutionError
from travelplanner.schema.system_state import AgentArtifactModel


def _tiny_graph() -> dict:
    # Minimal place_distance_graph-like dict for RoutingLookup:
    # 3 places, with p0+p1 same cluster (walk), p2 separate cluster (hub-chain).
    places = [
        {
            "id": "stop_0",
            "name": "Hotel Central",
            "latitude": 0.0,
            "longitude": 0.0,
            "cluster_id": 0,
            "cluster_hub_id": "stop_0",
        },
        {
            "id": "stop_1",
            "name": "Rijksmuseum",
            "latitude": 0.001,
            "longitude": 0.001,
            "cluster_id": 0,
            "cluster_hub_id": "stop_0",
        },
        {
            "id": "stop_2",
            "name": "Keukenhof Gardens",
            "latitude": 0.1,
            "longitude": 0.1,
            "cluster_id": 1,
            "cluster_hub_id": "stop_2",
        },
    ]
    clusters = [
        {
            "cluster_id": 0,
            "member_place_ids": ["stop_0", "stop_1"],
            "hub_place_id": "stop_0",
            "centroid_latitude": 0.0005,
            "centroid_longitude": 0.0005,
        },
        {
            "cluster_id": 1,
            "member_place_ids": ["stop_2"],
            "hub_place_id": "stop_2",
            "centroid_latitude": 0.1,
            "centroid_longitude": 0.1,
        },
    ]
    # Two clusters => ordered hub pairs = 2 (0->1, 1->0) per mode.
    hub_hub_legs = []
    for mode, dur01, dur10 in [
        ("DRIVE", 1800.0, 1850.0),
        ("BICYCLE", 5400.0, 5600.0),
        ("TRANSIT", 3600.0, 3700.0),
    ]:
        hub_hub_legs.append(
            {
                "from_hub_id": "stop_0",
                "to_hub_id": "stop_2",
                "travel_mode": mode,
                "matrix_policy_band": "drive" if mode == "DRIVE" else "bicycle",
                "distance_meters": 25000,
                "duration_seconds": dur01,
                "haversine_meters": 20000.0,
                "source": "google_route_matrix",
                "condition": "ROUTE_EXISTS",
            }
        )
        hub_hub_legs.append(
            {
                "from_hub_id": "stop_2",
                "to_hub_id": "stop_0",
                "travel_mode": mode,
                "matrix_policy_band": "drive" if mode == "DRIVE" else "bicycle",
                "distance_meters": 25000,
                "duration_seconds": dur10,
                "haversine_meters": 20000.0,
                "source": "google_route_matrix",
                "condition": "ROUTE_EXISTS",
            }
        )

    # Build full directed edges for 3 places => 6 edges.
    def walk_edge(a: str, b: str, dist: float, dur: float) -> dict:
        return {
            "from_place_id": a,
            "to_place_id": b,
            "distance_meters": dist,
            "duration_seconds": dur,
            "travel_mode_effective": "WALK",
            "quality": "haversine_walk",
            "legs": [
                {
                    "kind": "walk_approx",
                    "from_place_id": a,
                    "to_place_id": b,
                    "travel_mode": "WALK",
                    "distance_meters": dist,
                    "duration_seconds": dur,
                    "quality": "haversine_walk",
                }
            ],
        }

    def hub_edge(a: str, b: str, from_hub: str, to_hub: str, mode: str, dur: float) -> dict:
        return {
            "from_place_id": a,
            "to_place_id": b,
            "distance_meters": 25000.0,
            "duration_seconds": dur,
            "travel_mode_effective": "COMPOSED",
            "quality": "hub_chain",
            "primary_hub_travel_mode": mode,
            "legs": [
                {
                    "kind": "walk_approx",
                    "from_place_id": a,
                    "to_place_id": from_hub,
                    "travel_mode": "WALK",
                    "distance_meters": 0.0,
                    "duration_seconds": 0.0,
                    "quality": "haversine_walk",
                },
                {
                    "kind": "matrix",
                    "from_place_id": from_hub,
                    "to_place_id": to_hub,
                    "travel_mode": mode,
                    "distance_meters": 25000.0,
                    "duration_seconds": dur,
                    "quality": "google_matrix",
                },
                {
                    "kind": "walk_approx",
                    "from_place_id": to_hub,
                    "to_place_id": b,
                    "travel_mode": "WALK",
                    "distance_meters": 0.0,
                    "duration_seconds": 0.0,
                    "quality": "haversine_walk",
                },
            ],
        }

    edges = [
        walk_edge("stop_0", "stop_1", 300.0, 240.0),
        walk_edge("stop_1", "stop_0", 300.0, 240.0),
        hub_edge("stop_0", "stop_2", "stop_0", "stop_2", "DRIVE", 1800.0),
        hub_edge("stop_2", "stop_0", "stop_2", "stop_0", "DRIVE", 1850.0),
        hub_edge("stop_1", "stop_2", "stop_0", "stop_2", "DRIVE", 1800.0),
        hub_edge("stop_2", "stop_1", "stop_2", "stop_0", "DRIVE", 1850.0),
    ]
    return {
        "schema_version": "1.5",
        "places": places,
        "clusters": clusters,
        "hub_hub_legs": hub_hub_legs,
        "edges": edges,
        "geojson": {"type": "FeatureCollection", "features": []},
        "stats": {"place_count": 3, "cluster_count": 2, "edges_stored": 6},
    }


def test_resolve_place_id_exact_id() -> None:
    g = _tiny_graph()
    r = resolve_place_id(g, "stop_1")
    assert r.matched.place_id == "stop_1"
    assert r.matched.score == 1.0


def test_resolve_place_id_fuzzy_name() -> None:
    g = _tiny_graph()
    r = resolve_place_id(g, "rijks")
    assert r.matched.place_id == "stop_1"


def test_resolve_place_id_ambiguous_raises() -> None:
    g = _tiny_graph()
    # Make two similar names so "hotel" becomes ambiguous.
    g["places"][0]["name"] = "Hotel One"
    g["places"][1]["name"] = "Hotel Two"
    with pytest.raises(PlaceResolutionError):
        resolve_place_id(g, "hotel", min_score=0.1, ambiguity_delta=0.5)


def test_resolve_place_id_empty_query_raises() -> None:
    g = _tiny_graph()
    with pytest.raises(PlaceResolutionError, match="empty query"):
        resolve_place_id(g, "   ")


def test_resolve_place_id_includes_candidates_on_failure() -> None:
    g = _tiny_graph()
    with pytest.raises(PlaceResolutionError) as exc:
        resolve_place_id(g, "does-not-exist", min_score=0.99)
    assert exc.value.query == "does-not-exist"
    assert exc.value.candidates  # suggestions included


def test_distance_between_fastest_maps_to_duration() -> None:
    g = _tiny_graph()
    res_a = distance_between(g, "Hotel", "Rijksmuseum", preference="duration")
    res_b = distance_between(g, "Hotel", "Rijksmuseum", preference="fastest")
    assert res_a.option.duration_seconds == res_b.option.duration_seconds


def test_distance_between_shortest_maps_to_distance() -> None:
    g = _tiny_graph()
    res_a = distance_between(g, "Hotel", "Rijksmuseum", preference="distance")
    res_b = distance_between(g, "Hotel", "Rijksmuseum", preference="shortest")
    assert res_a.option.distance_meters == res_b.option.distance_meters


def test_closest_to_empty_candidates_raises() -> None:
    g = _tiny_graph()
    with pytest.raises(ValueError, match="candidate_names is empty"):
        closest_to(g, "Hotel", [])


def test_distance_between_uses_graph() -> None:
    g = _tiny_graph()
    res = distance_between(g, "Hotel", "Rijksmuseum", preference="duration")
    assert res.option.travel_mode == "WALK"
    assert res.option.duration_seconds == pytest.approx(240.0)


def test_closest_to_ranks_candidates() -> None:
    g = _tiny_graph()
    # Target is far-away gardens; hotel and museum are in same cluster and have identical DRIVE hub leg.
    # Tie-break is deterministic by matched name.
    out = closest_to(g, "Gardens", ["Rijksmuseum", "Hotel"], preference="duration")
    assert out.target.matched.place_id == "stop_2"
    assert len(out.ranked) == 2
    assert out.winner.candidate.matched.place_id in ("stop_0", "stop_1")


# ---------------------------------------------------------------------------
# Mock Execution Agent: retains typed artifacts like docs/workflow.md Phase 2
# ---------------------------------------------------------------------------


def _extract_place_graph(
    agent_artifacts: dict[str, list[AgentArtifactModel]],
) -> dict | None:
    """Same shape as execution code reading StateContractModel.agent_artifacts."""
    for art in agent_artifacts.get("routing_check_agent", []):
        if art.type == ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH and isinstance(art.content, dict):
            return art.content
    return None


def test_mock_execution_agent_distance_from_retained_artifact() -> None:
    """Simulate: Route Check ran → artifact retained → agent queries by human labels."""
    content = _tiny_graph()
    artifact = AgentArtifactModel(
        name="live-trip_place_distance_graph",
        type=ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
        content=content,
        description="3 places, 2 clusters",
    )
    agent_artifacts: dict[str, list[AgentArtifactModel]] = {
        "routing_check_agent": [artifact],
    }

    graph = _extract_place_graph(agent_artifacts)
    assert graph is not None
    assert graph["schema_version"] == "1.5"

    res = distance_between(graph, "Hotel", "Rijksmuseum", preference="duration")
    assert res.option.travel_mode == "WALK"
    assert res.option.duration_seconds == pytest.approx(240.0)
    assert "haversine_walk" in res.explanation


def test_mock_execution_agent_closest_of_candidates_to_target() -> None:
    content = _tiny_graph()
    agent_artifacts = {
        "routing_check_agent": [
            AgentArtifactModel(
                name="trip_graph",
                type=ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
                content=content,
                description="mock",
            )
        ]
    }
    graph = _extract_place_graph(agent_artifacts)
    assert graph is not None

    out = closest_to(
        graph,
        target_name="Keukenhof",
        candidate_names=["Hotel Central", "Rijksmuseum"],
        preference="duration",
    )
    # Both candidates are same cluster → same hub leg to Keukenhof; order tie-broken by name.
    assert out.target.matched.place_id == "stop_2"
    assert len(out.ranked) == 2
    assert {c.candidate.matched.place_id for c in out.ranked} == {"stop_0", "stop_1"}


def test_mock_execution_agent_resolve_then_use_ids() -> None:
    """Agent can mix fuzzy names and explicit stop ids from the same retained graph."""
    graph = _extract_place_graph(
        {
            "routing_check_agent": [
                AgentArtifactModel(
                    name="g",
                    type=ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
                    content=_tiny_graph(),
                    description="",
                )
            ]
        }
    )
    assert graph is not None
    r1 = resolve_place_id(graph, "rijks")
    r2 = resolve_place_id(graph, "stop_0")
    assert r1.matched.place_id == "stop_1"
    assert r2.matched.place_id == "stop_0"

    res = distance_between(graph, r2.matched.place_id, r1.matched.name, preference="fastest")
    assert res.option.duration_seconds == pytest.approx(240.0)

