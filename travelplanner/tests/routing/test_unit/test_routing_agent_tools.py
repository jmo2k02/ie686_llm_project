"""Tests for agent-facing routing helpers (:mod:`routing_agent_tools`)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from travelplanner.integrations import routing_agent_tools as tools
from tests.routing.test_unit.test_routing_lookup_queries import _tiny_graph


def test_route_one_leg_validates_addresses_before_api():
    r = tools.route_one_leg(origin_address="  ", destination_address="B")
    assert r["ok"] is False
    assert r["stage"] == "validate_input"


def test_route_one_leg_requires_api_key():
    with patch.object(tools.os, "getenv", return_value=""):
        r = tools.route_one_leg(
            origin_address="A St", destination_address="B Ave", api_key=""
        )
    assert r["ok"] is False
    assert r["stage"] == "api_key"


def test_route_one_leg_success_mocked():
    fake = MagicMock()
    fake.content = {"metrics": {"distance_km": 2.5, "duration_seconds": 600}}
    fake.description = "DRIVE: 2.5 km"
    fake.name = "t"
    fake.type = "route_timing_artifact"
    with patch(
        "travelplanner.integrations.routing_agent_tools.execute_routing_check_task",
        return_value=fake,
    ):
        r = tools.route_one_leg(
            origin_address="A St",
            destination_address="B Ave",
            api_key="k",
        )
    assert r["ok"] is True
    assert r["metrics"]["distance_km"] == 2.5


def test_build_distance_graph_from_stops_empty_stops():
    r = tools.build_distance_graph_from_stops([])
    assert r["ok"] is False
    assert r["stage"] == "validate_input"


def test_build_distance_graph_from_stops_parse_yields_no_places():
    with patch.object(
        tools,
        "parse_places_input_payload",
        return_value=(None, [], None),
    ), patch.object(tools.os, "getenv", return_value="k"):
        r = tools.build_distance_graph_from_stops([{"name": "only"}])
    assert r["ok"] is False
    assert r["stage"] == "parse_stops"


def test_distance_between_places_ok():
    g = _tiny_graph()
    r = tools.distance_between_places(g, "Hotel", "Rijks")
    assert r["ok"] is True
    assert "summary" in r
    assert r["from_place_id"] == "stop_0"
    assert r["to_place_id"] == "stop_1"


def test_distance_between_places_resolve_failure():
    r = tools.distance_between_places(_tiny_graph(), "", "x")
    assert r["ok"] is False
    assert r["stage"] == "validate_input"


def test_orchestrator_routing_tool_schemas_has_three_tools():
    assert len(tools.ORCHESTRATOR_ROUTING_TOOL_SCHEMAS) == 3
    names = {s["function"]["name"] for s in tools.ORCHESTRATOR_ROUTING_TOOL_SCHEMAS}
    assert names == {
        "build_place_graph_with_routing_agent",
        "distance_between_places",
        "closest_places_to_target",
    }


def test_routing_tool_schemas_superset():
    assert len(tools.ROUTING_TOOL_SCHEMAS) == len(tools.ORCHESTRATOR_ROUTING_TOOL_SCHEMAS) + 2


def test_build_place_graph_with_routing_agent_wraps_run_result():
    fake_graph = {"places": [{"id": "stop_0"}], "stats": {"place_count": 1}}
    with patch(
        "travelplanner.integrations.routing_agent_tools.run_routing_graph_result",
        return_value={
            "ok": True,
            "artifact": {
                "type": "place_distance_graph",
                "description": "1 places",
                "content": fake_graph,
            },
            "decided_cluster_context": "mixed",
        },
    ):
        r = tools.build_place_graph_with_routing_agent(
            [{"address": "X", "name": "a"}], api_key="k"
        )
    assert r["ok"] is True
    assert r["graph"] == fake_graph
    assert r["decided_cluster_context"] == "mixed"


def test_build_place_graph_with_routing_agent_empty_stops():
    r = tools.build_place_graph_with_routing_agent([])
    assert r["ok"] is False
    assert r["stage"] == "validate_input"


def test_closest_places_to_target_ok():
    g = _tiny_graph()
    r = tools.closest_places_to_target(
        g, "Hotel Central", ["Rijksmuseum", "Keukenhof Gardens"]
    )
    assert r["ok"] is True
    assert r["winner"]["place_id"] == "stop_1"


def test_closest_places_to_target_empty_candidates():
    r = tools.closest_places_to_target(_tiny_graph(), "Hotel", [])
    assert r["ok"] is False
