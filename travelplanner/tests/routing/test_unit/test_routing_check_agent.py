"""Tests for the Routing Check agent (workflow Route Check → route_timing_artifact)."""

from __future__ import annotations

from unittest.mock import patch

import travelplanner.integrations.routing_check_agent as rca
from travelplanner.integrations.routing_check_agent import (
    ARTIFACT_TYPE_ROUTE_TIMING,
    ROUTING_CHECK_TASK_TYPE,
    RoutingCheckAgentState,
    make_graph,
)
from travelplanner.schema.system_state import AgentArtifactModel


def _mock_artifact() -> AgentArtifactModel:
    return AgentArtifactModel(
        name="t1_route_timing",
        type="route_timing_artifact",
        content={
            "request": {
                "origin": "A St",
                "destination": "B Ave",
                "travel_mode": "DRIVE",
            },
            "metrics": {
                "distance_meters": 1000,
                "distance_km": 1.0,
                "duration_seconds": 120.0,
            },
        },
        description="mocked route timing",
    )


def test_constants_match_workflow_task_type() -> None:
    assert ROUTING_CHECK_TASK_TYPE == "routing-check"
    assert ARTIFACT_TYPE_ROUTE_TIMING == "route_timing_artifact"


def test_execute_routing_returns_error_dict_when_api_key_missing() -> None:
    st = rca.RoutingCheckAgentState(
        decided_mode="single_od",
        origin_address="A Rd",
        destination_address="B Rd",
        api_key="",
    )
    with patch.object(rca.os, "getenv", return_value=""):
        out = rca._execute_routing(st)
    assert isinstance(out, dict)
    assert "error" in out
    assert "API key" in (out["error"] or "")


def test_execute_routing_returns_error_when_single_od_addresses_blank() -> None:
    out = rca._execute_routing(
        rca.RoutingCheckAgentState(
            decided_mode="single_od",
            origin_address=" ",
            destination_address="\t",
            api_key="k",
        ),
    )
    assert out.get("error")
    assert "origin_address" in (out["error"] or "").lower()


def test_invoke_graph_surfaces_execute_error_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    out = make_graph().invoke(
        RoutingCheckAgentState(
            task_ref="t1",
            origin_address="A St",
            destination_address="B Ave",
            travel_mode="drive",
            api_key="",
        )
    )
    assert out["error"]
    assert "API key" in out["error"]


def test_make_graph_produces_route_timing_artifact() -> None:
    state = RoutingCheckAgentState(
        task_ref="t1",
        origin_address="A St",
        destination_address="B Ave",
        travel_mode="drive",
        api_key="test-key",
    )
    with patch(
        "travelplanner.integrations.routing_check_agent.execute_routing_check_task",
        return_value=_mock_artifact(),
    ):
        out = make_graph().invoke(state)
    assert out["artifact"] is not None
    assert out["artifact"].type == ARTIFACT_TYPE_ROUTE_TIMING
    assert out["artifact"].name == "t1_route_timing"
    assert out["message_history"] is not None
    assert (
        out["message_history"].agent_ref
        == "travelplanner.integrations.routing_check_agent"
    )
