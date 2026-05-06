"""Unit tests for the LLM-assisted place-distance-graph routing agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import travelplanner.agents.routing_agent as ra


def _graph_build_mock():
    gm = MagicMock()
    gm.stats.place_count = 1
    gm.stats.cluster_count = 1
    gm.stats.edges_stored = 2
    gm.model_dump.return_value = {"places": [{"id": "p1"}]}
    return gm


def test_build_routing_graph_empty_stops_returns_error():
    out = ra.build_routing_graph().invoke(
        ra.RoutingAgentState(stops=[], api_key="test-key").model_dump(mode="json")
    )
    assert out["artifact"] is None
    assert out["error"] == "No input data"


def test_explicit_cluster_skips_llm_invoke():
    with patch.object(ra, "invoke_structured_model") as invoke_llm, patch.object(
        ra,
        "build_place_distance_graph",
        return_value=_graph_build_mock(),
    ), patch.object(
        ra, "parse_places_input_payload", return_value=(None, ["place"], None)
    ), patch.object(ra, "place_distance_graph_config_for_context"):
        out = ra.build_routing_graph().invoke(
            {
                "stops": [{"address": "Somewhere", "name": "A"}],
                "cluster_context": "dense_urban",
                "api_key": "k",
                "model_name": "m",
                "temperature": 0.0,
            }
        )
    invoke_llm.assert_not_called()
    assert out.get("error") is None
    assert out["artifact"] is not None
    assert out["artifact"].type == ra.ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH


def test_llm_fallback_invokes_when_cluster_unknown():
    with patch.object(
        ra,
        "invoke_structured_model",
        return_value=(
            ra.ClusterPresetResponse(cluster_context="mixed"),
            "up",
            "{}",
        ),
    ) as invoke_llm, patch.object(
        ra,
        "build_place_distance_graph",
        return_value=_graph_build_mock(),
    ), patch.object(
        ra,
        "parse_places_input_payload",
        return_value=(None, ["place"], None),
    ), patch.object(ra, "place_distance_graph_config_for_context"):
        out = ra.build_routing_graph().invoke(
            {
                "stops": [{"address": "Somewhere", "name": "A"}],
                "cluster_context": None,
                "api_key": "k",
                "model_name": "m",
                "temperature": 0.0,
            }
        )
    invoke_llm.assert_called_once()
    assert out.get("error") is None
    assert out["decided_cluster_context"] == "mixed"


@pytest.mark.parametrize("cluster", ["dense_urban", "mixed", "sparse"])
def test_run_routing_agent_returns_artifact_when_mocked(cluster: str):
    with patch.object(
        ra,
        "build_place_distance_graph",
        return_value=_graph_build_mock(),
    ), patch.object(
        ra,
        "parse_places_input_payload",
        return_value=(None, ["place"], None),
    ), patch.object(ra, "place_distance_graph_config_for_context"):
        art = ra.run_routing_agent(
            [{"address": "X", "name": "n"}],
            cluster_context=cluster,
            api_key="k",
        )
    assert art.content["places"][0]["id"] == "p1"


def test_run_routing_agent_raises_on_missing_api_for_nonempty_stops():
    with patch.object(ra.os, "getenv", return_value=""):
        with pytest.raises(RuntimeError, match="Missing Google Maps API key"):
            ra.run_routing_agent(
                [{"address": "X"}], cluster_context="mixed", api_key=""
            )


def test_build_graph_returns_error_when_parse_produces_no_places():
    with patch.object(
        ra,
        "parse_places_input_payload",
        return_value=(None, [], None),
    ), patch.object(ra, "invoke_structured_model") as invoke_llm:
        out = ra.build_routing_graph().invoke(
            {
                "stops": [{"address": "x", "name": "n"}],
                "cluster_context": "mixed",
                "api_key": "k",
            }
        )
    invoke_llm.assert_not_called()
    assert not out.get("artifact")
    assert out.get("error")
    assert "usable places" in (out["error"] or "").lower()


def test_run_routing_graph_result_reports_failure_without_raising():
    out = ra.run_routing_graph_result([], cluster_context="mixed", api_key="k")
    assert out["ok"] is False
    assert out.get("error")


def test_run_routing_graph_result_ok_when_mocked():
    with patch.object(
        ra,
        "build_place_distance_graph",
        return_value=_graph_build_mock(),
    ), patch.object(
        ra,
        "parse_places_input_payload",
        return_value=(None, ["place"], None),
    ), patch.object(ra, "place_distance_graph_config_for_context"):
        out = ra.run_routing_graph_result(
            [{"address": "Z", "name": "n"}],
            cluster_context="dense_urban",
            api_key="k",
        )
    assert out["ok"] is True
    assert out["artifact"]["type"] == ra.ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH