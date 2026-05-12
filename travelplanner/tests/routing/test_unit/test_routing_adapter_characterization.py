"""Characterization tests for the routing-check adapter gap.

These tests prove the routing tools do not yet wire into
`make_subagent_tools()` and cannot be called by the main execution agent.
They lock the desired post-refactor behavior:

- `check_route_timing` wraps
  ``routing_agent_tools.route_one_leg``
- `build_place_distance_graph` wraps
  ``routing_agent_tools.build_place_graph_with_routing_agent``
- `distance_between_places` wraps
  ``routing_agent_tools.distance_between_places``
- `closest_places_to_target` wraps
  ``routing_agent_tools.closest_places_to_target``

The source mapping is fixed by the plan; these tests verify the
registration layer is complete.

Wave 1 / Task 1 — Establishment baseline for
.plan `refactor-websearch-routing-tools.md`.
"""

from __future__ import annotations

import unittest
import unittest.mock
from typing import Any, Callable


class TestRoutingAdapterModuleExists(unittest.TestCase):
    """Routing-check adapter module must exist in subagent_tools/."""

    def test_routing_check_adapter_module_importable(self) -> None:
        """Module must be importable from subagent_tools."""
        import travelplanner.agents.subagent_tools.routing_check as rc

        self.assertTrue(
            hasattr(rc, "check_route_timing"),
            "routing_check adapter missing check_route_timing",
        )
        self.assertTrue(
            hasattr(rc, "build_place_distance_graph"),
            "routing_check adapter missing build_place_distance_graph",
        )
        self.assertTrue(
            hasattr(rc, "distance_between_places"),
            "routing_check adapter missing distance_between_places",
        )
        self.assertTrue(
            hasattr(rc, "closest_places_to_target"),
            "routing_check adapter missing closest_places_to_target",
        )


class TestRoutingAdapterFactories(unittest.TestCase):
    """Each routing tool must expose a make_*_tool factory callable."""

    def test_check_route_timing_factory(self) -> None:
        """Factory must return a single-arg callable: (query: str) -> str."""
        from travelplanner.agents.subagent_tools.routing_check import (
            make_check_route_timing_tool,
        )

        factory = make_check_route_timing_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test",
        )
        self.assertTrue(
            callable(factory),
            "make_check_route_timing_tool did not return a callable",
        )

    def test_build_place_distance_graph_factory(self) -> None:
        """Factory must return a callable accepting stops list."""
        from travelplanner.agents.subagent_tools.routing_check import (
            make_build_place_distance_graph_tool,
        )

        factory = make_build_place_distance_graph_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test",
        )
        self.assertTrue(
            callable(factory),
            "make_build_place_distance_graph_tool did not return a callable",
        )

    def test_distance_between_places_factory(self) -> None:
        from travelplanner.agents.subagent_tools.routing_check import (
            make_distance_between_places_tool,
        )

        factory = make_distance_between_places_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test",
        )
        self.assertTrue(
            callable(factory),
            "make_distance_between_places_tool did not return a callable",
        )

    def test_closest_places_to_target_factory(self) -> None:
        from travelplanner.agents.subagent_tools.routing_check import (
            make_closest_places_to_target_tool,
        )

        factory = make_closest_places_to_target_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test",
        )
        self.assertTrue(
            callable(factory),
            "make_closest_places_to_target_tool did not return a callable",
        )


class TestRoutingToolArgsSchemas(unittest.TestCase):
    """Each routing tool must have a Pydantic schema in tool_args.py."""

    def test_check_route_timing_schema(self) -> None:
        from travelplanner.agents import tool_args

        self.assertTrue(
            hasattr(tool_args, "CheckRouteTimingArgs"),
            "CheckRouteTimingArgs missing from tool_args",
        )

    def test_build_place_distance_graph_schema(self) -> None:
        from travelplanner.agents import tool_args

        self.assertTrue(
            hasattr(tool_args, "BuildPlaceDistanceGraphArgs"),
            "BuildPlaceDistanceGraphArgs missing from tool_args",
        )

    def test_distance_between_places_schema(self) -> None:
        from travelplanner.agents import tool_args

        self.assertTrue(
            hasattr(tool_args, "DistanceBetweenPlacesArgs"),
            "DistanceBetweenPlacesArgs missing from tool_args",
        )

    def test_closest_places_to_target_schema(self) -> None:
        from travelplanner.agents import tool_args

        self.assertTrue(
            hasattr(tool_args, "ClosestPlacesToTargetArgs"),
            "ClosestPlacesToTargetArgs missing from tool_args",
        )


class TestRoutingCheckRouteTimingContract(unittest.TestCase):
    """check_route_timing adapter: ok/stage/error contract on success and failure."""

    def test_returns_ok_on_successful_route_check(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "route_one_leg",
            return_value={"ok": True, "stage": "done", "distance_km": 14.2, "duration_min": 22},
        ):
            result = rc.check_route_timing(
                origin_address="A St",
                destination_address="B Ave",
                api_key="fake_key",
            )
        self.assertTrue(result.get("ok"), f"Expected ok=True, got: {result}")
        self.assertEqual(result["stage"], "done")
        self.assertIn("distance_km", result)

    def test_returns_error_when_api_key_missing(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(rc, "route_one_leg") as mock_route:
            result = rc.check_route_timing(
                origin_address="A St",
                destination_address="B Ave",
            )
        mock_route.assert_not_called()
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "api_key")
        self.assertIn("error", result)

    def test_returns_error_when_route_one_leg_fails(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "route_one_leg",
            return_value={"ok": False, "stage": "api_error", "error": "quota exceeded"},
        ):
            result = rc.check_route_timing(
                origin_address="A St",
                destination_address="B Ave",
                api_key="real_key",
            )
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "api_error")
        self.assertIn("quota exceeded", result["error"])


class TestRoutingBuildPlaceDistanceGraphContract(unittest.TestCase):
    """build_place_distance_graph adapter: ok/stage/error contract on success/failure."""

    def test_returns_ok_on_valid_stops(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "build_place_graph_with_routing_agent",
            return_value={
                "ok": True,
                "graph": {"places": {}, "distances": {}},
                "decided_cluster_context": "mixed",
            },
        ):
            result = rc.build_place_distance_graph(
                stops=[{"name": "Hotel", "address": "H"}, {"name": "Rijks", "address": "R"}],
                api_key="fake_key",
            )
        self.assertTrue(result.get("ok"), f"Expected ok=True, got: {result}")
        self.assertIn("graph", result)

    def test_returns_error_when_stops_empty(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        result = rc.build_place_distance_graph(stops=[], api_key="fake_key")
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "validate_input")

    def test_returns_error_when_api_key_missing(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(rc, "build_place_graph_with_routing_agent") as mock:
            result = rc.build_place_distance_graph(
                stops=[{"name": "Hotel", "address": "H"}],
            )
        mock.assert_not_called()
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "api_key")


class TestRoutingDistanceBetweenPlacesContract(unittest.TestCase):
    """distance_between_places adapter: ok/stage contract."""

    def test_passes_args_to_underlying_function(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "_distance",
            return_value={"ok": True, "from_place_id": "p0", "to_place_id": "p1", "distance_km": 5.0},
        ) as mock_dist:
            result = rc.distance_between_places(
                graph={"places": {}},
                from_place_id="p0",
                to_place_id="p1",
            )
        mock_dist.assert_called_once()
        self.assertTrue(result.get("ok"))

    def test_returns_error_when_underlying_fails(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "_distance",
            return_value={"ok": False, "stage": "place_not_found", "error": "unknown id"},
        ):
            result = rc.distance_between_places(
                graph={},
                from_place_id="unknown",
                to_place_id="other",
            )
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "place_not_found")


class TestRoutingClosestPlacesToTargetContract(unittest.TestCase):
    """closest_places_to_target adapter: ok/stage contract."""

    def test_passes_args_to_underlying_function(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "_closest",
            return_value={
                "ok": True,
                "winner": {"place_id": "p1", "name": "Rijks", "address": "Museumplein 6"},
                "distance_km": 3.2,
            },
        ) as mock_closest:
            result = rc.closest_places_to_target(
                graph={"places": {}},
                target_name="Rijks",
                candidate_names=["Rijks", "Van Gogh"],
            )
        mock_closest.assert_called_once()
        self.assertTrue(result.get("ok"))

    def test_returns_error_when_underlying_fails(self) -> None:
        import travelplanner.agents.subagent_tools.routing_check as rc

        with unittest.mock.patch.object(
            rc,
            "_closest",
            return_value={"ok": False, "stage": "validate_input", "error": "target not found"},
        ):
            result = rc.closest_places_to_target(
                graph={},
                target_name="Unknown",
                candidate_names=["A", "B"],
            )
        self.assertFalse(result.get("ok"))
        self.assertEqual(result["stage"], "validate_input")


if __name__ == "__main__":
    unittest.main()