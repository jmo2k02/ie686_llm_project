"""Characterization tests for the subagent tool registry.

These tests lock the **expected state** of the registry after the refactor:
tools that should exist (search_web, routing tools) and docs that must be
aligned. Currently they FAIL because the tools are not yet registered.

Wave 1 / Task 1 — Establishment baseline for
.plan `refactor-websearch-routing-tools.md`.
"""

from __future__ import annotations

import re
import unittest
from unittest.mock import patch
from typing import Set

from travelplanner.agents.tools import make_subagent_tools
from travelplanner.agents.execution.prompts import _SUBAGENT_TOOLS_DOCS


class TestSubagentToolRegistryCompleteness(unittest.TestCase):
    """Test 1: make_subagent_tools() must return exactly nine named tools."""

    def test_nine_named_tools_returned(self) -> None:
        """Registry must expose exactly 9 tools after refactor."""
        tools = make_subagent_tools()
        names = {t.name for t in tools}
        expected: Set[str] = {
            "search_flights",
            "search_hotels",
            "search_restaurants",
            "search_attractions",  # plan: resolve drift vs execution/graph
            "search_web",           # plan: NEW — general_web_search adapter
            "check_route_timing",    # plan: NEW — routing check
            "build_place_distance_graph",  # plan: NEW — routing graph
            "distance_between_places",     # plan: NEW — routing lookup
            "closest_places_to_target",    # plan: NEW — routing lookup
        }
        missing = expected - names
        self.assertEqual(
            len(names),
            9,
            f"Registry should have 9 tools but has {len(names)}. "
            f"Got names: {sorted(names)}",
        )
        self.assertFalse(
            missing,
            f"Missing tools from registry: {sorted(missing)}",
        )

    def test_all_current_tools_still_present(self) -> None:
        """Regression: already-wired tools must remain registered."""
        tools = make_subagent_tools()
        names = {t.name for t in tools}
        currently_wired = {
            "search_flights",
            "search_hotels",
            "search_restaurants",
        }
        missing_current = currently_wired - names
        self.assertFalse(
            missing_current,
            f"Regression — previously wired tools missing: {sorted(missing_current)}",
        )


class TestSubagentToolsDocsAlignment(unittest.TestCase):
    """Test 2: _SUBAGENT_TOOLS_DOCS alignment with registered tools.

    Every documented tool must be registered.
    Every registered tool must be documented.
    The placeholder comment '## Hier eure Tools einfuegen!!' must be gone.
    """

    def test_placeholder_comment_removed(self) -> None:
        """Placeholder German comment must be replaced by real tool docs."""
        self.assertNotIn(
            "Hier eure Tools einfuegen",
            _SUBAGENT_TOOLS_DOCS,
            "Placeholder comment still present in _SUBAGENT_TOOLS_DOCS",
        )
        self.assertNotIn(
            "## Hier",
            _SUBAGENT_TOOLS_DOCS,
            "Block comment still present in _SUBAGENT_TOOLS_DOCS",
        )

    def _extract_documented_tool_names(self) -> set[str]:
        """Parse tool names from _SUBAGENT_TOOLS_DOCS bullet format.

        Each bullet entry starts with ``- `tool_name(...)`` but may span multiple
        lines (e.g. search_hotels description continues on next line).
        We split on '- `' boundaries then scan each entry for a tool call pattern.
        """
        bullet_entries = _SUBAGENT_TOOLS_DOCS.split("- `")
        tool_names: set[str] = set()
        for entry in bullet_entries[1:]:
            m = re.match(r"(\w+)\(", entry)
            if m:
                tool_names.add(m.group(1))
        return tool_names

    def test_every_registered_tool_is_documented(self) -> None:
        """No registered tool may be undocumented (docs drift = silent model failures)."""
        tools = make_subagent_tools()
        registered = {t.name for t in tools}
        documented = self._extract_documented_tool_names()
        undocumented = registered - documented
        self.assertFalse(
            undocumented,
            f"Undocumented registered tools (model will not call them reliably): "
            f"{sorted(undocumented)}",
        )

    def test_every_documented_tool_is_registered(self) -> None:
        """No documented tool may be missing from registry (model calls ghost = error)."""
        tools = make_subagent_tools()
        registered = {t.name for t in tools}
        documented = self._extract_documented_tool_names()
        ghost_documented = documented - registered
        self.assertFalse(
            ghost_documented,
            f"Ghost-documented tools (don't exist in registry): {sorted(ghost_documented)}",
        )

    def test_search_web_documented(self) -> None:
        """search_web must appear in prompt docs so the model knows to call it."""
        names = self._extract_documented_tool_names()
        self.assertIn(
            "search_web",
            names,
            "search_web is not documented — model will not invoke it",
        )

    def test_routing_tools_documented(self) -> None:
        """All four routing tools must appear in prompt docs."""
        names = self._extract_documented_tool_names()
        routing = {
            "check_route_timing",
            "build_place_distance_graph",
            "distance_between_places",
            "closest_places_to_target",
        }
        undocumented = routing - names
        self.assertFalse(
            undocumented,
            f"Routing tools undocumented: {sorted(undocumented)}",
        )


class TestSubagentToolArgsSchemas(unittest.TestCase):
    """Test 3: Every registered tool must have a non-None Pydantic args_schema."""

    def _get_tool(self, name: str):
        tools = make_subagent_tools()
        for t in tools:
            if t.name == name:
                return t
        return None

    def test_search_flights_has_schema(self) -> None:
        t = self._get_tool("search_flights")
        self.assertIsNotNone(t, "search_flights not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "search_flights args_schema is None")

    def test_search_hotels_has_schema(self) -> None:
        t = self._get_tool("search_hotels")
        self.assertIsNotNone(t, "search_hotels not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "search_hotels args_schema is None")

    def test_search_restaurants_has_schema(self) -> None:
        t = self._get_tool("search_restaurants")
        self.assertIsNotNone(t, "search_restaurants not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "search_restaurants args_schema is None")

    def test_search_attractions_has_schema(self) -> None:
        t = self._get_tool("search_attractions")
        self.assertIsNotNone(t, "search_attractions not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "search_attractions args_schema is None")

    def test_search_web_has_schema(self) -> None:
        t = self._get_tool("search_web")
        self.assertIsNotNone(t, "search_web not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "search_web args_schema is None")

    def test_check_route_timing_has_schema(self) -> None:
        t = self._get_tool("check_route_timing")
        self.assertIsNotNone(t, "check_route_timing not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "check_route_timing args_schema is None")

    def test_build_place_distance_graph_has_schema(self) -> None:
        t = self._get_tool("build_place_distance_graph")
        self.assertIsNotNone(t, "build_place_distance_graph not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "build_place_distance_graph args_schema is None")

    def test_distance_between_places_has_schema(self) -> None:
        t = self._get_tool("distance_between_places")
        self.assertIsNotNone(t, "distance_between_places not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "distance_between_places args_schema is None")

    def test_closest_places_to_target_has_schema(self) -> None:
        t = self._get_tool("closest_places_to_target")
        self.assertIsNotNone(t, "closest_places_to_target not in registry")
        schema = getattr(t, "args_schema", None)
        self.assertIsNotNone(schema, "closest_places_to_target args_schema is None")


class TestSubagentToolsGraphWiring(unittest.TestCase):
    """Test 4: make_subagent_tools() result is compatible with make_graph()."""

    def test_tools_splat_into_make_graph_without_type_error(self) -> None:
        """tools=[*make_subagent_tools(), *make_travelplan_tools(plan)] must not raise."""
        from unittest.mock import patch, MagicMock

        from travelplanner.agents.execution.graph import make_graph
        from travelplanner.travelplan import TravelPlan

        plan = TravelPlan()
        with patch(
            "travelplanner.agents.execution.graph.make_chat_model",
            return_value=MagicMock(),
        ):
            with patch(
                "travelplanner.agents.execution.graph.create_deep_agent",
                return_value=MagicMock(),
            ):
                graph = make_graph(plan)
                self.assertIsNotNone(graph)

    def test_make_subagent_tools_returns_list_of_base_tool(self) -> None:
        """Return type must be list[BaseTool] so LangGraph accepts it."""
        from langchain_core.tools import BaseTool

        tools = make_subagent_tools()
        self.assertIsInstance(tools, list)
        for t in tools:
            self.assertIsInstance(t, BaseTool, f"{t.name} is not a BaseTool")

    def test_all_nine_tools_individually_wirable(self) -> None:
        """Each tool must be a StructuredTool with a bindable function."""
        tools = make_subagent_tools()
        by_name = {t.name: t for t in tools}
        for name in [
            "search_flights", "search_hotels", "search_restaurants",
            "search_attractions", "search_web", "check_route_timing",
            "build_place_distance_graph", "distance_between_places",
            "closest_places_to_target",
        ]:
            self.assertIn(name, by_name, f"{name} not in registry")
            t = by_name[name]
            self.assertTrue(
                callable(t.func),
                f"{name}.func is not callable",
            )


class TestSubagentToolModelSelection(unittest.TestCase):
    """Regression coverage for per-tool model routing."""

    def test_search_restaurants_uses_task_planning_model_from_config(self) -> None:
        captured: dict[str, str] = {}

        def fake_restaurant_factory(model_name: str, temperature: float, task_ref: str):
            captured["model_name"] = model_name
            return lambda query: query

        def fake_get_setting(path: str, default=None):
            if path == "models.workflows.task_planning.model_name":
                return "openrouter:test/task-planning"
            if path == "models.agents.flight_search.model_name":
                return "openrouter:test/flight-search"
            return default

        with patch(
            "travelplanner.agents.tools.get_setting",
            side_effect=fake_get_setting,
        ), patch(
            "travelplanner.agents.tools.make_search_restaurants_tool",
            side_effect=fake_restaurant_factory,
        ):
            make_subagent_tools()

        self.assertEqual(
            captured["model_name"],
            "openrouter:test/task-planning",
        )


if __name__ == "__main__":
    unittest.main()
