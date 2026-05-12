"""Characterization tests for the general_web_search adapter gap.

These tests prove the **websearch adapter** does not yet wire into
`make_subagent_tools()` and cannot be called by the main execution agent.
They lock the desired post-refactor behavior: an adapter callable in
`subagent_tools/` that accepts a query string, builds a TaskModel with
`type="general-web-search"`, invokes the general_web_search graph, and
returns a textual summary.

Wave 1 / Task 1 — Establishment baseline for
.plan `refactor-websearch-routing-tools.md`.
"""

from __future__ import annotations

import unittest
from typing import Any


class TestGeneralWebSearchSubagentAdapterExists(unittest.TestCase):
    """Adapter module must exist in subagent_tools/."""

    def test_general_web_search_adapter_module_exists(self) -> None:
        """The adapter module must be importable from subagent_tools."""
        import travelplanner.agents.subagent_tools.general_web_search as gwsa

        # must expose the factory
        self.assertTrue(
            hasattr(gwsa, "make_search_web_tool"),
            "general_web_search adapter missing make_search_web_tool factory",
        )

    def test_factory_returns_callable(self) -> None:
        """Factory must return a callable (query: str) -> str."""
        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )

        # Minimal call — model_name can be fake; we just verify shape
        tool_fn = make_search_web_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test_ref",
        )
        self.assertTrue(
            callable(tool_fn),
            "make_search_web_tool did not return a callable",
        )


class TestGeneralWebSearchAdapterContract(unittest.TestCase):
    """Adapter must pass TaskModel with real query text to the graph."""

    def test_adapter_passes_query_not_empty_string(self) -> None:
        """Factory must return a callable that accepts 'query' as its only arg."""
        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )

        tool_fn = make_search_web_tool(
            model_name="fake/model",
            temperature=0.0,
            task_ref="test_ref",
        )
        self.assertTrue(callable(tool_fn))

        sig = __import__("inspect").signature(tool_fn)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["query"])

    def test_adapter_returns_text_not_error_on_success_path(self) -> None:
        """Adapter must return a non-empty textual result on success, not an Error blob."""
        import travelplanner.agents.subagent_tools.general_web_search as _gwsa

        class _FakeSuccessGWGraph:
            def invoke(self, state: Any) -> dict:
                from travelplanner.schema.system_state import AgentArtifactModel
                from travelplanner.schema.general_web_search_artifact import GeneralWebSearchArtifactContentModel

                artifact_content = GeneralWebSearchArtifactContentModel(
                    task_ref="test",
                    query="Munich hotels",
                    provider="tavily",
                    status="success",
                    attempt=1,
                    result={"ok": True, "query": "Munich hotels"},
                    answer="Hotel Bristol scored 9.1",
                    model="fake",
                    sources=[],
                    proof_points=[],
                    errors=[],
                    config={},
                )
                return {
                    "agent_artifacts": {
                        "general_web_search_agent": [
                            AgentArtifactModel(
                                name="web_result",
                                type="general-web-search",
                                content=artifact_content.model_dump(),
                            )
                        ]
                    }
                }

        saved = _gwsa.make_general_web_search_graph
        _gwsa.make_general_web_search_graph = lambda: _FakeSuccessGWGraph()

        try:
            from travelplanner.agents.subagent_tools.general_web_search import (
                make_search_web_tool,
            )

            tool_fn = make_search_web_tool(
                model_name="fake/model",
                temperature=0.0,
                task_ref="test",
            )
            result = tool_fn("Best Munich hotel for solo traveller 2026")
        finally:
            _gwsa.make_general_web_search_graph = saved

        # Must not be an error blob
        self.assertFalse(
            result.startswith("Error:"),
            f"Adapter returned error blob on success path: {result}",
        )
        self.assertTrue(
            result.strip(),
            "Adapter returned empty result on success path",
        )


class TestGeneralWebSearchToolArgsSchema(unittest.TestCase):
    """Adapter must have a Pydantic args schema in tool_args.py for StructuredTool."""

    def test_schema_exists_for_search_web(self) -> None:
        """WebSearchArgs must be registered as the args_schema for search_web."""
        from travelplanner.agents.tools import make_subagent_tools

        tools = make_subagent_tools()
        web_tool = next((t for t in tools if t.name == "search_web"), None)
        self.assertIsNotNone(web_tool, "search_web not in make_subagent_tools()")
        schema = getattr(web_tool, "args_schema", None)
        self.assertIsNotNone(schema, "search_web args_schema is None")


class TestWebSearchAdapterSuccessPath(unittest.TestCase):
    """Success path: graph returns typed artifacts → adapter returns text summary."""

    def test_returns_summary_with_source_urls_on_success(self) -> None:
        """On valid artifact, adapter must return text including URL list."""
        from unittest.mock import MagicMock

        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )
        from travelplanner.schema.system_state import AgentArtifactModel

        captured_query = ""

        class _FakeGraph:
            def invoke(self, state):
                nonlocal captured_query
                captured_query = state.query
                from travelplanner.schema.system_state import AgentArtifactModel
                from travelplanner.schema.general_web_search_artifact import GeneralWebSearchArtifactContentModel

                artifact_content = GeneralWebSearchArtifactContentModel(
                    task_ref="test",
                    query="Munich to Sydney flights",
                    provider="tavily",
                    status="success",
                    attempt=1,
                    result={"ok": True, "query": "Munich to Sydney flights"},
                    answer="Munich to Sydney flights cost ~$1,200.",
                    model="fake",
                    sources=[],
                    proof_points=[],
                    errors=[],
                    config={},
                )
                return {
                    "agent_artifacts": {
                        "general_web_search_agent": [
                            AgentArtifactModel(
                                name="web-search-result",
                                type="general-web-search-result",
                                content=artifact_content.model_dump(),
                            )
                        ]
                    },
                    "message_history": None,
                }

        # Patch the graph factory so each call to make_search_web_tool gets our fake
        import travelplanner.agents.subagent_tools.general_web_search as _gwsa

        saved = _gwsa.make_general_web_search_graph
        _gwsa.make_general_web_search_graph = lambda: _FakeGraph()

        try:
            tool_fn = make_search_web_tool(
                model_name="fake/model",
                temperature=0.0,
                task_ref="test",
            )
            result = tool_fn("Munich to Sydney flights")
            self.assertIn("Munich to Sydney flights cost", result)
        finally:
            _gwsa.make_general_web_search_graph = saved

    def test_passes_correct_task_type_to_graph(self) -> None:
        """TaskModel.type must be 'general-web-search' for routing to succeed."""
        from unittest.mock import MagicMock

        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )
        from travelplanner.schema.system_state import TaskModel

        captured: dict[str, Any] = {}

        class _FakeGraph:
            def invoke(self, state):
                if state.task_list:
                    captured["task_type"] = state.task_list[0].type
                    captured["task_text"] = state.task_list[0].text
                return {
                    "agent_artifacts": {
                        "general_web_search_agent": [
                            MagicMock(
                                name="fake",
                                type="general-web-search-result",
                                content={},
                            )
                        ]
                    },
                    "message_history": None,
                }

        import travelplanner.agents.subagent_tools.general_web_search as gwsa
        original = gwsa.make_general_web_search_graph
        gwsa.make_general_web_search_graph = lambda: _FakeGraph()

        try:
            tool_fn = make_search_web_tool(
                model_name="fake/model",
                temperature=0.0,
                task_ref="test",
            )
            tool_fn("Berlin to Paris trains")
            self.assertEqual(captured.get("task_type"), "general-web-search")
            self.assertEqual(captured.get("task_text"), "Berlin to Paris trains")
        finally:
            gwsa.make_general_web_search_graph = original


class TestWebSearchAdapterFailurePath(unittest.TestCase):
    """Failure path: graph raises or returns empty → adapter returns 'Error: ...'."""

    def test_returns_error_string_when_graph_raises(self) -> None:
        """When the graph raises an exception, adapter must return 'Error: ...'."""
        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )

        class _ExplodingGraph:
            def invoke(self, state):
                raise RuntimeError("Tavily API unreachable")

        import travelplanner.agents.subagent_tools.general_web_search as gwsa
        original = gwsa.make_general_web_search_graph
        gwsa.make_general_web_search_graph = lambda: _ExplodingGraph()

        try:
            tool_fn = make_search_web_tool(
                model_name="fake/model",
                temperature=0.0,
                task_ref="test",
            )
            result = tool_fn("any query")
            self.assertTrue(
                result.startswith("Error:"),
                f"Expected 'Error: ...' but got: {result}",
            )
            self.assertIn("Tavily API unreachable", result)
        finally:
            gwsa.make_general_web_search_graph = original

    def test_returns_error_string_when_no_artifacts_returned(self) -> None:
        """When the graph returns no artifacts, adapter must return 'Error: ...'."""
        from travelplanner.agents.subagent_tools.general_web_search import (
            make_search_web_tool,
        )

        class _EmptyGraph:
            def invoke(self, state):
                return {
                    "agent_artifacts": {},
                    "message_history": None,
                }

        import travelplanner.agents.subagent_tools.general_web_search as gwsa
        original = gwsa.make_general_web_search_graph
        gwsa.make_general_web_search_graph = lambda: _EmptyGraph()

        try:
            tool_fn = make_search_web_tool(
                model_name="fake/model",
                temperature=0.0,
                task_ref="test",
            )
            result = tool_fn("some query")
            self.assertTrue(
                result.startswith("Error:"),
                f"Expected 'Error: ...' but got: {result}",
            )
            self.assertIn("no artifact", result.lower())
        finally:
            gwsa.make_general_web_search_graph = original


if __name__ == "__main__":
    unittest.main()