from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from travelplanner.baseline_agent.agent import (
    _require_tavily_human_message,
    run_baseline,
)
from travelplanner.baseline_agent.config import BaselineAgentConfig, load_config_from_env
from travelplanner.baseline_agent.run_from_json import _load_cases, run_cases


class TestBaselineAgentConfig(unittest.TestCase):
    def test_loads_yaml_settings_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            global_cfg = tmp_path / "config.yaml"
            local_cfg = tmp_path / "missing-local.yaml"
            output_dir = tmp_path / "outputs"
            global_cfg.write_text(
                "\n".join(
                    [
                        "agents:",
                        "  baseline_agent:",
                        "    model_name: openrouter:minimax/minimax-m2.5",
                        "    temperature: 0.1",
                        "    recursion_limit: 18",
                        "    max_tool_calls: 3",
                        f"    output_dir: {output_dir}",
                        "    tavily:",
                        "      max_results: 4",
                        "      search_depth: basic",
                        "      include_answer: false",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "TRAVELPLANNER_GLOBAL_CONFIG_PATH": str(global_cfg),
                    "TRAVELPLANNER_LOCAL_CONFIG_PATH": str(local_cfg),
                    "TRAVELPLANNER_BASELINE_AGENT_TAVILY_SEARCH_DEPTH": "advanced",
                },
                clear=False,
            ):
                config = load_config_from_env()

        self.assertEqual(config.model_name, "openrouter:minimax/minimax-m2.5")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.recursion_limit, 18)
        self.assertEqual(config.min_tool_calls, 1)
        self.assertEqual(config.max_tool_calls, 3)
        self.assertEqual(config.output_dir, output_dir)
        self.assertEqual(config.tavily_max_results, 4)
        self.assertEqual(config.tavily_search_depth, "advanced")
        self.assertFalse(config.tavily_include_answer)

    def test_min_tool_calls_clamped_when_greater_than_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            global_cfg = tmp_path / "config.yaml"
            local_cfg = tmp_path / "missing-local.yaml"
            output_dir = tmp_path / "outputs"
            global_cfg.write_text(
                "\n".join(
                    [
                        "agents:",
                        "  baseline_agent:",
                        "    model_name: openrouter:test/model",
                        "    max_tool_calls: 2",
                        "    output_dir: " + str(output_dir).replace("\\", "/"),
                        "    tavily:",
                        "      max_results: 3",
                        "      search_depth: basic",
                        "      include_answer: true",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "TRAVELPLANNER_GLOBAL_CONFIG_PATH": str(global_cfg),
                    "TRAVELPLANNER_LOCAL_CONFIG_PATH": str(local_cfg),
                    "TRAVELPLANNER_BASELINE_AGENT_MIN_TOOL_CALLS": "9",
                },
                clear=False,
            ):
                config = load_config_from_env()

        self.assertEqual(config.max_tool_calls, 2)
        self.assertEqual(config.min_tool_calls, 2)


class TestBaselineRunnerInput(unittest.TestCase):
    def test_load_cases_accepts_cases_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(
                '{"cases":[{"name":"rome","query":"Plan Rome","constraints":["2 days"]}]}',
                encoding="utf-8",
            )
            cases = _load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "rome")
        self.assertEqual(cases[0].query, "Plan Rome")
        self.assertEqual(cases[0].constraints, ["2 days"])

    def test_load_cases_rejects_missing_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "Input JSON does not exist"):
            _load_cases(Path("/tmp/does-not-exist-baseline-agent.json"))

    def test_run_cases_writes_distinct_files_for_duplicate_case_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(
                '[{"name":"rome","query":"Plan A"},{"name":"rome","query":"Plan B"}]',
                encoding="utf-8",
            )
            out = Path(tmp) / "out"
            mock_result = MagicMock(
                markdown="# out",
                model_name="m",
                executed_tool_calls=0,
                requested_tool_calls=0,
            )
            with patch(
                "travelplanner.baseline_agent.run_from_json.run_baseline",
                return_value=mock_result,
            ):
                paths = run_cases(path, out)

        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0].resolve(), paths[1].resolve())
        self.assertEqual(paths[0].name, "01-rome.md")
        self.assertEqual(paths[1].name, "02-rome.md")


class TestRequireTavilyNudge(unittest.TestCase):
    def test_singular_when_min_is_one(self) -> None:
        text = _require_tavily_human_message(1).content
        self.assertIn("at least once", text.lower())

    def test_non_positive_min_returns_brief_continue_nudge(self) -> None:
        text = _require_tavily_human_message(0).content
        self.assertIn("Continue", text)
        self.assertNotIn("Workflow requirement", text)

    def test_plural_when_min_greater_than_one(self) -> None:
        text = _require_tavily_human_message(3).content
        self.assertIn("at least 3", text)


class TestBaselineRunFallbacks(unittest.TestCase):
    def test_run_baseline_finalizes_when_model_ignores_tool_nudge(self) -> None:
        class FakeModel:
            def bind_tools(self, _tools: object) -> "FakeModel":
                return self

            def invoke(self, messages: object) -> AIMessage:
                last = messages[-1]
                content = getattr(last, "content", "")
                if isinstance(content, str) and content.startswith("No more Tavily"):
                    return AIMessage(content="# Baseline itinerary: Rome")
                return AIMessage(content="I will answer without searching.")

        config = BaselineAgentConfig(
            model_name="fake/model",
            temperature=0.0,
            recursion_limit=20,
            min_tool_calls=1,
            max_tool_calls=4,
            output_dir=Path("/tmp"),
            tavily_max_results=5,
            tavily_search_depth="basic",
            tavily_include_answer=True,
        )
        with patch(
            "travelplanner.baseline_agent.agent.make_chat_model",
            return_value=FakeModel(),
        ):
            result = run_baseline(query="Plan Rome", constraints=[], config=config)

        self.assertIn("# Baseline itinerary: Rome", result.markdown)
        self.assertEqual(result.executed_tool_calls, 0)

    def test_run_baseline_returns_markdown_when_graph_recursion_limit_is_hit(self) -> None:
        class RecursingGraph:
            def stream(self, *_args: object, **_kwargs: object):
                raise GraphRecursionError("limit reached")

        with patch(
            "travelplanner.baseline_agent.agent.make_graph",
            return_value=RecursingGraph(),
        ):
            result = run_baseline(query="Plan Rome", constraints=[])

        self.assertIn("Baseline incomplete", result.markdown)
        self.assertEqual(result.executed_tool_calls, 0)
        self.assertEqual(result.requested_tool_calls, 0)

    def test_run_baseline_recursion_preserves_requested_tool_slots(self) -> None:
        seed = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_web_search",
                        "id": "call_test_1",
                        "args": {"query": "hotels in Rome"},
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        class PartialThenRecurseGraph:
            def stream(self, *_args: object, **_kwargs: object):
                yield {"messages": seed}
                raise GraphRecursionError("limit reached")

        with patch(
            "travelplanner.baseline_agent.agent.make_graph",
            return_value=PartialThenRecurseGraph(),
        ):
            result = run_baseline(query="Plan Rome", constraints=[])

        self.assertEqual(result.executed_tool_calls, 0)
        self.assertEqual(result.requested_tool_calls, 1)

    def test_run_baseline_preserves_partial_messages_on_recursion_error(self) -> None:
        seed = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="partial"),
        ]

        class PartialThenRecurseGraph:
            def stream(self, *_args: object, **_kwargs: object):
                yield {"messages": seed}
                raise GraphRecursionError("limit reached")

        with patch(
            "travelplanner.baseline_agent.agent.make_graph",
            return_value=PartialThenRecurseGraph(),
        ):
            result = run_baseline(query="Plan Rome", constraints=[])

        self.assertIn("Baseline incomplete", result.markdown)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.executed_tool_calls, 0)
        self.assertEqual(result.requested_tool_calls, 0)


if __name__ == "__main__":
    unittest.main()
