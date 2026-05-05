from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from travelplanner.agents.general_web_search_agent import make_graph
from travelplanner.schema.system_state import TaskModel


class TestRetryAndStatus(unittest.TestCase):
    def test_failed_search_uses_bounded_retries_and_failed_status(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES": "2",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL": "none",
            },
            clear=True,
        ):
            with patch(
                "travelplanner.agents.general_web_search_agent._search_tavily",
                return_value={
                    "ok": False,
                    "error": "timeout",
                    "results": [],
                    "query": "x",
                },
            ) as mock_search:
                graph = make_graph()
                result = graph.compile().invoke(
                    {
                        "query": "Plan me a trip",
                        "task_list": [
                            TaskModel(
                                name="general-search-1",
                                type="general-web-search",
                                text="find local event dates",
                                is_valid=True,
                                validation_comment=None,
                            )
                        ],
                        "agent_artifacts": {},
                    }
                )

        self.assertEqual(mock_search.call_count, 6)
        artifacts = result["agent_artifacts"]["general_web_search_agent"]
        content = artifacts[0].content
        self.assertEqual(content["status"], "failed")
        self.assertEqual(content["attempt"], 2)
        self.assertGreaterEqual(len(content["errors"]), 1)

    def test_agent_does_quality_gated_searches(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES": "1",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_SEARCHES": "3",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL": "openrouter:minimax/minimax-m2.5",
            },
            clear=True,
        ):
            with patch(
                "travelplanner.agents.general_web_search_agent._search_tavily",
                return_value={
                    "ok": True,
                    "query": "q",
                    "answer": "a",
                    "results": [
                        {"title": "x", "url": "http://x", "content": "y", "score": 0.3}
                    ],
                },
            ) as mock_search:
                with patch(
                    "travelplanner.agents.general_web_search_agent._synthesize_answer_with_model",
                    return_value={"ok": True, "model": "m", "text": "- x"},
                ):
                    graph = make_graph()
                    result = graph.compile().invoke(
                        {
                            "query": "Plan me a trip",
                            "task_list": [
                                TaskModel(
                                    name="general-search-1",
                                    type="general-web-search",
                                    text="find museums and opening hours",
                                    is_valid=True,
                                    validation_comment=None,
                                )
                            ],
                            "agent_artifacts": {},
                        }
                    )

        self.assertEqual(mock_search.call_count, 3)
        content = result["agent_artifacts"]["general_web_search_agent"][0].content
        self.assertEqual(content["config"]["max_searches"], 3)


if __name__ == "__main__":
    unittest.main()
