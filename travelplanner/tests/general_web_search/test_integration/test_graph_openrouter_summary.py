from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from travelplanner.agents.general_web_search_agent import make_graph
from travelplanner.schema.system_state import TaskModel


class _FakeChatModel:
    def invoke(self, _messages):
        return SimpleNamespace(content="- Fact 1\n- Fact 2")


class TestGraphOpenRouterSummary(unittest.TestCase):
    def test_graph_stores_artifact_and_openrouter_answer(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TAVILY_API_KEY": "test-key",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL": "openrouter:minimax/minimax-m2.5:free",
            },
            clear=True,
        ):
            with patch(
                "travelplanner.agents.general_web_search_agent._search_tavily",
                return_value={
                    "ok": True,
                    "query": "cheap flights berlin to rome",
                    "answer": "Sample answer",
                    "results": [{"title": "Sample", "url": "https://example.com"}],
                },
            ):
                with patch(
                    "travelplanner.agents.general_web_search_agent.make_chat_model",
                    return_value=_FakeChatModel(),
                ):
                    graph = make_graph()
                    result = graph.invoke(
                        {
                            "query": "Plan me a Rome trip",
                            "task_list": [
                                TaskModel(
                                    name="general-search-1",
                                    type="general-web-search",
                                    text="cheap flights berlin to rome",
                                    is_valid=True,
                                    validation_comment=None,
                                )
                            ],
                            "agent_artifacts": {},
                        }
                    )

        artifacts = result["agent_artifacts"]["general_web_search_agent"]
        self.assertEqual(len(artifacts), 1)
        content = artifacts[0].content
        self.assertEqual(content["status"], "success")
        self.assertEqual(content["task_ref"], "general-search-1")
        self.assertEqual(content["provider"], "tavily")
        self.assertEqual(content["attempt"], 1)
        self.assertTrue(content["result"]["ok"])
        self.assertTrue(content["answer"])
        self.assertIsInstance(content["proof_points"], list)
        self.assertEqual(
            content["model"],
            "openrouter:minimax/minimax-m2.5:free",
        )


if __name__ == "__main__":
    unittest.main()
