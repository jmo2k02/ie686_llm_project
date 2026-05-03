from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from travelplanner.agents.general_web_search_agent import make_graph
from travelplanner.schema.system_state import TaskModel


def _make_response(score: float, result_count: int, query: str = "") -> dict:
    results = [
        {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "content": f"Content {i}",
            "score": score,
        }
        for i in range(result_count)
    ]
    return {
        "ok": True,
        "query": query,
        "answer": f"Answer for '{query}'",
        "results": results,
        "raw_response": {"jsonrpc": "2.0", "id": 1, "result": {"content": results}},
    }


class TestIterativeSearchQualityGating(unittest.TestCase):
    def _invoke_graph(self, responses: list[dict]) -> dict:
        idx = [0]

        def canned_search(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            return responses[i % len(responses)]

        env = {
            "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES": "1",
            "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_SEARCHES": "3",
            "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL": "none",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "travelplanner.agents.general_web_search_agent._search_tavily",
                side_effect=canned_search,
            ):
                graph = make_graph()
                result = graph.compile().invoke(
                    {
                        "query": "test query",
                        "task_list": [
                            TaskModel(
                                name="test-task-1",
                                type="general-web-search",
                                text="test task text",
                                is_valid=True,
                                validation_comment=None,
                            )
                        ],
                        "agent_artifacts": {},
                    }
                )
        return result

    def _search_outcomes(self, result: dict) -> list:
        artifacts = result.get("agent_artifacts", {})
        search_artifacts = artifacts.get("general_web_search_agent", [])
        self.assertGreaterEqual(len(search_artifacts), 1)
        content = search_artifacts[0].content
        return content.get("config", {}).get("search_outcomes", [])

    def test_single_search_when_results_high_quality(self):
        mock = [_make_response(score=0.8, result_count=3, query="test task text")]
        result = self._invoke_graph(mock)
        outcomes = self._search_outcomes(result)
        self.assertEqual(
            len(outcomes), 1, f"Expected 1 search, got {len(outcomes)}: {outcomes}"
        )

    def test_second_search_triggered_on_low_score(self):
        task_text = "test task text"
        refined = f"{task_text} site:wikidata.org OR site:openstreetmap.org"
        mock = [
            _make_response(score=0.2, result_count=3, query=task_text),
            _make_response(score=0.8, result_count=3, query=refined),
        ]
        result = self._invoke_graph(mock)
        outcomes = self._search_outcomes(result)
        self.assertEqual(
            len(outcomes), 2, f"Expected 2 searches, got {len(outcomes)}: {outcomes}"
        )

    def test_third_search_triggered_when_second_also_low(self):
        task_text = "test task text"
        refined = f"{task_text} site:wikidata.org OR site:openstreetmap.org"
        alt = f"latest news events {task_text}"
        mock = [
            _make_response(score=0.2, result_count=3, query=task_text),
            _make_response(score=0.2, result_count=3, query=refined),
            _make_response(score=0.2, result_count=3, query=alt),
        ]
        result = self._invoke_graph(mock)
        outcomes = self._search_outcomes(result)
        self.assertEqual(
            len(outcomes), 3, f"Expected 3 searches, got {len(outcomes)}: {outcomes}"
        )

    def test_max_searches_respected(self):
        task_text = "test task text"
        refined = f"{task_text} site:wikidata.org OR site:openstreetmap.org"
        alt = f"latest news events {task_text}"
        mock = [
            _make_response(score=0.2, result_count=3, query=task_text),
            _make_response(score=0.2, result_count=3, query=refined),
            _make_response(score=0.2, result_count=3, query=alt),
        ]
        result = self._invoke_graph(mock)
        outcomes = self._search_outcomes(result)
        self.assertEqual(
            len(outcomes),
            3,
            f"Expected at most 3 searches (max_searches default), got {len(outcomes)}: {outcomes}",
        )

    def test_quality_gate_also_checks_result_count(self):
        task_text = "test task text"
        refined = f"{task_text} site:wikidata.org OR site:openstreetmap.org"
        mock = [
            _make_response(score=0.8, result_count=1, query=task_text),
            _make_response(score=0.8, result_count=3, query=refined),
        ]
        result = self._invoke_graph(mock)
        outcomes = self._search_outcomes(result)
        self.assertEqual(
            len(outcomes), 2, f"Expected 2 searches, got {len(outcomes)}: {outcomes}"
        )


if __name__ == "__main__":
    unittest.main()
