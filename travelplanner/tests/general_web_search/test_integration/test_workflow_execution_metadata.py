from __future__ import annotations

import unittest
from unittest.mock import patch

from travelplanner.schema.system_state import TaskModel
from travelplanner.workflows.task_planning import make_graph


class _FakeCompiledGraph:
    def __init__(self, result: dict) -> None:
        self._result = result

    def invoke(self, _state):
        return self._result


class TestWorkflowExecutionMetadata(unittest.TestCase):
    def test_web_search_history_key_is_recorded(self) -> None:
        with patch("travelplanner.workflows.task_planning.make_constraint_graph") as mock_c:
            with patch("travelplanner.workflows.task_planning.make_planner_graph") as mock_p:
                with patch("travelplanner.workflows.task_planning.make_reviewer_graph") as mock_r:
                    with patch("travelplanner.workflows.task_planning.make_general_web_search_graph") as mock_w:
                        mock_c.return_value = _FakeCompiledGraph(
                            {"constraint_list": [], "message_history": {"messages": []}}
                        )
                        mock_p.return_value = _FakeCompiledGraph(
                            {
                                "task_list": [
                                    TaskModel(
                                        name="general-search-1",
                                        type="general-web-search",
                                        text="find event dates",
                                        is_valid=True,
                                        validation_comment=None,
                                    )
                                ],
                                "message_history": {"messages": []},
                            }
                        )
                        mock_r.return_value = _FakeCompiledGraph(
                            {
                                "approved_task_list": [
                                    TaskModel(
                                        name="general-search-1",
                                        type="general-web-search",
                                        text="find event dates",
                                        is_valid=True,
                                        validation_comment=None,
                                    )
                                ],
                                "message_history": {"messages": []},
                            }
                        )
                        mock_w.return_value = _FakeCompiledGraph(
                            {
                                "agent_artifacts": {
                                    "general_web_search_agent": [
                                        {
                                            "name": "general-search-1",
                                            "type": "general-web-search-result",
                                            "content": {"status": "failed"},
                                        }
                                    ]
                                },
                                "message_history": {"messages": []},
                            }
                        )
                        graph = make_graph()
                        result = graph.invoke({"query": "q"})

        self.assertIn("general_web_search_agent", result["message_histories"])


if __name__ == "__main__":
    unittest.main()
