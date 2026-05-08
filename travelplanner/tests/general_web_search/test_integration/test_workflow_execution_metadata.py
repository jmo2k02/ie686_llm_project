from __future__ import annotations

import unittest
from unittest.mock import patch

from travelplanner.schema.system_state import TaskModel
from travelplanner.workflows.task_planning import make_graph


class _FakeCompiledInvoke:
    """Compiled graph used via ``.invoke()`` (web search + routing-check in workflow)."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def invoke(self, _state, config=None):  # noqa: ANN001
        return self._result


class _FakeCompilableSubgraph:
    """Constraint / planner subgraph nodes: ``compile()`` → callable for ``add_node``."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def compile(self):
        result = self._result

        def runnable(_state):  # noqa: ANN001
            return result

        return runnable


class _FakeCompilableInvoke:
    """General web search: ``compile()`` → object with ``invoke``."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def compile(self) -> _FakeCompiledInvoke:
        return _FakeCompiledInvoke(self._result)


class TestWorkflowExecutionMetadata(unittest.TestCase):
    def test_web_search_history_key_is_recorded(self) -> None:
        with patch("travelplanner.workflows.task_planning.make_constraint_graph") as mock_c:
            with patch("travelplanner.workflows.task_planning.make_planner_graph") as mock_p:
                with patch(
                    "travelplanner.workflows.task_planning.make_general_web_search_graph",
                ) as mock_w:
                    with patch(
                        "travelplanner.workflows.task_planning.make_routing_check_graph",
                    ) as mock_r:
                        mock_c.return_value = _FakeCompilableSubgraph(
                            {
                                "constraint_list": [],
                                "message_history": {"messages": []},
                            },
                        )
                        mock_p.return_value = _FakeCompilableSubgraph(
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
                            },
                        )
                        mock_w.return_value = _FakeCompilableInvoke(
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
                            },
                        )
                        mock_r.return_value = _FakeCompiledInvoke(
                            {
                                "message_history": {"messages": []},
                                "artifact": None,
                            },
                        )
                        graph = make_graph().compile()
                        result = graph.invoke({"query": "q"})

        self.assertIn("general_web_search_agent", result["message_histories"])


if __name__ == "__main__":
    unittest.main()
