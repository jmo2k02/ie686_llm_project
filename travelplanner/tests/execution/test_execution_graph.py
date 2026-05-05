from __future__ import annotations

import unittest

from travelplanner.agents.execution.graph import dispatch_search_tasks, make_graph
from travelplanner.schema.system_state import AgentArtifactModel, StateContractModel, TaskModel


class FakeSearchGraph:
    def __init__(self, agent_key: str, calls: list[dict]) -> None:
        self.agent_key = agent_key
        self.calls = calls

    def invoke(self, agent_input: dict) -> dict:
        self.calls.append(agent_input)
        task = agent_input["task_list"][0]
        return {
            "agent_artifacts": {
                self.agent_key: [
                    AgentArtifactModel(
                        name=f"artifact-{task.name}",
                        type=f"{task.type}_search_result",
                        content={"task": task.text},
                    )
                ]
            },
            "message_history": None,
        }


def _make_fake_agent_map(calls: list[dict]) -> dict:
    return {
        "flight": ("flight_search_agent", lambda: FakeSearchGraph("flight_search_agent", calls)),
        "restaurant": (
            "restaurant_search_agent",
            lambda: FakeSearchGraph("restaurant_search_agent", calls),
        ),
    }


class TestExecutionGraph(unittest.TestCase):
    def test_dispatch_invokes_matching_agents_for_valid_tasks_only(self) -> None:
        calls: list[dict] = []
        state = StateContractModel(
            query="Plan a trip to Rome",
            task_list=[
                TaskModel(
                    name="find-flight",
                    type="flight",
                    text="Find flights to Rome",
                    is_valid=True,
                ),
                TaskModel(
                    name="skip-invalid",
                    type="restaurant",
                    text="Find dinner",
                    is_valid=False,
                    validation_comment="missing date",
                ),
                TaskModel(
                    name="skip-unsupported",
                    type="opening_times",
                    text="Check museum opening times",
                    is_valid=True,
                ),
            ],
        )

        update = dispatch_search_tasks(state, task_agent_map=_make_fake_agent_map(calls))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["query"], "Plan a trip to Rome")
        self.assertEqual(calls[0]["task_list"][0].name, "find-flight")
        self.assertIn("flight_search_agent", update["agent_artifacts"])
        self.assertEqual(update["agent_artifacts"]["flight_search_agent"][0].name, "artifact-find-flight")
        self.assertNotIn("restaurant_search_agent", update["agent_artifacts"])

    def test_dispatch_preserves_existing_artifacts_and_appends_new_ones(self) -> None:
        calls: list[dict] = []
        existing = AgentArtifactModel(
            name="existing-flight",
            type="flight_search_result",
            content={"source": "previous"},
        )
        state = StateContractModel(
            query="Plan a trip to Rome",
            agent_artifacts={"flight_search_agent": [existing]},
            task_list=[
                TaskModel(
                    name="new-flight",
                    type="flight",
                    text="Find flights to Rome",
                    is_valid=True,
                )
            ],
        )

        update = dispatch_search_tasks(state, task_agent_map=_make_fake_agent_map(calls))

        artifacts = update["agent_artifacts"]["flight_search_agent"]
        self.assertEqual([artifact.name for artifact in artifacts], ["existing-flight", "artifact-new-flight"])

    def test_make_graph_can_run_with_fake_agent_map(self) -> None:
        calls: list[dict] = []
        graph = make_graph(task_agent_map=_make_fake_agent_map(calls)).compile()
        state = StateContractModel(
            query="Plan a trip to Rome",
            task_list=[
                TaskModel(
                    name="find-flight",
                    type="flight",
                    text="Find flights to Rome",
                    is_valid=True,
                )
            ],
        )

        result = StateContractModel.model_validate(graph.invoke(state))

        self.assertEqual(len(calls), 1)
        self.assertEqual(result.agent_artifacts["flight_search_agent"][0].name, "artifact-find-flight")


if __name__ == "__main__":
    unittest.main()
