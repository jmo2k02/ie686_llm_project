from __future__ import annotations

import unittest
from unittest.mock import patch

from travelplanner.agents.execution.graph import _compose_user_prompt, make_graph
from travelplanner.agents.execution.prompts import SYSTEM_PROMPT, VALIDATION_SYSTEM_PROMPT
from travelplanner.schema.system_state import StateContractModel
from travelplanner.travelplan import TravelPlan


class TestValidationAgentSetup(unittest.TestCase):
    def test_validation_mode_uses_repair_prompt_and_removes_init_plan(self) -> None:
        captured: dict = {}

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return object()

        with (
            patch("travelplanner.agents.execution.graph.make_chat_model", return_value=object()),
            patch("travelplanner.agents.execution.graph.make_subagent_tools", return_value=[]),
            patch("travelplanner.agents.execution.graph.create_deep_agent", side_effect=fake_create_deep_agent),
        ):
            make_graph(TravelPlan(), model="test:model", validation_mode=True)

        self.assertEqual(captured["system_prompt"], VALIDATION_SYSTEM_PROMPT)
        tool_names = [tool.name for tool in captured["tools"]]
        self.assertNotIn("init_plan", tool_names)
        self.assertIn("view_plan", tool_names)

    def test_normal_mode_keeps_init_plan_and_execution_prompt(self) -> None:
        captured: dict = {}

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return object()

        with (
            patch("travelplanner.agents.execution.graph.make_chat_model", return_value=object()),
            patch("travelplanner.agents.execution.graph.make_subagent_tools", return_value=[]),
            patch("travelplanner.agents.execution.graph.create_deep_agent", side_effect=fake_create_deep_agent),
        ):
            make_graph(TravelPlan(), model="test:model")

        self.assertEqual(captured["system_prompt"], SYSTEM_PROMPT)
        self.assertIn("init_plan", [tool.name for tool in captured["tools"]])

    def test_validation_feedback_is_included_in_execution_prompt(self) -> None:
        state = StateContractModel(
            query="Plan Rome",
            validation_feedback="Dinner overlaps with museum visit.",
            validation_attempts=1,
        )

        prompt = _compose_user_prompt(state)

        self.assertIn("Validator feedback", prompt)
        self.assertIn("Dinner overlaps with museum visit.", prompt)
        self.assertIn("repairing the existing plan", prompt)


if __name__ == "__main__":
    unittest.main()
