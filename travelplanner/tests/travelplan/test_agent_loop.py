"""Agent loop test for the execution agent.

Drives the deepagents-built graph with a scripted fake chat model so we can
verify the tool-call loop end-to-end without burning real API tokens.

Run from travelplanner/ directory:
    uv run python -m pytest tests/travelplan/test_agent_loop.py -v
"""
from __future__ import annotations

import unittest
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travelplanner.agents.execution import make_graph
from travelplanner.travelplan import TravelPlan


class _ScriptedToolCallModel(FakeMessagesListChatModel):
    """Fake chat model that:

    1. Implements ``bind_tools`` as a no-op (deepagents binds tools to the
       model during graph construction; the parent ``BaseChatModel`` raises
       NotImplementedError, which would crash the graph build).
    2. Does NOT cycle past the last response. The base
       ``FakeMessagesListChatModel`` wraps around to index 0 once the list
       is exhausted, which would re-fire ``init_plan`` and silently wipe
       the plan if the agent invokes the model one extra time.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = min(self.i, len(self.responses) - 1)
        self.i = idx + 1
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[idx])]
        )


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
    )


class TestExecutionAgentLoop(unittest.TestCase):
    def test_scripted_loop_drives_travelplan_to_expected_state(self):
        plan = TravelPlan()
        scripted = _ScriptedToolCallModel(
            responses=[
                _tool_call("c1", "init_plan", {"title": "Rome 1-day"}),
                _tool_call(
                    "c2",
                    "add_day",
                    {"label": "Arrival", "calendar_date_iso": "2026-06-01"},
                ),
                _tool_call(
                    "c3",
                    "add_slot",
                    {
                        "day_index": 1,
                        "name": "Breakfast",
                        "start_time_iso": "2026-06-01T08:00",
                        "end_time_iso": "2026-06-01T10:00",
                        "category": "meal",
                        "cost": 15.0,
                    },
                ),
                _tool_call("c4", "view_plan", {}),
                AIMessage(
                    content="Done. Built a 1-day Rome plan with breakfast.",
                    tool_calls=[],
                ),
            ]
        )

        graph = make_graph(plan, model=scripted)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Plan a 1-day Rome trip with breakfast.")]}
        )

        # The closure-bound plan should have been mutated end-to-end.
        self.assertEqual(plan.title, "Rome 1-day")
        self.assertEqual(len(plan.days), 1)
        self.assertEqual(plan.days[0].label, "Arrival")
        self.assertEqual(plan.days[0].calendar_date.isoformat(), "2026-06-01")
        self.assertEqual(len(plan.days[0].slots), 1)
        slot = plan.days[0].slots[0]
        self.assertEqual(slot.name, "Breakfast")
        self.assertEqual(slot.cost, 15.0)
        self.assertEqual(slot.category, "meal")

        # Loop terminated on the no-tool-calls AIMessage.
        final = result["messages"][-1]
        self.assertIn("Done", final.content)
        self.assertFalse(getattr(final, "tool_calls", []))

    def test_loop_survives_a_tool_error_and_recovers(self):
        """If a tool returns 'Error: ...', the agent loop must NOT crash —
        it should be able to call another tool with corrected arguments."""
        plan = TravelPlan()
        scripted = _ScriptedToolCallModel(
            responses=[
                _tool_call("c1", "add_day", {"label": "X"}),
                # First add_slot uses a bad ISO string -> tool returns "Error: ..."
                _tool_call(
                    "c2",
                    "add_slot",
                    {
                        "day_index": 1,
                        "name": "Breakfast",
                        "start_time_iso": "not-iso",
                        "end_time_iso": "2026-06-01T10:00",
                    },
                ),
                # Recovery: correct ISO -> succeeds
                _tool_call(
                    "c3",
                    "add_slot",
                    {
                        "day_index": 1,
                        "name": "Breakfast",
                        "start_time_iso": "2026-06-01T08:00",
                        "end_time_iso": "2026-06-01T10:00",
                    },
                ),
                AIMessage(content="Recovered.", tool_calls=[]),
            ]
        )

        graph = make_graph(plan, model=scripted)
        graph.invoke({"messages": [HumanMessage(content="add a breakfast slot")]})

        self.assertEqual(len(plan.days), 1)
        self.assertEqual(len(plan.days[0].slots), 1)
        self.assertEqual(plan.days[0].slots[0].name, "Breakfast")


if __name__ == "__main__":
    unittest.main()
