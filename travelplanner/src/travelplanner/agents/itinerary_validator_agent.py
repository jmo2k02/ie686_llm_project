"""Itinerary Validator Agent — final quality gate before returning to the user.

Input:  ``CalenderModel`` (timetable), constraints, task_list, agent_artifacts
Output: ``{validation_passed: bool, validation_feedback: str}``

- Pass   → workflow ends (END)
- Fail   → loops back to Execution Agent so it can rebuild the timetable.

The validator runs a structured LLM call that checks:
1. All hard constraints are respected (dates, budget, travelers, destinations).
2. Internal travel consistency: activities are ordered logically, travel time between
   locations is accounted for, opening hours respected.
3. Every valid task has a corresponding activity (completeness).

It is deterministic (temperature 0) and never raises.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.config import get_setting
from travelplanner.schema.system_state import MessageHistoryModel, StateContractModel
from travelplanner.utils.llm import invoke_structured_model


_DEFAULT_MODEL = "openai:gpt-4o-mini"


class ValidationResponse(BaseModel):
    """Structured output produced by the Itinerary Validator."""

    passed: bool = Field(description="Whether the itinerary passes all checks")
    feedback: str = Field(
        default="",
        description=(
            "If passed == false: concise, actionable instructions for the Execution Agent "
            "on what to fix. If passed == true: may be empty or contain a brief confirmation."
        ),
    )
    issues_found: list[str] = Field(default_factory=list, description="List of specific issues detected")


SYSTEM_PROMPT = """You are the Itinerary Validator for TravelPlanner.

You receive a proposed day-by-day itinerary (timetable) together with the original
constraints and task list.  Your job is to decide PASS or FAIL.

Check categories:
1. **Constraint compliance**
   - Destination matches the requested destination.
   - Travel dates fall within the requested range.
   - Total cost estimate respects the budget (if a budget was given).
   - Number of travelers respected for lodging / transport capacity.
2. **Internal travel consistency**
   - Activities within a day are ordered logically (no time overlaps, no impossible jumps).
   - Travel time between locations is plausible or explicitly scheduled.
   - Opening hours / operating times are respected where known.
3. **Completeness**
   - Every valid task from the task list has a matching activity on the calendar.
   - No critical category (flight, hotel, transport) is missing without justification.

Rules:
- Return ``passed: true`` only if **all** categories are satisfied.
- If anything is wrong, return ``passed: false`` and provide clear, actionable
  ``feedback`` so the Execution Agent can fix it.  List each issue in ``issues_found``.
- Do not hallucinate facts not present in the input.

Return valid JSON matching the ValidationResponse schema.
"""


def _build_user_prompt(state: StateContractModel) -> str:
    timetable_blob = (
        state.timetable.model_dump_json(indent=2)
        if state.timetable is not None
        else "{}"
    )
    constraints_blob = json.dumps(
        [c.model_dump() for c in state.constraint_list], indent=2, ensure_ascii=False
    )
    tasks_blob = json.dumps(
        [t.model_dump() for t in state.task_list], indent=2, ensure_ascii=False
    )

    return (
        f"User query: {state.query}\n\n"
        f"Constraints:\n{constraints_blob}\n\n"
        f"Tasks:\n{tasks_blob}\n\n"
        f"Proposed timetable:\n{timetable_blob}\n\n"
        "Validate the itinerary and return PASS or FAIL with feedback."
    )


def validator_node(state: StateContractModel) -> dict[str, Any]:
    model_name = str(
        get_setting("models.agents.itinerary_validator.model_name", _DEFAULT_MODEL)
    )
    temperature = float(
        get_setting("models.agents.itinerary_validator.temperature", 0.0)
    )
    user_prompt = _build_user_prompt(state)

    try:
        structured_output, _, raw_response = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ValidationResponse,
        )
        passed = structured_output.passed
        feedback = structured_output.feedback
        issues_found = structured_output.issues_found
    except Exception as exc:
        # Graceful degradation: fail with error explanation so execution can retry
        passed = False
        feedback = f"Validator encountered an error: {exc}. Please rebuild the itinerary."
        issues_found = [feedback]
        raw_response = feedback

    history = MessageHistoryModel(
        user_agent="itinerary_validator",
        model=model_name,
        agent_ref="travelplanner.agents.itinerary_validator_agent",
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )

    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
        "message_histories": {
            **state.message_histories,
            "itinerary_validator": history,
        },
    }


def make_graph() -> StateGraph:
    graph = StateGraph(StateContractModel)
    graph.add_node("itinerary_validator", validator_node)
    graph.set_entry_point("itinerary_validator")
    graph.add_edge("itinerary_validator", END)
    return graph
