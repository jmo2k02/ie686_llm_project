"""Execution Agent — assembles search results and builds the final timetable.

2-step graph:
1. ``search_orchestrator`` — dynamically calls search agents per task type
   (flight, hotel, restaurant, attraction) and stores results in ``agent_artifacts``.
2. ``timetable_builder`` — LLM call that maps tasks + artifacts onto a
   day-by-day itinerary.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.agents.search_orchestrator import run_searches
from travelplanner.config import get_setting
from travelplanner.schema.calender import (
    ActivityModel,
    CalenderModel,
    DayModel,
    TripSummaryModel,
)
from travelplanner.schema.system_state import AgentArtifactModel, MessageHistoryModel, StateContractModel
from travelplanner.utils.llm import invoke_structured_model


_DEFAULT_MODEL = "openai:gpt-4o-mini"


class ExecutionResponse(BaseModel):
    """Structured output produced by the Timetable Builder LLM."""

    timetable: CalenderModel = Field(default_factory=CalenderModel)


SYSTEM_PROMPT = """You are the Execution Agent for TravelPlanner.

Your job: take the validated tasks and **completed search artifacts**, and produce
a structured day-by-day itinerary (timetable).

Input:
- User query and constraints
- Task list (each with type: flight/hotel/restaurant/attraction/transport)
- Agent artifacts: search results with prices, times, locations, options

Rules:
1. Respect travel_dates, budget, and traveler constraints.
2. Order activities logically per day (flights/travel first, then hotels,
   restaurants at meal times, attractions during open hours).
3. Reference source artifacts in each activity (source_task_name, source_artifact_type).
4. Use concrete data from search artifacts (real prices, times, addresses).
5. If a task has no usable artifact, create a placeholder and list the task
   name in ``unscheduled_tasks``.
6. Provide cost estimates per activity and totals in ``trip_summary``.
7. If ``validator_feedback`` is provided, address every issue before building.

Return valid JSON matching the ExecutionResponse schema.
"""


def _build_user_prompt(state: StateContractModel) -> str:
    constraints_blob = json.dumps(
        [c.model_dump() for c in state.constraint_list], indent=2, ensure_ascii=False
    )
    tasks_blob = json.dumps(
        [t.model_dump() for t in state.task_list], indent=2, ensure_ascii=False
    )
    artifacts_blob = json.dumps(
        {
            key: [a.model_dump() for a in val]
            for key, val in state.agent_artifacts.items()
        },
        indent=2,
        ensure_ascii=False,
    )

    feedback_section = ""
    if state.validation_feedback:
        feedback_section = (
            f"\nVALIDATOR FEEDBACK (attempt {state.validation_attempts + 1}/3):\n"
            f"{state.validation_feedback}\n\n"
            "Please fix all issues above before producing the new timetable."
        )

    return (
        f"User query: {state.query}\n\n"
        f"Constraints:\n{constraints_blob}\n\n"
        f"Tasks:\n{tasks_blob}\n\n"
        f"Search results (artifacts):\n{artifacts_blob}\n\n"
        f"{feedback_section}"
        "Produce the timetable now."
    )


def search_orchestrator_node(state: StateContractModel) -> dict[str, Any]:
    """Step 1: dynamically run search agents per task type."""
    return run_searches(state)


def timetable_builder_node(state: StateContractModel) -> dict[str, Any]:
    """Step 2: LLM call that maps tasks + artifacts to a structured timetable."""
    model_name = str(
        get_setting("models.agents.execution.model_name", _DEFAULT_MODEL)
    )
    temperature = float(
        get_setting("models.agents.execution.temperature", 0.0)
    )
    user_prompt = _build_user_prompt(state)

    try:
        structured_output, _, raw_response = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ExecutionResponse,
        )
        timetable = structured_output.timetable
    except Exception as exc:
        timetable = CalenderModel(
            trip_summary=TripSummaryModel(),
            unscheduled_tasks=[t.name for t in state.task_list],
        )
        raw_response = f"Timetable builder failed: {exc}"

    history = MessageHistoryModel(
        user_agent="execution_agent",
        model=model_name,
        agent_ref="travelplanner.agents.execution_agent",
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )

    return {
        "timetable": timetable,
        "message_histories": {
            **state.message_histories,
            "execution_agent": history,
        },
    }


def make_graph() -> StateGraph:
    graph = StateGraph(StateContractModel)
    graph.add_node("search_orchestrator", search_orchestrator_node)
    graph.add_node("timetable_builder", timetable_builder_node)
    graph.set_entry_point("search_orchestrator")
    graph.add_edge("search_orchestrator", "timetable_builder")
    graph.add_edge("timetable_builder", END)
    return graph
