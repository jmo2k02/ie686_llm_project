"""LangGraph wiring for the planner and planner-reviewer agents.

This module connects planner drafting, reviewer validation, retry routing, and
final output assembly. Prompt construction, validation policy, and heuristics
live in sibling modules.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from travelplanner.agents.planner import config
from travelplanner.agents.planner.constants import (
    PLANNER_HISTORY_KEY,
    PLANNER_REVIEWER_HISTORY_KEY,
)
from travelplanner.agents.planner.history import build_message_history
from travelplanner.agents.planner.prompts import (
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    build_planner_prompt,
    build_reviewer_prompt,
)
from travelplanner.agents.planner.validation import (
    normalize_tasks,
    review_feedback_text,
    validate_task_list,
)
from travelplanner.schema.normalized_constraints import NormalizedConstraints
from travelplanner.schema.system_state import (
    ConstraintModel,
    TaskModel,
    MessageHistoryModel,
)
from travelplanner.config import get_setting
from travelplanner.utils.llm import invoke_structured_model


class PlannerAgentState(BaseModel):
    query: str
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    # TODO integrate this new attribute into task generation
    normalized_constraints: NormalizedConstraints | None = None
    task_list: list[TaskModel] = Field(default_factory=list)
    message_histories: dict[str, MessageHistoryModel] = Field(default_factory=dict)
    planner_review_feedback: str | None = None
    planner_review_attempts: int = 0
    planner_approved: bool = False
    review_summary: str | None = None


class PlanningResponse(BaseModel):
    tasks: list[TaskModel] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    approved_task_list: list[TaskModel] = Field(default_factory=list)
    review_summary: str | None = None



def planner_node(
    state: PlannerAgentState,
    *,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    structured_output, user_prompt, raw_response = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=build_planner_prompt(
            state.query,
            state.constraint_list,
            state.planner_review_feedback,
        ),
        response_model=PlanningResponse,
    )
    history = build_message_history(
        user_agent=PLANNER_HISTORY_KEY,
        agent_ref="travelplanner.agents.planner.graph",
        query=state.query,
        user_prompt=user_prompt,
        raw_response=raw_response,
    )
    return {
        "task_list": normalize_tasks(structured_output.tasks, default_is_valid=False),
        "planner_approved": False,
        "review_summary": None,
        "message_histories": {
            **state.message_histories,
            PLANNER_HISTORY_KEY: history,
        },
    }


def reviewer_node(
    state: PlannerAgentState,
    *,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    structured_output, user_prompt, raw_response = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_prompt=build_reviewer_prompt(
            state.query,
            state.constraint_list,
            state.task_list,
        ),
        response_model=ReviewResponse,
    )
    reviewed_tasks, summary = validate_task_list(
        structured_output.approved_task_list,
        state.query,
        state.constraint_list,
    )
    if structured_output.review_summary:
        summary = [*summary, structured_output.review_summary]

    planner_approved = (
        bool(reviewed_tasks)
        and all(task.is_valid for task in reviewed_tasks)
        and not any(
            line.startswith("Missing likely task types:")
            or line.startswith("Potentially uncovered constraints:")
            for line in summary
        )
    )
    feedback = None if planner_approved else review_feedback_text(reviewed_tasks, summary)

    history = build_message_history(
        user_agent=PLANNER_REVIEWER_HISTORY_KEY,
        agent_ref="travelplanner.agents.planner.graph",
        query=state.query,
        user_prompt=user_prompt,
        raw_response=raw_response,
    )
    if summary:
        history.messages.append(
            {
                "role": "assistant",
                "content": "Deterministic review summary: " + " | ".join(summary),
            }
        )

    return {
        "task_list": reviewed_tasks,
        "planner_review_attempts": state.planner_review_attempts + 1,
        "planner_review_feedback": feedback,
        "planner_approved": planner_approved,
        "review_summary": feedback,
        "message_histories": {
            **state.message_histories,
            PLANNER_REVIEWER_HISTORY_KEY: history,
        },
    }


def route_after_review(state: PlannerAgentState) -> str:
    if state.planner_approved or state.planner_review_attempts >= config.MAX_REVIEW_ATTEMPTS:
        return "finalize_planner_output"
    return "planner_draft"


def finalize_output(state: PlannerAgentState) -> dict[str, Any]:
    interrupt("Decided for final task list")
    return {
        "task_list": state.task_list,
        "message_histories": state.message_histories,
    }


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    effective_model_name = model_name or str(
        get_setting(
            "models.workflows.task_planning.model_name",
            "gpt-5.4-nano-2026-03-17",
        )
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )

    graph = StateGraph(PlannerAgentState)
    graph.add_node(
        "planner_draft",
        lambda state: planner_node(
            state,
            model_name=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
        ),
    )
    graph.add_node(
        "reviewer_agent",
        lambda state: reviewer_node(
            state,
            model_name=config.REVIEWER_MODEL_NAME,
            temperature=config.REVIEWER_TEMP,
        ),
    )
    graph.add_node("finalize_planner_output", finalize_output)

    graph.set_entry_point("planner_draft")
    graph.add_edge("planner_draft", "reviewer_agent")
    graph.add_conditional_edges("reviewer_agent", route_after_review)
    graph.add_edge("finalize_planner_output", END)
    return graph
