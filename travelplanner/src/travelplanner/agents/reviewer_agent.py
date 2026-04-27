from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.utils.llm import invoke_structured_model
from travelplanner.schema.system_state import (
    ConstraintModel,
    MessageHistoryModel,
    TaskModel,
)


SYSTEM_PROMPT = """You are the TravelPlanner reviewer agent.

Review the planner's task list and return the reviewed task list.

Rules:
- Return JSON only.
- Validate whether each task is useful, specific, and grounded in the user request.
- Remove duplicates or merge obviously redundant tasks.
- Set is_valid to true only for tasks that are ready for downstream execution.
- If a task is invalid, explain why in validation_comment.
- If the planner missed an essential task, you may add it.
"""


class ReviewerAgentState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    proposed_task_list: list[TaskModel] = Field(default_factory=list)
    approved_task_list: list[TaskModel] = Field(default_factory=list)
    message_history: MessageHistoryModel | None = None


class ReviewResponse(BaseModel):
    approved_task_list: list[TaskModel] = Field(default_factory=list)
    review_summary: str | None = None


def _build_user_prompt(
    query: str,
    constraints: list[ConstraintModel],
    proposed_task_list: list[TaskModel],
) -> str:
    return "\n".join(
        [
            "Review the planner task list for the travel request below.",
            "",
            f"User request: {query.strip()}",
            "",
            "Constraints JSON:",
            json.dumps(
                [constraint.model_dump() for constraint in constraints],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Proposed tasks JSON:",
            json.dumps(
                [task.model_dump() for task in proposed_task_list],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Return strictly valid JSON with this shape:",
            '{"approved_task_list": [{"name": "...", "type": "flight|hotel|restaurant|attraction|opening_times|routing-check|general-web-search", "text": "...", "is_valid": true, "validation_comment": null}], "review_summary": "..."}',
        ]
    )


def _build_message_history(
    query: str,
    user_prompt: str,
    raw_response: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="reviewer_agent",
        model="llm",
        agent_ref="travelplanner.agents.reviewer_agent",
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )


def make_graph():
    def reviewer_node(state: ReviewerAgentState) -> dict[str, Any]:
        structured_output, user_prompt, raw_response = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(
                state.query,
                state.constraint_list,
                state.proposed_task_list,
            ),
            response_model=ReviewResponse,
        )
        history = _build_message_history(state.query, user_prompt, raw_response)
        return {
            "approved_task_list": structured_output.approved_task_list,
            "message_history": history,
        }

    graph = StateGraph(ReviewerAgentState)
    graph.add_node("reviewer_agent", reviewer_node)
    graph.set_entry_point("reviewer_agent")
    graph.add_edge("reviewer_agent", END)
    return graph.compile()
