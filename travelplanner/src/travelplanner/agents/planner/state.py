"""Pydantic state and response models for the planner graph.

These models define the planner graph state plus the structured LLM response
shapes parsed from planner and reviewer prompts.
"""

from pydantic import BaseModel, Field

from travelplanner.schema.system_state import (
    ConstraintModel,
    TaskModel,
    MessageHistoryModel,
)


class PlannerAgentState(BaseModel):
    query: str
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
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
