"""Backward-compatible imports for the planner agent.

The planner implementation lives in travelplanner.agents.planner. This module
keeps existing imports working while the planner is migrated into smaller,
focused modules.
"""

from __future__ import annotations

from travelplanner.agents.planner.context import build_planner_request_text
from travelplanner.agents.planner.graph import make_graph
from travelplanner.agents.planner.heuristics import (
    infer_required_task_types,
    task_type_matches_text,
)
from travelplanner.agents.planner.state import (
    PlannerAgentState,
    PlanningResponse,
    ReviewResponse,
)
from travelplanner.agents.planner.validation import (
    coverage_check,
    normalize_tasks,
    review_feedback_text,
    task_issues,
    task_signature,
    task_slug,
    validate_task_list,
)

__all__ = [
    "PlannerAgentState",
    "PlanningResponse",
    "ReviewResponse",
    "build_planner_request_text",
    "coverage_check",
    "infer_required_task_types",
    "make_graph",
    "normalize_tasks",
    "review_feedback_text",
    "task_issues",
    "task_signature",
    "task_slug",
    "task_type_matches_text",
    "validate_task_list",
]
