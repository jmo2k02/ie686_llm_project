"""Deterministic heuristics for planner task inference.

This module maps user requests and extracted constraints to likely required
task types. It should remain deterministic and side-effect free so prompt
generation and reviewer validation can share the same inference logic.
"""

from __future__ import annotations

from travelplanner.agents.planner.constants import TASK_TYPE_KEYWORDS
from travelplanner.agents.planner.context import build_planner_request_text
from travelplanner.schema.system_state import ConstraintModel, TaskModel
from travelplanner.utils.text import (
    contains_any,
    mentions_specific_time,
    trip_likely_has_overnight_stay,
)


def infer_required_task_types(
    query: str,
    constraints: list[ConstraintModel],
) -> list[str]:
    request_text = build_planner_request_text(query, constraints).lower()
    required: list[str] = []

    if contains_any(request_text, TASK_TYPE_KEYWORDS["flight"]):
        required.append("flight")

    if contains_any(
        request_text, TASK_TYPE_KEYWORDS["hotel"]
    ) or trip_likely_has_overnight_stay(request_text):
        required.append("hotel")

    if contains_any(request_text, TASK_TYPE_KEYWORDS["restaurant"]):
        required.append("restaurant")

    if contains_any(request_text, TASK_TYPE_KEYWORDS["attraction"]):
        required.append("attraction")

    if contains_any(
        request_text, TASK_TYPE_KEYWORDS["opening_times"]
    ) or mentions_specific_time(request_text):
        required.append("opening_times")

    if contains_any(request_text, TASK_TYPE_KEYWORDS["routing-check"]) or (
        " from " in request_text and " to " in request_text
    ):
        required.append("routing-check")

    if contains_any(request_text, TASK_TYPE_KEYWORDS["general-web-search"]):
        required.append("general-web-search")

    if not required:
        required.append("general-web-search")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in required:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def task_type_matches_text(task: TaskModel) -> bool:
    if task.type == "general-web-search":
        return True
    keywords = TASK_TYPE_KEYWORDS.get(task.type, ())
    haystack = f"{task.name} {task.text}".lower()
    return any(keyword in haystack for keyword in keywords)
