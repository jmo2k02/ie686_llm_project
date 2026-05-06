"""Task normalization and validation helpers for planner output.

This module checks whether proposed tasks are specific, grounded in the request,
deduplicated, and sufficient for likely required task types before downstream
search agents execute them.
"""

from __future__ import annotations

from travelplanner.agents.planner.config import MAX_TASKS
from travelplanner.agents.planner.constants import STOPWORDS
from travelplanner.agents.planner.context import build_planner_request_text
from travelplanner.agents.planner.heuristics import (
    infer_required_task_types,
    task_type_matches_text,
)
from travelplanner.schema.system_state import ConstraintModel, TaskModel
from travelplanner.utils.text import extract_keywords, normalize_whitespace


def task_slug(task_type: str, index: int) -> str:
    return f"{task_type.replace('_', '-').replace(' ', '-')}-{index}"


def task_signature(task: TaskModel) -> tuple[str, str]:
    return (task.type, normalize_whitespace(task.text).lower())


def normalize_tasks(
    tasks: list[TaskModel],
    *,
    default_is_valid: bool,
    max_tasks: int = MAX_TASKS,
) -> list[TaskModel]:
    normalized: list[TaskModel] = []
    seen_signatures: set[tuple[str, str]] = set()

    for index, task in enumerate(tasks, 1):
        text = normalize_whitespace(task.text) or normalize_whitespace(task.name)
        if not text:
            continue

        normalized_task = TaskModel(
            name=normalize_whitespace(task.name) or task_slug(task.type, index),
            type=task.type,
            text=text,
            is_valid=default_is_valid,
            validation_comment=normalize_whitespace(task.validation_comment) or None,
        )
        signature = task_signature(normalized_task)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        normalized.append(normalized_task)
        if len(normalized) >= max_tasks:
            break

    return normalized


def coverage_check(
    tasks: list[TaskModel],
    constraints: list[ConstraintModel],
) -> dict[str, list[str]]:
    task_keywords: set[str] = set()
    task_types = {task.type for task in tasks}
    for task in tasks:
        task_keywords.update(extract_keywords(task.name, STOPWORDS))
        task_keywords.update(extract_keywords(task.text, STOPWORDS))

    uncovered_constraints: list[str] = []
    for constraint in constraints:
        if constraint.user_skipped:
            continue
        constraint_keywords = extract_keywords(constraint.text, STOPWORDS)
        if constraint_keywords and not constraint_keywords.intersection(task_keywords):
            uncovered_constraints.append(constraint.text)

    return {
        "covered_task_types": sorted(task_types),
        "uncovered_constraints": uncovered_constraints,
    }


def task_issues(task: TaskModel, *, request_keywords: set[str]) -> list[str]:
    issues: list[str] = []
    if len(task.text) < 18:
        issues.append("task text is too short")

    lowered = task.text.lower()
    if any(token in lowered for token in ("...", "todo", "tbd", "something", "stuff")):
        issues.append("task text looks like a placeholder")

    if not task_type_matches_text(task):
        issues.append("task text does not clearly match its task type")

    task_keywords = extract_keywords(task.name, STOPWORDS) | extract_keywords(
        task.text,
        STOPWORDS,
    )
    if request_keywords and not task_keywords.intersection(request_keywords):
        issues.append("task is not clearly grounded in the request or constraints")

    return issues


def validate_task_list(
    tasks: list[TaskModel],
    query: str,
    constraints: list[ConstraintModel],
) -> tuple[list[TaskModel], list[str]]:
    normalized = normalize_tasks(tasks, default_is_valid=False)
    request_keywords = extract_keywords(
        build_planner_request_text(query, constraints),
        STOPWORDS,
    )
    required_types = infer_required_task_types(query, constraints)
    present_types = {task.type for task in normalized}
    reviewed: list[TaskModel] = []
    summary: list[str] = []

    missing_types = [
        task_type for task_type in required_types if task_type not in present_types
    ]
    if missing_types:
        summary.append("Missing likely task types: " + ", ".join(missing_types) + ".")

    coverage = coverage_check(normalized, constraints)
    if coverage["uncovered_constraints"]:
        summary.append(
            "Potentially uncovered constraints: "
            + "; ".join(coverage["uncovered_constraints"][:3])
            + "."
        )

    for task in normalized:
        issues = task_issues(task, request_keywords=request_keywords)
        reviewed.append(
            task.model_copy(
                update={
                    "is_valid": not issues,
                    "validation_comment": "; ".join(issues) if issues else None,
                }
            )
        )

    return reviewed, summary


def review_feedback_text(tasks: list[TaskModel], summary: list[str]) -> str | None:
    lines = list(summary)
    for task in tasks:
        if task.validation_comment:
            lines.append(f"Task '{task.name}': {task.validation_comment}.")
    if not lines:
        return None
    return "\n".join(f"- {line}" for line in lines)
