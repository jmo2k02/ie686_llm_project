"""Prompt construction for the planner and planner-reviewer agents.

This module owns system prompts and user prompt formatting. It should only
prepare model inputs and should not call LLMs or mutate graph state.
"""

from __future__ import annotations

import json

from travelplanner.agents.planner.config import MAX_TASKS
from travelplanner.agents.planner.constants import ALLOWED_TASK_TYPES, TASK_TYPE_GUIDANCE
from travelplanner.agents.planner.heuristics import infer_required_task_types
from travelplanner.agents.planner.validation import validate_task_list
from travelplanner.schema.commonsense_constraints import get_constraints_for
from travelplanner.schema.system_state import ConstraintModel, TaskModel


def _allowed_task_type_string() -> str:
    return ", ".join(ALLOWED_TASK_TYPES)


def _task_type_schema_string() -> str:
    return "|".join(ALLOWED_TASK_TYPES)


def _format_planner_commonsense_constraints() -> str:
    return "\n".join(
        f"- {constraint.text}" for constraint in get_constraints_for("planner_agent")
    )


PLANNER_SYSTEM_PROMPT = f"""You are the TravelPlanner planner agent.

Turn the extracted constraints into an initial task list for downstream search agents.

Rules:
- Return JSON only.
- Use only the allowed task types: {_allowed_task_type_string()}.
- Each task must be grounded in the user request and extracted constraints.
- Keep the task list small and useful.
- Do not mark tasks as valid yet. The reviewer does that.
- ALWAYS make sure to respect ALL commonsense constraints.

Commonsense Constraints:
{_format_planner_commonsense_constraints()}

"""

REVIEWER_SYSTEM_PROMPT = """You are the TravelPlanner reviewer agent.

Review the planner's task list and return the reviewed task list.

Rules:
- Return JSON only.
- Validate whether each task is useful, specific, and grounded in the user request.
- Remove duplicates or merge obviously redundant tasks.
- Set is_valid to true only for tasks that are ready for downstream execution.
- If a task is invalid, explain why in validation_comment.
- If the planner missed an essential task, you may add it.
"""

def format_task_type_guide() -> str:
    return "\n".join(
        f"- {task_type}: {description}"
        for task_type, description in TASK_TYPE_GUIDANCE.items()
    )


def format_summary_lines(summary: list[str]) -> str:
    if not summary:
        return "- No deterministic issues detected."
    return "\n".join(f"- {item}" for item in summary)


def build_planner_prompt(
    query: str,
    constraints: list[ConstraintModel],
    review_feedback: str | None,
) -> str:
    required_types = infer_required_task_types(query, constraints)
    sections = [
        "Build a concise first-pass task list for the travel request below.",
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
        "Task type guide:",
        format_task_type_guide(),
        "",
        f"Likely required task types: {', '.join(required_types)}",
        f"Keep the list to at most {MAX_TASKS} tasks.",
    ]
    if review_feedback:
        sections.extend(
            [
                "",
                "Reviewer feedback from the previous attempt:",
                review_feedback.strip(),
            ]
        )
    sections.extend(
        [
            "",
            "Return strictly valid JSON with this shape:",
            '{"tasks": [{"name": "...", "type": "'
            + _task_type_schema_string()
            + '", "text": "...", "is_valid": false, "validation_comment": null}]}',
        ]
    )
    return "\n".join(sections)


def build_reviewer_prompt(
    query: str,
    constraints: list[ConstraintModel],
    proposed_task_list: list[TaskModel],
) -> str:
    deterministic_review, summary = validate_task_list(
        proposed_task_list,
        query,
        constraints,
    )
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
            "Task type guide:",
            format_task_type_guide(),
            "",
            "Proposed tasks JSON:",
            json.dumps(
                [task.model_dump() for task in proposed_task_list],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Deterministic validation preview:",
            json.dumps(
                [task.model_dump() for task in deterministic_review],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Deterministic summary:",
            format_summary_lines(summary),
            "",
            "Return strictly valid JSON with this shape:",
            '{"approved_task_list": [{"name": "...", "type": "'
            + _task_type_schema_string()
            + '", "text": "...", "is_valid": true, "validation_comment": null}], "review_summary": "..."}',
        ]
    )
