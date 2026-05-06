"""Helpers for building planner-specific request context."""

from __future__ import annotations

from travelplanner.schema.system_state import ConstraintModel


def build_planner_request_text(
    query: str,
    constraints: list[ConstraintModel],
) -> str:
    parts = [
        query.strip(),
        *[constraint.text for constraint in constraints if not constraint.user_skipped],
    ]
    return "\n".join(part for part in parts if part)
