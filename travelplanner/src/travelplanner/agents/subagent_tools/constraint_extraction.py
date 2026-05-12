"""Callable wrapper around the constraint-iteration agent's pipeline graph.

Exposes ``make_extract_constraints_tool``: a factory that closure-binds
model/temperature/task_ref and returns a single-arg ``(query: str) -> str``
function suitable for wrapping in a ``StructuredTool``. Uses
``make_pipeline_graph`` (non-interactive) so it can be called from within
the execution agent without requiring human-in-the-loop interrupts.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.constraint_iteration_agent import (
    ConstraintIterationState,
    make_pipeline_graph,
)


EXTRACT_CONSTRAINTS_DESCRIPTION = (
    "Extract and validate structured travel constraints from a natural-language "
    "trip description. Parses destination, origin, travel dates, travelers, "
    "budget, accommodation preferences, transport mode, and interests. Also "
    "checks for commonsense violations (e.g. past dates, end before start). "
    "Returns a text summary of all extracted constraints plus any warnings. "
    "Use this when you need to verify that a specific part of the user's "
    "request was correctly understood — e.g. to confirm a budget figure, "
    "re-parse ambiguous date phrasing, or check traveler counts before "
    "booking slots. Do NOT use this instead of the dedicated search tools "
    "(flights, hotels, restaurants, attractions) — those return real booking "
    "data; this tool only extracts and validates the request parameters. "
    "Returns 'Error: ...' on failure — read it and retry with a clearer query."
)


def make_extract_constraints_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return an ``extract_constraints(query)`` callable bound to graph + config."""
    constraint_graph = make_pipeline_graph()

    def extract_constraints(query: str) -> str:
        try:
            state = ConstraintIterationState(
                query=query,
                constraint_list=[],
                model_name=model_name,
                temperature=temperature,
            )
            result = constraint_graph.invoke(state)
        except Exception as exc:
            return f"Error: {exc}"

        hard = result.get("hard_constraints", [])
        nc = result.get("normalized_constraints")
        violations = result.get("violations", [])

        if not hard and nc is None:
            return "Error: constraint extraction produced no output"

        lines: list[str] = []

        if nc is not None:
            nc_dict = nc.model_dump(exclude_none=True)
            if nc_dict:
                lines.append("Extracted constraints (normalized):")
                for key, val in nc_dict.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            lines.append(f"  {key}.{sub_key}: {sub_val}")
                    else:
                        lines.append(f"  {key}: {val}")
        elif hard:
            lines.append("Extracted constraints:")
            for c in hard:
                if not c.user_skipped:
                    lines.append(f"  - {c.text}")

        if violations:
            lines.append(f"\nWarnings ({len(violations)} violation(s) detected):")
            for v in violations:
                lines.append(f"  - {v.violated_constraint}")
                lines.append(f"    Reason: {v.explanation}")
                for suggestion in v.suggestions:
                    lines.append(f"    Suggestion: {suggestion}")
        else:
            lines.append("\nNo constraint violations detected.")

        return "\n".join(lines)

    return extract_constraints
