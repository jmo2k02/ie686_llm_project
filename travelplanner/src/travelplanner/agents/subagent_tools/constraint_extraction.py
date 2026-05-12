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
from travelplanner.agents.subagent_tools.utils import summarize_constraint_artifact


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

        artifacts = result.get("agent_artifacts", {}).get("constraint_agent", [])
        if not artifacts:
            return "Error: constraint extraction produced no artifact"

        violations = result.get("violations", [])
        return "\n\n".join(
            summarize_constraint_artifact(a, violations) for a in artifacts
        )

    return extract_constraints
