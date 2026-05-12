"""Callable wrapper around the attraction-search agent graph.

Exposes ``make_attraction_search_tool``: a factory that closure-binds
model/temperature/task_ref and returns a single-arg ``(query: str) -> str``
function suitable for wrapping in a ``StructuredTool``. The graph (extract
params → select archetype via embedding similarity against the experience pool
→ generate activity description → query SerpAPI for candidates → select place)
is reused as-is so this module holds no attraction-search business logic —
only the adapter that turns a natural-language query into the graph's
task-shaped input and renders the resulting artifact.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.attraction_search_agent import (
    AttractionSearchAgentState,
    make_graph as make_attraction_search_graph,
)
from travelplanner.agents.subagent_tools.utils import summarize_attraction_artifact
from travelplanner.schema.system_state import TaskModel


SEARCH_ATTRACTIONS_DESCRIPTION = (
    "Search for activities and attractions at a destination (SerpAPI + LLM). "
    "Accepts a natural-language request describing the destination, day, budget, "
    "and traveller profile. Optionally include a preferred time slot (morning, "
    "afternoon, or evening) in the query — if present, it will be strictly "
    "respected. The tool extracts structured parameters with an LLM, selects a "
    "matching traveller archetype via embedding similarity, generates a tailored "
    "activity, then resolves it to a real place via Google Maps (SerpAPI). "
    "Returns a textual summary of the selected activity (title, description, local "
    "touchpoint, duration, estimated price, place details) and a Google Maps URL "
    "the user can open to verify the result. Returns 'Error: ...' "
    "on failure — read it and retry with a clearer query."
)


def make_attraction_search_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return a ``search_attractions(query)`` callable bound to graph + config."""
    attraction_graph = make_attraction_search_graph()

    def search_attractions(query: str) -> str:
        try:
            task = TaskModel(
                name=task_ref,
                type="attraction",
                text=query,
                is_valid=True,
            )
            agent_state = AttractionSearchAgentState(
                query=query,
                model_name=model_name,
                temperature=temperature,
                task_list=[task],
                agent_artifacts={},
            )
            result = attraction_graph.invoke(agent_state)
        except Exception as exc:
            return f"Error: {exc}"

        artifacts = result.get("agent_artifacts", {}).get("attraction_search_agent", [])
        if not artifacts:
            return "Error: attraction search produced no artifact"
        return "\n\n".join(summarize_attraction_artifact(a) for a in artifacts)

    return search_attractions
