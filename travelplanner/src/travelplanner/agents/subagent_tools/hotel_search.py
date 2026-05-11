"""Callable wrapper around the hotel-search agent graph.

Exposes ``make_search_hotels``: a factory that closure-binds model/temperature
/task_ref and returns a single-arg ``(query: str) -> str`` function suitable
for wrapping in a ``StructuredTool``. The graph (parse params → query
LiteAPI → normalize) is reused as-is so this module holds no hotel-search
business logic — only the adapter that turns a natural-language query into
the graph's task-shaped input and renders the resulting artifact.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.hotel_search_agent import (
    IntelligentHotelSearchState,
    make_intelligent_hotel_graph,
)
from travelplanner.agents.subagent_tools.utils import summarize_hotel_artifact


SEARCH_HOTELS_DESCRIPTION = (
    "Search hotels via LiteAPI. Accepts a natural-language hotel request "
    "(location, dates, budget, facilities). The tool extracts structured "
    "parameters with an LLM, then queries the LiteAPI /hotels/rates endpoint "
    "for available hotels with live prices. Returns a textual summary of the "
    "top ranked hotel options (name, price per night, rating, facilities, area). "
    "Returns 'Error: ...' on failure — read it and retry with a clearer query."
)


def make_search_hotels_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return a ``search_hotels(query)`` callable bound to graph + config."""
    hotel_graph = make_intelligent_hotel_graph()

    def search_hotels(query: str) -> str:
        try:
            agent_state = IntelligentHotelSearchState(
                query=query,
                model_name=model_name,
            )
            result = hotel_graph.invoke(agent_state)
        except Exception as exc:
            return f"Error: {exc}"

        artifact = result.get("hotel_artifact")
        if artifact is None:
            return "Error: hotel search produced no artifact"
        return summarize_hotel_artifact(artifact)

    return search_hotels
