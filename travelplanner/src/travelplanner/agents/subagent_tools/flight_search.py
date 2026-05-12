"""Callable wrapper around the flight-search agent graph.

Exposes ``make_search_flights``: a factory that closure-binds model/temperature
/task_ref and returns a single-arg ``(query: str) -> str`` function suitable
for wrapping in a ``StructuredTool``. The graph (extract params → query
SerpAPI → normalize) is reused as-is so this module holds no flight-search
business logic — only the adapter that turns a natural-language query into
the graph's task-shaped input and renders the resulting artifact.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.flight_search_agent import (
    FlightSearchAgentState,
    make_graph as make_flight_search_graph,
)
from travelplanner.agents.subagent_tools.utils import summarize_flight_artifact
from travelplanner.schema.system_state import TaskModel


SEARCH_FLIGHTS_DESCRIPTION = (
    "Search flights via Google Flights (SerpAPI). Accepts a natural-language "
    "flight request (origin, destination, date(s), trip type). The tool "
    "extracts IATA codes, dates, and trip type with an LLM, then queries the "
    "provider for round-trip and one-way itineraries. Returns a "
    "textual summary of the cheapest selected option per direction (price, "
    "duration, legs, layovers) plus any price insights and a Google Flights "
    "URL the user can open to verify and book. Returns 'Error: ...' "
    "on failure — read it and retry with a clearer query."
)


def make_search_flights_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return a ``search_flights(query)`` callable bound to graph + config."""
    flight_graph = make_flight_search_graph()

    def search_flights(query: str) -> str:
        try:
            task = TaskModel(
                name=task_ref,
                type="flight",
                text=query,
                is_valid=True,
            )
            agent_state = FlightSearchAgentState(
                query=query,
                model_name=model_name,
                temperature=temperature,
                task_list=[task],
                agent_artifacts={},
            )
            result = flight_graph.invoke(agent_state)
        except Exception as exc:
            return f"Error: {exc}"

        artifacts = result.get("agent_artifacts", {}).get("flight_search_agent", [])
        if not artifacts:
            return "Error: flight search produced no artifact"
        return "\n\n".join(summarize_flight_artifact(a) for a in artifacts)

    return search_flights
