"""Callable wrapper around the restaurant-search agent graph.

Exposes ``make_search_restaurants``: a factory that closure-binds model/temperature
/task_ref and returns a single-arg ``(query: str) -> str`` function suitable
for wrapping in a ``StructuredTool``. The graph (extract params → query
Google Places → normalize) is reused as-is so this module holds no
restaurant-search business logic — only the adapter that turns a
natural-language query into the graph's task-shaped input and renders the
resulting artifact.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.restaurant_search_agent import (
    RestaurantSearchAgentState,
    make_graph,
)
from travelplanner.agents.subagent_tools.utils import summarize_restaurant_artifact
from travelplanner.schema.system_state import TaskModel


SEARCH_RESTAURANTS_DESCRIPTION = (
    "Search restaurants via Google Places API. Accepts a natural-language "
    "restaurant request (city, cuisine, budget, meal type, dietary "
    "restrictions). The tool extracts structured parameters with an LLM, "
    "then queries Google Places for matching venues. Returns a textual summary "
    "of the selected restaurant (name, rating, address, price level, opening "
    "hours). Returns 'Error: ...' on failure — read it and retry with a "
    "clearer query."
)


def make_search_restaurants_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return a ``search_restaurants(query)`` callable bound to graph + config."""
    restaurant_graph = make_graph()

    def search_restaurants(query: str) -> str:
        try:
            task = TaskModel(
                name=task_ref,
                type="restaurant",
                text=query,
                is_valid=True,
            )
            agent_state = RestaurantSearchAgentState(
                query=query,
                model_name=model_name,
                temperature=temperature,
                task_list=[task],
            )
            result = restaurant_graph.invoke(agent_state)
        except Exception as exc:
            return f"Error: {exc}"

        artifacts = result.get("agent_artifacts", {}).get("restaurant_search", [])
        if not artifacts:
            return "Error: restaurant search produced no artifact"
        return "\n\n".join(summarize_restaurant_artifact(a) for a in artifacts)

    return search_restaurants
