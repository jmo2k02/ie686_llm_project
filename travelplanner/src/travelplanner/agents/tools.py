from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from travelplanner.agents.subagent_tools.flight_search import make_search_flights_tool, SEARCH_FLIGHTS_DESCRIPTION
from travelplanner.agents.tool_args import FlightSearchArgs
from travelplanner.config import get_setting

_DEFAULT_MODEL = "openrouter:minimax/minimax-m2.5"


def make_subagent_tools(
    model_name: str | None = None,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> list[BaseTool]:
    """Wire sub-agent callables into ``StructuredTool``s for the calling agent.

    This module is wiring only — the actual search/format logic lives in
    ``travelplanner.agents.subagent_tools``. Each tool here just binds a
    pydantic args schema and a description to a pre-built callable.
    """
    model = model_name or str(
        get_setting("models.agents.flight_search.model_name", _DEFAULT_MODEL)
    )

    return [
        StructuredTool.from_function(
            func=make_search_flights_tool(model, temperature, task_ref),
            name="search_flights",
            description=SEARCH_FLIGHTS_DESCRIPTION,
            args_schema=FlightSearchArgs,
            handle_validation_error=True,
        ),
        ## Hier eure Tools einfuegen!!
    ]
