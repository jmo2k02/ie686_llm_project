from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from travelplanner.agents.subagent_tools.flight_search import make_search_flights_tool, SEARCH_FLIGHTS_DESCRIPTION
from travelplanner.agents.subagent_tools.hotel_search import make_search_hotels_tool, SEARCH_HOTELS_DESCRIPTION
from travelplanner.agents.subagent_tools.restaurant_search import make_search_restaurants_tool, SEARCH_RESTAURANTS_DESCRIPTION
from travelplanner.agents.subagent_tools.general_web_search import make_search_web_tool, SEARCH_WEB_DESCRIPTION
from travelplanner.agents.subagent_tools.routing_check import (
    check_route_timing,
    build_place_distance_graph,
    distance_between_places,
    closest_places_to_target,
    CHECK_ROUTE_TIMING_DESCRIPTION,
    BUILD_PLACE_DISTANCE_GRAPH_DESCRIPTION,
    DISTANCE_BETWEEN_PLACES_DESCRIPTION,
    CLOSEST_PLACES_TO_TARGET_DESCRIPTION,
)
from travelplanner.agents.subagent_tools.attraction_search import make_attraction_search_tool, SEARCH_ATTRACTIONS_DESCRIPTION
from travelplanner.agents.tool_args import (
    FlightSearchArgs,
    HotelSearchArgs,
    RestaurantSearchArgs,
    AttractionSearchArgs,
    WebSearchArgs,
    CheckRouteTimingArgs, BuildPlaceDistanceGraphArgs,
    DistanceBetweenPlacesArgs, ClosestPlacesToTargetArgs,
)
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
        StructuredTool.from_function(
            func=make_search_hotels_tool(model, temperature, task_ref),
            name="search_hotels",
            description=SEARCH_HOTELS_DESCRIPTION,
            args_schema=HotelSearchArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=make_search_restaurants_tool(model, temperature, task_ref),
            name="search_restaurants",
            description=SEARCH_RESTAURANTS_DESCRIPTION,
            args_schema=RestaurantSearchArgs,
            handle_validation_error=True,
        ),
        ## Hier eure Tools einfuegen!!
        StructuredTool.from_function(
            func=make_attraction_search_tool(model, temperature, task_ref),
            name="search_attractions",
            description=SEARCH_ATTRACTIONS_DESCRIPTION,
            args_schema=AttractionSearchArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=make_search_web_tool(model, temperature, task_ref),
            name="search_web",
            description=SEARCH_WEB_DESCRIPTION,
            args_schema=WebSearchArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=check_route_timing,
            name="check_route_timing",
            description=CHECK_ROUTE_TIMING_DESCRIPTION,
            args_schema=CheckRouteTimingArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=build_place_distance_graph,
            name="build_place_distance_graph",
            description=BUILD_PLACE_DISTANCE_GRAPH_DESCRIPTION,
            args_schema=BuildPlaceDistanceGraphArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=distance_between_places,
            name="distance_between_places",
            description=DISTANCE_BETWEEN_PLACES_DESCRIPTION,
            args_schema=DistanceBetweenPlacesArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=closest_places_to_target,
            name="closest_places_to_target",
            description=CLOSEST_PLACES_TO_TARGET_DESCRIPTION,
            args_schema=ClosestPlacesToTargetArgs,
            handle_validation_error=True,
        ),
    ]
