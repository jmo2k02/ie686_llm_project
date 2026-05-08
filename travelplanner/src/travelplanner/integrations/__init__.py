"""Public API for routing integrations.

**Orchestrator pattern:** build a graph once with
``build_place_graph_with_routing_agent``, retain ``result["graph"]``, then answer
follow-ups with ``distance_between_places`` / ``closest_places_to_target`` (no extra
Google calls). See ``ORCHESTRATOR_ROUTING_TOOL_SCHEMAS`` for tool JSON.

Direct agent API::

    from travelplanner.agents.routing_agent import run_routing_agent

    stops = [
        {"address": "Dam 1, Amsterdam", "name": "Start"},
        {"address": "Museumplein 6, Amsterdam", "name": "End"},
    ]
    artifact = run_routing_agent(stops, api_key="YOUR_API_KEY")
    # artifact.content has clusters, hub_hub_legs, edges
"""

from travelplanner.integrations.google_routes import (
    TravelMode,
    compute_route_plan,
    geocode_address_to_lat_lng,
    resolve_travel_mode,
    route_plan_to_jsonable,
)
from travelplanner.integrations.google_route_matrix import (
    compute_all_travel_modes_hub_matrices,
)
from travelplanner.integrations.place_distance_graph import (
    ClusterContext,
    MapPlaceInputModel,
    PlaceDistanceGraphBuildConfig,
    PlaceDistanceGraphModel,
    build_place_distance_graph,
    parse_places_input_payload,
    place_distance_graph_config_for_context,
)

# Task constants (for workflow integration)
from travelplanner.integrations.routing_contracts import (
    ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH,
    ARTIFACT_TYPE_ROUTE_TIMING,
    ROUTING_CHECK_TASK_TYPE,
)
from travelplanner.integrations.routing_check_agent import (
    RoutingCheckAgentState,
    make_graph as make_routing_check_graph,
)
from travelplanner.integrations.routing_execution import execute_routing_check_task
from travelplanner.integrations.routing_agent_tools import (
    ORCHESTRATOR_ROUTING_TOOL_SCHEMAS,
    ROUTING_TOOL_SCHEMAS,
    build_distance_graph_from_stops,
    build_place_graph_with_routing_agent,
    closest_places_to_target,
    distance_between_places,
    route_one_leg,
)

__all__ = [
    # Core building blocks
    "build_place_distance_graph",
    "parse_places_input_payload",
    "place_distance_graph_config_for_context",
    "compute_route_plan",
    "geocode_address_to_lat_lng",
    "compute_all_travel_modes_hub_matrices",
    "resolve_travel_mode",
    "route_plan_to_jsonable",
    # Types
    "TravelMode",
    "ClusterContext",
    "MapPlaceInputModel",
    "PlaceDistanceGraphBuildConfig",
    "PlaceDistanceGraphModel",
    # Task execution (for workflow integration)
    "execute_routing_check_task",
    "ROUTING_CHECK_TASK_TYPE",
    "ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH",
    "ARTIFACT_TYPE_ROUTE_TIMING",
    # Agent-callable routing (structured dict results)
    "build_place_graph_with_routing_agent",
    "closest_places_to_target",
    "route_one_leg",
    "build_distance_graph_from_stops",
    "distance_between_places",
    "ORCHESTRATOR_ROUTING_TOOL_SCHEMAS",
    "ROUTING_TOOL_SCHEMAS",
]
