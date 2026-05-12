"""Callable wrappers around routing helpers from routing_agent_tools.

Wraps four routing tools for the main execution agent:
- check_route_timing: wraps route_one_leg for single OD timing
- build_place_distance_graph: wraps build_place_graph_with_routing_agent
- distance_between_places: query distance between two places in a graph
- closest_places_to_target: find closest place from a list in a graph

Each function returns a JSON-friendly dict with ok/stage/error on failure.
"""

from __future__ import annotations

from typing import Any, Callable

from travelplanner.integrations.routing_agent_tools import (
    build_place_graph_with_routing_agent,
    closest_places_to_target as _closest,
    distance_between_places as _distance,
    route_one_leg,
)


# ---------------------------------------------------------------------------
# Tool descriptions — mirror the style of flight_search.py / hotel_search.py
# ---------------------------------------------------------------------------

CHECK_ROUTE_TIMING_DESCRIPTION = """Check travel time and distance for a single origin-destination pair via Google Routes API.

**What it does**: Queries the Google Routes API for travel timing between two addresses. Pass origin_address, destination_address, and travel_mode. Returns a dict with ok=true, distance_km (float), duration_min (float), and route_summary (human-readable description of the route) on success. Returns ok=false with stage and error fields on failure.

**When to call it**: Use for a single timing query when you do not need to build a full multi-stop graph. For example: check how long it takes to get from Munich Hauptbahnhof to Munich Airport by transit, or check driving time from Hotel Barcelona to Sagrada Familia.

**Args**:
  - origin_address: str — Full street address or place name of the starting point. Example: "Munich Hauptbahnhof" or "Marienplatz 1, Munich, Germany"
  - destination_address: str — Full street address or place name of the destination. Example: "Munich Airport" or "Terminal 2, Munich Airport, Germany"
  - travel_mode: str — Mode of transport. Options: "drive" (car/taxi), "transit" (train, bus, subway), "bicycling" (bike), "walk" (walking). Default: "drive".

**Returns**: Dict with keys:
  - ok: bool — true on success, false on failure
  - distance_km: float — distance in kilometers
  - duration_min: float — travel time in minutes
  - route_summary: str — human-readable description of the route (e.g., "via A9 highway, 40 km")
  - stage: str — present only on failure, indicates at which step the failure occurred
  - error: str — present only on failure, contains the error message

**On failure**: If ok=false, read the stage and error fields. Common causes: addresses not found by Google Maps (try more specific addresses), travel_mode not available for that route (try "drive" instead of "transit"), API timeout (retry once). If error persists, treat the route as unavailable.

**Example query**: check_route_timing(origin_address="Munich Hauptbahnhof", destination_address="Munich Airport", travel_mode="transit")
"""

BUILD_PLACE_DISTANCE_GRAPH_DESCRIPTION = """Build a place-distance graph for multi-stop trip routing via Google Routes API.

**What it does**: Takes a list of stops (each with address and optional name) and builds a complete distance/duration matrix between all pairs via Google Routes API. The cluster_context parameter helps the API optimize for urban density (dense_urban = city center, mixed = suburb-to-suburb, sparse = highway between cities). Returns a dict with ok=true, graph (a dict mapping place_id to {name, address, distances to all other places}), and decided_cluster_context (the API's chosen context). Call this ONCE per trip segment (e.g., once for all stops in Barcelona, once for all stops in Paris), then reuse the graph dict for multiple distance queries via distance_between_places and closest_places_to_target.

**When to call it**: At the start of planning a multi-stop trip segment. Call once, then make multiple distance queries against the returned graph without calling this again. For example: build graph for all Barcelona stops, then query distances between each pair.

**Args**:
  - stops: list[dict] — List of stop dictionaries. Each dict must have: address (str, full address or place name). Optional: name (str, human-readable label for this stop). Example: [{"address": "Sagrada Familia, Barcelona"}, {"address": "Park Guell, Barcelona"}, {"address": "Las Ramblas, Barcelona"}]
  - cluster_context: str — Hint about the urban density of the trip segment. Options: "dense_urban" (city center with many short hops), "mixed" (combination of city and suburbs), "sparse" (long distances, highway driving). Default: "mixed". The API may override this based on actual distances found.

**Returns**: Dict with keys:
  - ok: bool — true on success, false on failure
  - graph: dict — place_id (str, "place_0", "place_1", etc.) -> {name: str or None, address: str, distances: dict of {place_id -> {distance_km: float, duration_min: float, summary: str}}}
  - decided_cluster_context: str — the cluster context actually used (may differ from requested)
  - stage: str — present only on failure
  - error: str — present only on failure

**On failure**: If ok=false, read the stage and error. Common causes: some addresses not found (remove problematic stops and retry), API timeout (retry once). If error persists, you may need to use check_route_timing for individual legs instead.

**CRITICAL**: Call this ONCE per trip segment, then reuse the graph. Do NOT call it again for the same segment unless the stop list changes.

**Example query**: build_place_distance_graph(stops=[{"address": "Sagrada Familia, Barcelona"}, {"address": "Park Guell, Barcelona"}], cluster_context="dense_urban")
"""

DISTANCE_BETWEEN_PLACES_DESCRIPTION = """Query the distance and travel time between two places in a pre-built place graph.

**What it does**: Takes a graph (from build_place_distance_graph output), a from_place_id, and a to_place_id, and returns the pre-computed distance and duration between those two places. The graph contains distances between all pairs of places that were built in the previous step.

**When to call it**: After calling build_place_distance_graph, use this to get the distance between any two stops in the graph. For example: after building a graph for Barcelona, query the distance from "place_0" (Sagrada Familia) to "place_1" (Park Guell).

**Args**:
  - graph: dict — The graph dict returned by build_place_distance_graph. Must not be modified after build_place_distance_graph call.
  - from_place_id: str — The place ID of the origin, e.g., "place_0", "place_1". Must exist in the graph.
  - to_place_id: str — The place ID of the destination, e.g., "place_2", "place_3". Must exist in the graph.

**Returns**: Dict with keys:
  - ok: bool — true if both place IDs exist in the graph, false if either is not found
  - distance_km: float — distance in kilometers between the two places
  - duration_min: float — travel time in minutes
  - summary: str — human-readable route summary
  - stage: str — present only on failure
  - error: str — present only on failure

**On failure**: If ok=false, one or both place IDs were not found in the graph. Check that you are using the correct place_id values (they are "place_0", "place_1", etc. in the order you passed stops to build_place_distance_graph). If the graph was not built, use check_route_timing instead.

**Example query**: distance_between_places(graph=my_graph, from_place_id="place_0", to_place_id="place_1")
"""

CLOSEST_PLACES_TO_TARGET_DESCRIPTION = """Find the closest place to a target among candidates in a pre-built place graph.

**What it does**: Takes a graph (from build_place_distance_graph output), a target_name (e.g., "train station"), and a list of candidate place IDs. Returns the candidate that is closest to the target based on pre-computed distances in the graph.

**When to call it**: When you need to find which stop is closest to a specific point of interest. For example: after building a graph for Munich stops, find which hotel is closest to the main train station.

**Args**:
  - graph: dict — The graph dict returned by build_place_distance_graph.
  - target_name: str — Human-readable description of the target, e.g., "train station", "airport", "city center". This is for display in the result only — the function finds which candidate is closest in the graph.
  - candidate_names: list[str] — List of place IDs to consider as candidates, e.g., ["place_0", "place_2", "place_3"]. These must exist in the graph.

**Returns**: Dict with keys:
  - ok: bool — true on success, false if no candidates found or other error
  - winner: dict — the winning candidate with keys: place_id (str), name (str or None), address (str)
  - distance_km: float — distance in km from winner to the target
  - duration_min: float — travel time in minutes
  - summary: str — human-readable summary
  - stage: str — present only on failure
  - error: str — present only on failure

**On failure**: If ok=false, read stage and error. Common causes: no candidates found in graph (check place_ids are correct), API error (retry once). If error persists, use check_route_timing for individual checks instead.

**Example query**: closest_places_to_target(graph=my_graph, target_name="train station", candidate_names=["place_0", "place_2", "place_3"])
"""


# ---------------------------------------------------------------------------
# Core routing tool callables
# ---------------------------------------------------------------------------


def check_route_timing(
    origin_address: str,
    destination_address: str,
    *,
    travel_mode: str = "drive",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return route timing for one OD pair."""
    import os as _os

    key = api_key or _os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        return {"ok": False, "stage": "api_key", "error": "Missing Google Maps API key"}
    return route_one_leg(
        origin_address=origin_address,
        destination_address=destination_address,
        travel_mode=travel_mode,
        api_key=key,
    )


def build_place_distance_graph(
    stops: list[dict[str, str]],
    *,
    cluster_context: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build a place graph from stops."""
    import os as _os

    key = api_key or _os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        return {"ok": False, "stage": "api_key", "error": "Missing Google Maps API key"}
    if not stops:
        return {"ok": False, "stage": "validate_input", "error": "stops must be non-empty"}
    kwargs: dict[str, Any] = {"stops": stops, "api_key": key}
    if cluster_context:
        kwargs["cluster_context"] = cluster_context
    return build_place_graph_with_routing_agent(**kwargs)


def distance_between_places(
    graph: dict[str, Any],
    from_place_id: str,
    to_place_id: str,
) -> dict[str, Any]:
    """Query distance between two places in a graph."""
    return _distance(graph=graph, from_place=from_place_id, to_place=to_place_id)


def closest_places_to_target(
    graph: dict[str, Any],
    target_name: str,
    candidate_names: list[str],
) -> dict[str, Any]:
    """Find closest place to target in a graph."""
    return _closest(graph=graph, target_name=target_name, candidate_names=candidate_names)


# ---------------------------------------------------------------------------
# Registration-friendly callable factories
# These match the make_*/tool(model_name, temperature, task_ref) pattern
# used by flight_search.py / hotel_search.py for the execution-agent registry.
# ---------------------------------------------------------------------------


def make_check_route_timing_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[..., dict[str, Any]]:
    """Return a ``check_route_timing`` callable bound to config.

    The returned callable accepts origin_address, destination_address,
    travel_mode (kwonly), and api_key (kwonly) and returns a dict.
    """

    def tool_call(
        origin_address: str,
        destination_address: str,
        *,
        travel_mode: str = "drive",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        return check_route_timing(
            origin_address=origin_address,
            destination_address=destination_address,
            travel_mode=travel_mode,
            api_key=api_key,
        )

    return tool_call


def make_build_place_distance_graph_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[..., dict[str, Any]]:
    """Return a ``build_place_distance_graph`` callable bound to config."""

    def tool_call(
        stops: list[dict[str, str]],
        *,
        cluster_context: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        return build_place_distance_graph(
            stops=stops,
            cluster_context=cluster_context,
            api_key=api_key,
        )

    return tool_call


def make_distance_between_places_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[..., dict[str, Any]]:
    """Return a ``distance_between_places`` callable bound to config."""

    def tool_call(
        graph: dict[str, Any],
        from_place_id: str,
        to_place_id: str,
    ) -> dict[str, Any]:
        return distance_between_places(
            graph=graph,
            from_place_id=from_place_id,
            to_place_id=to_place_id,
        )

    return tool_call


def make_closest_places_to_target_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[..., dict[str, Any]]:
    """Return a ``closest_places_to_target`` callable bound to config."""

    def tool_call(
        graph: dict[str, Any],
        target_name: str,
        candidate_names: list[str],
    ) -> dict[str, Any]:
        return closest_places_to_target(
            graph=graph,
            target_name=target_name,
            candidate_names=candidate_names,
        )

    return tool_call
