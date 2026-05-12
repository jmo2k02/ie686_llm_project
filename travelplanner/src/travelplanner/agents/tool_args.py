from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field


class FlightSearchArgs(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Natural-language description of the desired flight(s). Include "
                "origin, destination, date(s), and trip type if known — e.g. "
                "'Munich to Sydney from June 24 2026 until July 16 2026', "
                "'one-way LHR to JFK on 2026-09-10', or a multi-city itinerary "
                "like 'FRA → CDG on 2026-06-01, then CDG → BCN on 2026-06-05'. "
                "The tool will extract IATA codes, dates, and trip type via an "
                "LLM, then query Google Flights via SerpAPI."
            ),
        ),
    ]

      
class AttractionSearchArgs(BaseModel):
  query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Natural-language description for who and when the activity is for. "
                "Include destination, day, budget, traveller profile, time slot (optionally)," 
                "and previous activities that have been done (if any), to avoid " 
                "recommending the same thing again. Additionally, include any specific "
                "hints (if necessary) for the search. For example: "
                "'Find an activity for one person visiting Barcelona on Day 2 "
                "of their trip, with a budget of 80 EUR. They are interested in the local "
                "startup scene and want to blend remote work with exploration of "
                "creative and professional communities at a slow pace. Previously, "
                "they had visited a co-working space in Poblenou.'"
                "The tool will extract structured parameters via an LLM, select a "
                "matching traveller archetype via embedding similarity, generate a "
                "tailored activity, then resolve it to a real place via Google Maps "
                "(SerpAPI)."
            ),
        ),
    ]

    
class HotelSearchArgs(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Natural-language description of the desired hotel accommodation. "
                "Include location, check-in/check-out dates, number of guests, "
                "budget, and any required facilities — e.g. "
                "'Hotel in Barcelona, Spain from 2026-06-01 to 2026-06-05 for 2 guests, "
                "budget 150 EUR per night, need wifi and pool' or "
                "'Romantic hotel in Paris for honeymoon next week, max 300/night, "
                "must have spa and restaurant'. "
                "The tool will extract parameters via an LLM, then query LiteAPI "
                "for available hotels with live rates."
            ),
        ),
    ]


class RestaurantSearchArgs(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Natural-language description of the desired restaurant. "
                "Include city, cuisine type, budget, meal type, and any dietary "
                "restrictions — e.g. "
                "'Italian dinner in Barcelona for 2 people, medium budget' or "
                "'Vegan lunch spot in Berlin, cheap, near Alexanderplatz'. "
                "The tool will extract parameters via an LLM, then query Google "
                "Places for matching venues."
            ),
        ),
    ]


class CheckRouteTimingArgs(BaseModel):
    origin_address: Annotated[
        str,
        Field(min_length=1, description="Origin address, e.g. 'Munich Central Station'"),
    ]
    destination_address: Annotated[
        str,
        Field(min_length=1, description="Destination address, e.g. 'Marienplatz, Munich'"),
    ]
    travel_mode: Annotated[
        str,
        Field(default="drive", description="Travel mode: drive, transit, bicycling, walk"),
    ]


class WebSearchArgs(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "A specific, factual question. Not a general research topic. "
                "E.g. 'official opening hours for Sagrada Familia in May 2026', "
                "'contact email for the Louvre Museum Paris', "
                "'current train strike information for Munich to Berlin route on 2026-07-15'. "
                "The tool queries Tavily and returns a source-backed answer with URLs."
            ),
        ),
    ]


class BuildPlaceDistanceGraphArgs(BaseModel):
    stops: Annotated[
        list[dict[str, str]],
        Field(
            min_length=1,
            description=(
                "List of stop dicts, each with address and/or name keys. "
                "At least one stop is required. Used to build a place-distance graph."
            ),
        ),
    ]
    cluster_context: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Cluster preset hint: dense_urban, mixed, or sparse. "
                "Omit to let the routing agent infer automatically."
            ),
        ),
    ] = None


class DistanceBetweenPlacesArgs(BaseModel):
    graph: Annotated[
        dict[str, Any],
        Field(
            description=(
                "The place-distance graph dict previously returned "
                "by build_place_distance_graph. No API calls are made."
            ),
        ),
    ]
    from_place_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Place ID or name for the origin. Must exist in the graph."
            ),
        ),
    ]
    to_place_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Place ID or name for the destination. Must exist in the graph."
            ),
        ),
    ]


class ClosestPlacesToTargetArgs(BaseModel):
    graph: Annotated[
        dict,
        Field(
            description=(
                "The place-distance graph dict previously returned "
                "by build_place_distance_graph. No API calls are made."
            ),
        ),
    ]
    target_name: Annotated[
        str,
        Field(
            min_length=1,
            description="Name or ID of the target place in the graph.",
        ),
    ]
    candidate_names: Annotated[
        list[str],
        Field(
            min_length=1,
            description=(
                "List of candidate place names or IDs to rank against the target."
            ),
        ),
    ]
