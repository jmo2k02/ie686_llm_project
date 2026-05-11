from __future__ import annotations

from typing import Annotated

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