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
