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
