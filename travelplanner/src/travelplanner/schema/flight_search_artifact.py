from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class FlightSegmentParams(BaseModel):
    departure_id: Annotated[str, Field(description="IATA airport code")]
    arrival_id: Annotated[str, Field(description="IATA airport code")]
    outbound_date: Annotated[str, Field(description="YYYY-MM-DD")]


class FlightParamsModel(BaseModel):
    trip_type: Annotated[
        Literal[1, 2],
        Field(description="1=round trip, 2=one way"),
    ]
    segments: Annotated[
        list[FlightSegmentParams],
        Field(description="One segment for the departure→arrival pair"),
    ]
    return_date: Annotated[
        str | None,
        Field(description="YYYY-MM-DD, only set when trip_type=1"),
    ] = None
    adults: int = 1
    currency: str = "EUR"


class FlightAirportModel(BaseModel):
    name: Annotated[str, Field(description="Full airport name")]
    id: Annotated[str, Field(description="IATA airport code, e.g. 'FRA'")]
    time: Annotated[str, Field(description="Local departure/arrival time, e.g. '2026-06-10 07:00'")]


class FlightLegModel(BaseModel):
    departure_airport: FlightAirportModel
    arrival_airport: FlightAirportModel
    duration_minutes: int
    airline: str
    flight_number: str
    airplane: str | None = None
    travel_class: str
    legroom: str | None = None
    extensions: list[str] = Field(default_factory=list)


class FlightLayoverModel(BaseModel):
    name: str
    id: str
    duration_minutes: int


class FlightOptionModel(BaseModel):
    legs: list[FlightLegModel]
    layovers: list[FlightLayoverModel] = Field(default_factory=list)
    total_duration_minutes: int
    price: float
    currency: str
    type: Annotated[str, Field(description="'Round trip' or 'One way'")]
    carbon_emissions_kg: int | None = None
    departure_token: str | None = None


class FlightPriceInsightsModel(BaseModel):
    lowest_price: float | None = None
    price_level: Annotated[
        str | None, Field(description="'low', 'typical', or 'high'")
    ] = None
    typical_price_range: list[float] = Field(default_factory=list)


class FlightSearchErrorModel(BaseModel):
    code: Literal[
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "unknown_error",
    ]
    message: str


class FlightSearchArtifactContentModel(BaseModel):
    task_ref: str
    status: Literal["success", "partial", "failed", "skipped"]
    provider: Literal["serpapi_google_flights"]
    departure_id: Annotated[str, Field(description="IATA code, resolved by LLM")]
    arrival_id: Annotated[str, Field(description="IATA code, resolved by LLM")]
    outbound_date: Annotated[str, Field(description="YYYY-MM-DD")]
    return_date: Annotated[str | None, Field(description="YYYY-MM-DD, None for one-way")] = None
    adults: int
    currency: str
    selected_flights: Annotated[
        list[FlightOptionModel],
        Field(
            default_factory=list,
            description=(
                "Committed flight choice for the orchestrator: "
                "one bundled entry for round-trip (type 1) or one entry for one-way (type 2). "
                "Always the cheapest available option."
            ),
        ),
    ] = Field(default_factory=list)
    best_flights: list[FlightOptionModel] = Field(default_factory=list)
    other_flights: list[FlightOptionModel] = Field(default_factory=list)
    return_flights: Annotated[
        list[FlightOptionModel],
        Field(
            default_factory=list,
            description="Return-leg options for round-trip (type 1); fetched via a second one-way call for details only — price is already bundled in selected_flights[0].",
        ),
    ] = Field(default_factory=list)
    price_insights: FlightPriceInsightsModel | None = None
    google_flights_url: Annotated[
        str | None,
        Field(description="Google Flights search URL from SerpAPI search_metadata — lets the user verify and book"),
    ] = None
    errors: list[FlightSearchErrorModel] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
