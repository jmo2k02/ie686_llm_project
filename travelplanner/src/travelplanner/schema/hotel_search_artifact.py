from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class HotelSearchCoordinatesModel(BaseModel):
    """Geographic coordinates for hotel search."""

    latitude: Annotated[float, Field(description="Latitude coordinate")]
    longitude: Annotated[float, Field(description="Longitude coordinate")]


class HotelSearchParametersModel(BaseModel):
    """Search parameters used for hotel query."""

    location: Annotated[str, Field(description="Location string (e.g., 'Barcelona, Spain')")]
    coordinates: Annotated[
        HotelSearchCoordinatesModel | None,
        Field(default=None, description="Geocoded coordinates"),
    ]
    check_in_date: Annotated[str, Field(description="Check-in date (YYYY-MM-DD)")]
    check_out_date: Annotated[str, Field(description="Check-out date (YYYY-MM-DD)")]
    nights: Annotated[int, Field(description="Number of nights", ge=1)]
    budget_max: Annotated[float, Field(description="Maximum budget per night", ge=0)]
    guest_count: Annotated[int, Field(description="Number of guests", ge=1)]
    rooms: Annotated[int, Field(default=1, description="Number of rooms", ge=1)]


class HotelOptionModel(BaseModel):
    """A single hotel option in the shortlist."""

    search_result_id: Annotated[
        str, Field(description="Booking/offer ID from provider")
    ]
    accommodation_id: Annotated[str, Field(description="Hotel ID from provider")]
    name: Annotated[str, Field(description="Hotel name")]
    nightly_rate: Annotated[float, Field(description="Price per night", ge=0)]
    total_cost: Annotated[float, Field(description="Total cost for stay", ge=0)]
    currency: Annotated[str, Field(description="Currency code (e.g., 'EUR', 'USD')")]
    area: Annotated[
        str | None, Field(default=None, description="Neighborhood/district")
    ]
    address: Annotated[str | None, Field(default=None, description="Full address")]
    facilities: Annotated[
        list[str], Field(default_factory=list, description="List of hotel facilities (from hotelFacilities)")
    ]
    rating: Annotated[float, Field(description="Star or review rating (0-10)", ge=0, le=10)]
    reviews: Annotated[int, Field(default=0, description="Number of reviews", ge=0)]
    check_in_time: Annotated[
        str, Field(default="15:00", description="Check-in time (HH:MM)")
    ]
    check_out_time: Annotated[
        str, Field(default="11:00", description="Check-out time (HH:MM)")
    ]
    latitude: Annotated[float, Field(description="Hotel latitude")]
    longitude: Annotated[float, Field(description="Hotel longitude")]
    photos: Annotated[
        list[str], Field(default_factory=list, description="Photo URLs")
    ]
    booking_available: Annotated[
        bool, Field(default=True, description="Whether hotel is bookable")
    ]
    over_budget: Annotated[
        bool, Field(description="Whether hotel exceeds budget_max")
    ]
    over_budget_amount: Annotated[
        float, Field(default=0.0, description="Amount over budget", ge=0)
    ]
    rank: Annotated[int | None, Field(default=None, description="Rank in shortlist")]


class HotelSearchMetadataModel(BaseModel):
    """Metadata about the hotel search execution."""

    total_results: Annotated[int, Field(description="Total results from API", ge=0)]
    returned_results: Annotated[
        int, Field(description="Number of results returned", ge=0)
    ]
    search_radius_km: Annotated[
        float, Field(default=5.0, description="Search radius in km", ge=0)
    ]
    api_response_time_ms: Annotated[
        int | None, Field(default=None, description="API response time in ms", ge=0)
    ]
    geocoding_service: Annotated[
        str, Field(default="nominatim", description="Geocoding provider used")
    ]
    api_provider: Annotated[
        str, Field(default="nuitee_liteapi", description="Hotel API provider")
    ]
    liteapi_production_mode: Annotated[
        bool, Field(default=True, description="LiteAPI production mode flag")
    ]


class HotelSearchErrorModel(BaseModel):
    code: Literal[
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "no_results",
        "geocoding_error",
        "unknown_error",
    ]
    message: str


class HotelSearchArtifactContentModel(BaseModel):
    """Complete hotel search artifact matching TravelPlanner artifact schema."""

    task_ref: Annotated[str, Field(description="Task reference string")]
    status: Annotated[
        Literal["success", "partial", "failed", "skipped"],
        Field(description="Search execution status"),
    ]
    attempt: Annotated[int, Field(description="Number of search attempts")]
    search_parameters: Annotated[
        HotelSearchParametersModel, Field(description="Search query parameters")
    ]
    options: Annotated[
        list[HotelOptionModel],
        Field(default_factory=list, description="Ranked list of hotel options (3-10)"),
    ]
    metadata: Annotated[
        HotelSearchMetadataModel, Field(description="Search execution metadata")
    ]
    errors: Annotated[
        list[HotelSearchErrorModel],
        Field(default_factory=list, description="Errors encountered during search"),
    ]
    config: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Search configuration options"),
    ] = Field(default_factory=dict)