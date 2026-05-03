from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class RestaurantParamsModel(BaseModel):
    """Structured parameters extracted from a natural-language restaurant task."""

    city: Annotated[str, Field(description="Destination city for the search")]
    cuisine: Annotated[str | None, Field(description="Cuisine type or dish preference, if mentioned")] = None
    budget: Annotated[
        Literal["low", "medium", "high"] | None,
        Field(description="Price budget category, if mentioned"),
    ] = None
    meal_type: Annotated[
        Literal["breakfast", "brunch", "lunch", "dinner", "any"] | None,
        Field(description="Target meal, if mentioned"),
    ] = None
    dietary_restrictions: Annotated[list[str], Field(default_factory=list, description="Dietary constraints, e.g. ['vegetarian', 'gluten-free']")]
    min_rating: Annotated[float | None, Field(description="Minimum acceptable rating, if mentioned")] = None
    num_people: Annotated[int, Field(description="Number of diners, default 1")] = 1
    preferred_time: Annotated[str | None, Field(description="Preferred reservation time, e.g. '19:30' or 'evening'",)] = None
    additional_notes: Annotated[str | None, Field(description="Any extra free-text requirements")] = None


class RestaurantLocationModel(BaseModel):
    lat: float
    lng: float


class RestaurantCandidateModel(BaseModel):
    """A single place returned by the LiteAPI places backend."""

    place_id: Annotated[str, Field(description="Google Place ID from LiteAPI")]
    name: Annotated[str, Field(description="Display name of the place")]
    address: Annotated[str | None, Field(description="Formatted address")] = None
    types: Annotated[list[str], Field(default_factory=list, description="Google place types, e.g. ['restaurant', 'food']")]
    rating: Annotated[float | None, Field(description="Google rating, if available")] = None
    price_level: Annotated[str | None, Field(description="Price level indicator, if available")] = None
    phone: Annotated[str | None, Field(description="Phone number, if available")] = None
    website: Annotated[str | None, Field(description="Website URL, if available")] = None
    opening_hours: Annotated[str | None, Field(description="Opening hours text, if available")] = None
    location: Annotated[RestaurantLocationModel | None, Field(description="Lat/lng coordinates, if available")] = None
    photos: Annotated[list[str], Field(default_factory=list, description="Photo reference strings, if available")]
    raw: Annotated[dict[str, Any], Field(default_factory=dict, description="Unmodified LiteAPI raw payload for extensibility")]


class RestaurantSelectionModel(BaseModel):
    """LLM selection of the best candidate from a list."""

    selected_index: int
    selection_reason: str


class RestaurantItemModel(BaseModel):
    """Final enriched restaurant recommendation for the itinerary."""

    name: str
    address: str | None = None
    place_id: str | None = None
    cuisine: str | None = None
    meal_type: str | None = None
    rating: float | None = None
    price_level: str | None = None
    price_range: Annotated[str | None, Field(description="Resolved price symbol, e.g. '$', '$$', '$$$'")] = None
    phone: str | None = None
    website: str | None = None
    opening_hours: str | None = None
    location: Annotated[dict[str, float] | None, Field(description='{"lat": ..., "lng": ...}')] = None
    dietary_suitability: Annotated[list[str], Field(default_factory=list, description="Matched dietary tags for this venue")]
    selection_reason: Annotated[str | None, Field(description="One-sentence reason from selector LLM")] = None
    provenance: Annotated[str, Field(description='"liteapi_places" or "fallback_llm_suggestion"')]


class RestaurantSearchErrorModel(BaseModel):
    code: Literal[
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "llm_error",
        "unknown_error",
        "no_results",
    ]
    message: str


class RestaurantArtifactContentModel(BaseModel):
    task_ref: str
    status: Literal["success", "partial", "failed", "skipped"]
    provider: Literal["google_places_api_new"]
    query: Annotated[str, Field(description="Original natural-language query / task text")]
    city: str
    cuisine: str | None = None
    budget: str | None = None
    meal_type: str | None = None
    items: list[RestaurantItemModel] = Field(default_factory=list)
    errors: list[RestaurantSearchErrorModel] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
