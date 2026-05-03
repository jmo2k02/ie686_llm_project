from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class GeneratedActivityModel(BaseModel):
    day: int
    time_slot: Annotated[Literal["morning", "afternoon", "evening"], Field(description="Half-day slot")]
    title: Annotated[str, Field(description="Short evocative title, max 8 words")]
    description: Annotated[str, Field(description="Narrative, 3-5 sentences, destination-specific")]
    local_touchpoint: Annotated[str, Field(description="Who the traveller meets and why it is not tourist-facing")]
    search_keywords: Annotated[list[str], Field(description="2-3 Google Maps keyword strings, no city name")]
    estimated_duration_hours: float
    has_specific_location: Annotated[
        bool,
        Field(description="False if the activity is inherently location-agnostic"),
    ]


class GeneratedActivitiesResponse(BaseModel):
    activities: list[GeneratedActivityModel]


class AttractionCandidateModel(BaseModel):
    title: str
    address: str | None = None
    gps_coordinates: Annotated[
        dict[str, float] | None,
        Field(description='{"lat": ..., "lng": ...}'),
    ] = None
    rating: float | None = None
    reviews: int | None = None
    price: str | None = None
    type: str | None = None
    data_id: str | None = None
    hours: str | None = None
    review_snippets: list[str] = Field(default_factory=list)


class CandidateSelectionModel(BaseModel):
    selected_index: int
    selection_reason: str


class AttractionItemModel(BaseModel):
    # Activity (LLM-generated)
    day: int
    time_slot: str
    title: str
    description: str
    local_touchpoint: str
    estimated_duration_hours: float
    has_specific_location: bool
    # Place (SERPAPI result, all nullable)
    location_name: Annotated[str, Field(description="Place name if found, destination city if not")]
    location_address: str | None = None
    coordinates: Annotated[dict[str, float] | None, Field(description='{"lat": ..., "lng": ...}')] = None
    place_id: str | None = None
    place_rating: float | None = None
    place_review_count: int | None = None
    place_price_level: Annotated[str | None, Field(description='Raw price symbol e.g. "$", "$$"')] = None
    place_type: str | None = None
    place_hours: str | None = None
    selection_reason: Annotated[str | None, Field(description="One-sentence reason from selector LLM")] = None
    place_found: bool
    # Budget + provenance
    estimated_price_range: Annotated[
        str,
        Field(description='Price symbol from SERPAPI or budget fallback ("$"/"$$"/"$$$")'),
    ]
    selected_archetype: Annotated[str, Field(description="Matched archetype name, e.g. 'digital_nomad'")]
    provenance: Annotated[
        str,
        Field(description='"LLM activity | SERPAPI google_maps" or "LLM activity | no place found"'),
    ]


class AttractionSearchErrorModel(BaseModel):
    code: Literal[
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "llm_error",
        "unknown_error",
    ]
    message: str


class AttractionArtifactContentModel(BaseModel):
    task_ref: str
    status: Literal["success", "partial", "failed", "skipped"]
    provider: Literal["openai_embeddings+llm+serpapi_google_maps"]
    destination: str
    days: int
    budget: str
    selected_archetype: str
    items: list[AttractionItemModel] = Field(default_factory=list)
    errors: list[AttractionSearchErrorModel] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
