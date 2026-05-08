"""Schema for Google Route Plan API responses (computeRoutes)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteStepDetailModel(BaseModel):
    distance_meters: int | None = None
    duration_seconds: float | None = None
    start_lat_lng: dict[str, float] | None = None
    end_lat_lng: dict[str, float] | None = None
    html_instructions: str | None = None
    travel_mode: str | None = None


RouteStepKind = Literal["driving", "walking", "transit", "bicycling", "transit_step"]


class RouteStepModel(BaseModel):
    kind: RouteStepKind
    summary: str = ""
    distance_meters: int | None = None
    duration_seconds: float | None = None
    detail: RouteStepDetailModel | None = None
    line_short: str | None = None
    board_stop_name: str | None = None
    alight_stop_name: str | None = None


class RouteLegModel(BaseModel):
    distance_meters: int | None = None
    duration_seconds: float | None = None
    start_address: str | None = None
    end_address: str | None = None


class RouteMetricModel(BaseModel):
    distance_meters: int = 0
    distance_km: float = 0.0
    duration_seconds: float = 0.0


class RouteRequestModel(BaseModel):
    origin: str
    destination: str
    travel_mode: str


class AlternativeTransitRouteModel(BaseModel):
    rank: int
    metrics: RouteMetricModel
    steps: list[RouteStepModel] = Field(default_factory=list)


class RoutePlanModel(BaseModel):
    """Single route response from Google Routes API (computeRoutes)."""

    request: RouteRequestModel
    metrics: RouteMetricModel
    steps: list[RouteStepModel] = Field(default_factory=list)
    transit_alternatives: list[AlternativeTransitRouteModel] = Field(
        default_factory=list
    )


RouteDetailLevel = Literal["route_summary", "standard", "full"]
