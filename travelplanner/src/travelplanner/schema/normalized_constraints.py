from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TravelersNormalized(BaseModel):
    adults: int = 1
    children_under_6: int = 0
    children_6_to_16: int = 0


class NormalizedConstraints(BaseModel):
    # Core trip info
    destination: str | None = None
    origin: str | None = None
    date_from: str | None = None           # YYYY.MM.DD
    date_to: str | None = None             # YYYY.MM.DD
    travelers: TravelersNormalized = Field(default_factory=TravelersNormalized)
    budget_amount: float | None = None
    budget_currency: str | None = None     # ISO 4217, e.g. "EUR"
    accommodation: str | None = None
    transport: Literal["Flight", "Car", "No Preference"] | None = None
    interests: str | None = None

    # Derived from destination
    destination_country: str | None = None
    destination_country_code: str | None = None   # ISO 3166-1 alpha-2
    destination_currency: str | None = None        # ISO 4217
    destination_timezone: str | None = None        # IANA, e.g. "Europe/Madrid"
    destination_language: str | None = None

    # Derived from origin
    origin_country: str | None = None
    origin_country_code: str | None = None         # ISO 3166-1 alpha-2
    origin_currency: str | None = None             # ISO 4217
