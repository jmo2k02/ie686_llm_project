from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


SlotCategory = Literal[
    "meal",
    "attraction",
    "transport",
    "lodging",
    "leisure",
    "other",
]


class Slot(BaseModel):
    """A single time-bounded entry on a day in a travel plan."""

    name: str = Field(min_length=1, description="Short label, e.g. 'Breakfast'")
    description: str = Field(default="", description="Free-text description of the slot")
    start_time: datetime = Field(description="Slot start datetime")
    end_time: datetime = Field(
        description="Slot end datetime; must be strictly after start_time"
    )
    category: SlotCategory = Field(default="other", description="Slot category")
    location: str | None = Field(
        default=None, description="Where the slot takes place"
    )
    cost: float | None = Field(
        default=None, ge=0.0, description="Estimated cost in EUR; None if unknown"
    )
    links: list[str] = Field(
        default_factory=list,
        description="Source, verification, or booking URLs associated with the slot",
    )
    notes: str | None = Field(default=None, description="Free-form notes for the agent")

    @model_validator(mode="after")
    def _check_time_order(self) -> "Slot":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time.isoformat()}) must be strictly after "
                f"start_time ({self.start_time.isoformat()})"
            )
        return self

    def overlaps(self, other: "Slot") -> bool:
        """Two slots overlap iff their half-open intervals [start, end) intersect.

        Boundary-touching slots (one ends exactly when the other starts) do NOT overlap.
        """
        return self.start_time < other.end_time and other.start_time < self.end_time
