from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

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

    name: Annotated[
        str,
        Field(min_length=1, description="Short label, e.g. 'Breakfast'"),
    ]
    description: Annotated[
        str,
        Field(default="", description="Free-text description of the slot"),
    ]
    start_time: Annotated[
        datetime,
        Field(description="Slot start datetime"),
    ]
    end_time: Annotated[
        datetime,
        Field(description="Slot end datetime; must be strictly after start_time"),
    ]
    category: Annotated[
        SlotCategory,
        Field(default="other", description="Slot category"),
    ]
    location: Annotated[
        str | None,
        Field(default=None, description="Where the slot takes place"),
    ]
    cost: Annotated[
        float | None,
        Field(default=None, ge=0.0, description="Estimated cost in EUR; None if unknown"),
    ]
    notes: Annotated[
        str | None,
        Field(default=None, description="Free-form notes for the agent"),
    ]

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
