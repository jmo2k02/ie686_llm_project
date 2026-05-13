from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from travelplanner.travelplan.slot import SlotCategory


class InitPlanArgs(BaseModel):
    title: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional plan title (e.g. 'Rome 5-day'). Pass None to leave "
                "the plan untitled."
            ),
        ),
    ]


class AddDayArgs(BaseModel):
    label: Annotated[
        str | None,
        Field(default=None, description="Optional human label, e.g. 'Arrival day'."),
    ]
    calendar_date_iso: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional ISO-8601 date string like '2026-06-01'.",
        ),
    ]


class RemoveDayArgs(BaseModel):
    day_index: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "1-based index of the day to remove. Remaining days are renumbered."
            ),
        ),
    ]


class _SlotPayload(BaseModel):
    """Shared slot fields for add_slot and insert_slot."""

    day_index: Annotated[
        int,
        Field(ge=1, description="1-based day index where the slot belongs."),
    ]
    name: Annotated[
        str,
        Field(min_length=1, description="Short label, e.g. 'Breakfast'."),
    ]
    start_time_iso: Annotated[
        str,
        Field(
            description="ISO-8601 start datetime, e.g. '2026-06-01T08:00'.",
        ),
    ]
    end_time_iso: Annotated[
        str,
        Field(
            description=(
                "ISO-8601 end datetime, must be strictly after start_time_iso."
            ),
        ),
    ]
    description: Annotated[
        str,
        Field(default="", description="Free-text description of the slot."),
    ]
    category: Annotated[
        SlotCategory,
        Field(
            default="other",
            description=(
                "Slot category — one of meal, attraction, transport, lodging, "
                "leisure, other."
            ),
        ),
    ]
    location: Annotated[
        str | None,
        Field(default=None, description="Where the slot takes place."),
    ]
    cost: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Estimated cost in EUR; omit if unknown.",
        ),
    ]
    links: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Source, verification, or booking URLs for this slot. Include URLs "
                "returned by sub-agent tools."
            ),
        ),
    ]
    notes: Annotated[
        str | None,
        Field(default=None, description="Free-form notes."),
    ]


class AddSlotArgs(_SlotPayload):
    pass


class InsertSlotArgs(_SlotPayload):
    position: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "1-based position where to insert. Position 1 inserts at the front; "
                "position N+1 appends. Must be in 1..len(slots)+1."
            ),
        ),
    ]


class DeleteSlotArgs(BaseModel):
    day_index: Annotated[
        int,
        Field(ge=1, description="1-based day index."),
    ]
    position: Annotated[
        int,
        Field(ge=1, description="1-based slot position to delete."),
    ]


class ViewPlanArgs(BaseModel):
    """No arguments."""


class CostSummaryArgs(BaseModel):
    """No arguments."""
