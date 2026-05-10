"""Calendar / Timetable schema for the Execution Agent output.

An itinerary is a list of days, each containing time-boxed activities that
reference the source task and any agent artifacts that back them.
"""

from __future__ import annotations

from datetime import date, time
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ActivityModel(BaseModel):
    """A single time-boxed activity inside a day."""

    name: Annotated[str, Field(description="Human-readable activity name")]
    activity_type: Annotated[
        Literal["flight", "hotel", "restaurant", "attraction", "transport", "opening_times", "general-web-search", "routing-check", "free_time", "other"],
        Field(description="Category matching the originating task type or a free slot"),
    ]
    start_time: Annotated[time | None, Field(default=None, description="Scheduled start time")]
    end_time: Annotated[time | None, Field(default=None, description="Scheduled end time")]
    duration_minutes: Annotated[int | None, Field(default=None, description="Planned duration in minutes")]
    location: Annotated[str | None, Field(default=None, description="Address or place name")]
    cost_estimate: Annotated[float | None, Field(default=None, description="Estimated cost in user currency")]
    currency: Annotated[str | None, Field(default="EUR", description="Currency for cost_estimate")]
    notes: Annotated[str | None, Field(default=None, description="Additional notes or constraints")]
    source_task_name: Annotated[str | None, Field(default=None, description="Reference to the TaskModel.name that produced this activity")]
    source_artifact_type: Annotated[str | None, Field(default=None, description="Reference to the artifact type backing this activity")]
    confirmed: Annotated[bool, Field(default=False, description="Whether this activity has been validated externally")]


class DayModel(BaseModel):
    """One day of the itinerary."""

    date: Annotated[date | None, Field(default=None, description="Calendar date for this day")]
    day_number: Annotated[int, Field(description="1-based day index within the trip")]
    activities: Annotated[list[ActivityModel], Field(default_factory=list, description="Ordered list of activities")]
    daily_budget: Annotated[float | None, Field(default=None, description="Budget allocated for this day")]
    notes: Annotated[str | None, Field(default=None, description="Day-level notes")]


class TripSummaryModel(BaseModel):
    """High-level trip metadata."""

    destination: Annotated[str | None, Field(default=None)]
    origin: Annotated[str | None, Field(default=None)]
    start_date: Annotated[date | None, Field(default=None)]
    end_date: Annotated[date | None, Field(default=None)]
    total_budget: Annotated[float | None, Field(default=None)]
    total_cost_estimate: Annotated[float | None, Field(default=None)]
    currency: Annotated[str | None, Field(default="EUR")]
    traveler_count: Annotated[int | None, Field(default=None)]


class CalenderModel(BaseModel):
    """Complete timetable produced by the Execution Agent."""

    trip_summary: Annotated[TripSummaryModel, Field(default_factory=TripSummaryModel)]
    days: Annotated[list[DayModel], Field(default_factory=list, description="Chronological days")]
    unscheduled_tasks: Annotated[
        list[str],
        Field(default_factory=list, description="Task names that could not be placed on the calendar"),
    ]

    @property
    def all_activities(self) -> list[ActivityModel]:
        """Flatten all activities across days."""
        result: list[ActivityModel] = []
        for day in self.days:
            result.extend(day.activities)
        return result
