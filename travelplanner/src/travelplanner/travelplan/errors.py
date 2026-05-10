from __future__ import annotations


class TravelPlanError(Exception):
    """Base class for TravelPlan errors."""


class SlotOverlapError(TravelPlanError):
    """Raised when a slot would overlap an existing slot on the same day."""


class SlotNotFoundError(TravelPlanError):
    """Raised when a slot position is out of range on a day."""


class DayNotFoundError(TravelPlanError):
    """Raised when a day index is out of range in a plan."""
