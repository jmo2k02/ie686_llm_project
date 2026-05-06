from __future__ import annotations

from datetime import date as _date

from pydantic import BaseModel, Field

from travelplanner.travelplan.errors import SlotNotFoundError, SlotOverlapError
from travelplanner.travelplan.slot import Slot


class Day(BaseModel):
    """A single day inside a TravelPlan, holding an ordered sequence of slots.

    Slot positions exposed by this class are 1-based to match the rendered
    markdown the agent sees.
    """

    index: int = Field(ge=1, description="1-based day index in the plan")
    calendar_date: _date | None = Field(
        default=None, description="Optional calendar date for this day"
    )
    label: str | None = Field(
        default=None, description="Optional human label, e.g. 'Arrival day'"
    )
    slots: list[Slot] = Field(
        default_factory=list, description="Ordered slots on this day"
    )

    def _assert_no_overlap(self, candidate: Slot, ignore_zero_index: int | None = None) -> None:
        for i, existing in enumerate(self.slots):
            if ignore_zero_index is not None and i == ignore_zero_index:
                continue
            if candidate.overlaps(existing):
                raise SlotOverlapError(
                    f"Slot '{candidate.name}' "
                    f"[{candidate.start_time.isoformat()} – {candidate.end_time.isoformat()}] "
                    f"overlaps existing slot '{existing.name}' "
                    f"[{existing.start_time.isoformat()} – {existing.end_time.isoformat()}] "
                    f"on day {self.index}"
                )

    def _check_position(self, position: int, *, allow_end: bool) -> int:
        upper = len(self.slots) + (1 if allow_end else 0)
        if position < 1 or position > max(upper, 1 if allow_end else 0):
            raise SlotNotFoundError(
                f"Position {position} out of range on day {self.index} "
                f"(have {len(self.slots)} slot(s); valid range "
                f"{'1..' + str(upper) if upper >= 1 else 'none'})"
            )
        return position - 1

    def append_slot(self, slot: Slot) -> int:
        """Append a slot to the end of the day. Returns the 1-based position."""
        self._assert_no_overlap(slot)
        self.slots.append(slot)
        return len(self.slots)

    def insert_slot(self, position: int, slot: Slot) -> int:
        """Insert a slot at the given 1-based position.

        Position 1 inserts at the front; position len(slots)+1 appends.
        """
        zero_idx = self._check_position(position, allow_end=True)
        self._assert_no_overlap(slot)
        self.slots.insert(zero_idx, slot)
        return position

    def delete_slot(self, position: int) -> Slot:
        """Delete and return the slot at the given 1-based position."""
        zero_idx = self._check_position(position, allow_end=False)
        return self.slots.pop(zero_idx)

    def sorted_slots(self) -> list[Slot]:
        """Return slots sorted by start_time. Does not mutate the day."""
        return sorted(self.slots, key=lambda s: s.start_time)

    def total_cost(self) -> float:
        """Sum of slot costs on this day; treats unknown costs as 0."""
        return sum((s.cost or 0.0) for s in self.slots)
