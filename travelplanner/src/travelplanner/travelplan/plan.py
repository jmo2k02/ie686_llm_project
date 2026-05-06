from __future__ import annotations

from datetime import date as _date
from typing import Annotated

from pydantic import BaseModel, Field

from travelplanner.travelplan.day import Day
from travelplanner.travelplan.errors import DayNotFoundError
from travelplanner.travelplan.slot import Slot


class CostSummary(BaseModel):
    total: Annotated[float, Field(description="Total estimated cost across all days")]
    per_day: Annotated[
        dict[int, float],
        Field(description="Per-day estimated cost, keyed by 1-based day index"),
    ]


class TravelPlan(BaseModel):
    """A multi-day travel plan composed of ordered Day buckets.

    Day indices are 1-based and renumbered after a removal. Slot positions on a
    day are also 1-based.
    """

    title: Annotated[
        str | None,
        Field(default=None, description="Optional plan title"),
    ]
    days: Annotated[
        list[Day],
        Field(default_factory=list, description="Ordered days in the plan"),
    ]

    # ── Day operations ────────────────────────────────────────────────────────

    def add_day(
        self,
        label: str | None = None,
        calendar_date: _date | None = None,
    ) -> Day:
        """Append a new empty day to the plan and return it."""
        day = Day(index=len(self.days) + 1, label=label, calendar_date=calendar_date)
        self.days.append(day)
        return day

    def remove_day(self, day_index: int) -> Day:
        """Remove a day by 1-based index; renumber remaining days."""
        day = self._resolve_day(day_index)
        self.days.pop(day_index - 1)
        for i, d in enumerate(self.days, start=1):
            d.index = i
        return day

    def get_day(self, day_index: int) -> Day:
        """Return the day at the given 1-based index."""
        return self._resolve_day(day_index)

    def _resolve_day(self, day_index: int) -> Day:
        if day_index < 1 or day_index > len(self.days):
            raise DayNotFoundError(
                f"Day index {day_index} out of range (have {len(self.days)} day(s))"
            )
        return self.days[day_index - 1]

    # ── Slot operations (delegate to Day) ─────────────────────────────────────

    def add_slot(self, day_index: int, slot: Slot) -> int:
        return self._resolve_day(day_index).append_slot(slot)

    def insert_slot(self, day_index: int, position: int, slot: Slot) -> int:
        return self._resolve_day(day_index).insert_slot(position, slot)

    def delete_slot(self, day_index: int, position: int) -> Slot:
        return self._resolve_day(day_index).delete_slot(position)

    # ── Cost ──────────────────────────────────────────────────────────────────

    def total_cost(self) -> float:
        return sum(d.total_cost() for d in self.days)

    def daily_costs(self) -> dict[int, float]:
        return {d.index: d.total_cost() for d in self.days}

    def cost_summary(self) -> CostSummary:
        return CostSummary(total=self.total_cost(), per_day=self.daily_costs())

    # ── Markdown ──────────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Render the plan as a markdown table with one column per day.

        Cells render slots sorted by start_time. Empty days/slots render as
        empty cells. A cost summary line follows the table.
        """
        lines: list[str] = []
        if self.title:
            lines.append(f"# TravelPlan: {self.title}")
        else:
            lines.append("# TravelPlan")
        lines.append("")

        if not self.days:
            lines.append("_No days yet._")
            return "\n".join(lines)

        headers = [_render_day_header(d) for d in self.days]
        cells_per_day = [_render_day_cells(d) for d in self.days]
        max_rows = max((len(c) for c in cells_per_day), default=0)

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        if max_rows == 0:
            lines.append("| " + " | ".join("" for _ in headers) + " |")
        else:
            for row in range(max_rows):
                row_cells = [
                    cells[row] if row < len(cells) else ""
                    for cells in cells_per_day
                ]
                lines.append("| " + " | ".join(row_cells) + " |")

        lines.append("")
        summary = self.cost_summary()
        per_day_str = ", ".join(
            f"Day {idx}: €{cost:.2f}" for idx, cost in summary.per_day.items()
        )
        lines.append(
            f"**Total estimated cost: €{summary.total:.2f}**"
            + (f" ({per_day_str})" if per_day_str else "")
        )
        return "\n".join(lines)


def _render_day_header(day: Day) -> str:
    parts = [f"Day {day.index}"]
    if day.calendar_date is not None:
        parts.append(day.calendar_date.isoformat())
    if day.label:
        parts.append(day.label)
    return " — ".join(parts)


def _render_day_cells(day: Day) -> list[str]:
    return [
        _render_slot_cell(position, slot)
        for position, slot in enumerate(day.sorted_slots(), start=1)
    ]


def _render_slot_cell(position: int, slot: Slot) -> str:
    start = slot.start_time
    end = slot.end_time
    if start.date() == end.date():
        time_range = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
    else:
        time_range = (
            f"{start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%Y-%m-%d %H:%M')}"
        )

    bits = [f"**{position}. {slot.name}**", f"[{slot.category}]", time_range]
    if slot.location:
        bits.append(f"@ {slot.location}")
    if slot.cost is not None:
        bits.append(f"(€{slot.cost:.2f})")
    cell = " ".join(bits)
    if slot.description:
        cell += f" — {slot.description}"
    return cell
