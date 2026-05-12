from __future__ import annotations

import hashlib
from datetime import date as _date
from datetime import datetime, timezone
from typing import Callable

from icalendar import Calendar, Event
from pydantic import BaseModel, Field

from travelplanner.travelplan.day import Day
from travelplanner.travelplan.errors import DayNotFoundError
from travelplanner.travelplan.slot import Slot


_ICAL_PRODID = "-//TravelPlanner//TravelPlan//EN"
_ICAL_UID_DOMAIN = "travelplanner.local"


class CostSummary(BaseModel):
    total: float = Field(description="Total estimated cost across all days")
    per_day: dict[int, float] = Field(
        description="Per-day estimated cost, keyed by 1-based day index"
    )


class TravelPlan(BaseModel):
    """A multi-day travel plan composed of ordered Day buckets.

    Day indices are 1-based and renumbered after a removal. Slot positions on a
    day are also 1-based.
    """

    title: str | None = Field(default=None, description="Optional plan title")
    days: list[Day] = Field(
        default_factory=list, description="Ordered days in the plan"
    )

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

        Cells render slots sorted by start_time and include category,
        location, cost, and description. A cost summary line follows the
        table.
        """
        return self._render_table(_render_slot_cell, include_cost_summary=True)

    def to_markdown_compact(self) -> str:
        """Render the plan as a compact markdown table.

        Each slot cell shows ONLY the position, name, and time range — no
        category, location, cost, or description. Useful when the renderer
        has limited horizontal space (e.g. the CLI dashboard panel).
        """
        return self._render_table(_render_slot_cell_compact, include_cost_summary=True)

    def to_ical(self) -> str:
        """Render the plan as an RFC 5545 iCalendar (``.ics``) string.

        Each slot becomes one ``VEVENT`` with ``SUMMARY``, ``DTSTART``,
        ``DTEND``, ``DESCRIPTION``, and ``LOCATION``. Naive datetimes are
        passed through as floating times (recommended for travel plans —
        users typically mean local-at-destination); tz-aware datetimes are
        preserved.

        UIDs are deterministic per (plan title, day index, slot name,
        start, end), so re-importing the same plan replaces previous
        events instead of duplicating them.
        """
        dtstamp = datetime.now(timezone.utc)
        cal = Calendar()
        cal.add("version", "2.0")
        cal.add("prodid", _ICAL_PRODID)
        cal.add("calscale", "GREGORIAN")
        cal.add("method", "PUBLISH")
        if self.title:
            cal.add("x-wr-calname", self.title)

        for day in self.days:
            for slot in day.sorted_slots():
                cal.add_component(_build_vevent(self.title, day.index, slot, dtstamp))

        return cal.to_ical().decode("utf-8")

    def _render_table(
        self,
        slot_renderer: "Callable[[int, Slot], str]",
        *,
        include_cost_summary: bool,
    ) -> str:
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
        cells_per_day = [
            [slot_renderer(pos, slot) for pos, slot in enumerate(d.sorted_slots(), start=1)]
            for d in self.days
        ]
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

        if include_cost_summary:
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


def _format_time_range(slot: Slot) -> str:
    start = slot.start_time
    end = slot.end_time
    if start.date() == end.date():
        return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
    return f"{start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%Y-%m-%d %H:%M')}"


def _render_slot_cell(position: int, slot: Slot) -> str:
    bits = [
        f"**{position}. {slot.name}**",
        f"[{slot.category}]",
        _format_time_range(slot),
    ]
    if slot.location:
        bits.append(f"@ {slot.location}")
    if slot.cost is not None:
        bits.append(f"(€{slot.cost:.2f})")
    cell = " ".join(bits)
    if slot.description:
        cell += f" — {slot.description}"
    return cell


def _render_slot_cell_compact(position: int, slot: Slot) -> str:
    return f"**{position}. {slot.name}** {_format_time_range(slot)}"


# ── iCalendar helpers ──────────────────────────────────────────────────────


def _slot_uid(plan_title: str | None, day_index: int, slot: Slot) -> str:
    """Deterministic UID so re-imports update events instead of duplicating."""
    raw = "|".join(
        [
            plan_title or "",
            str(day_index),
            slot.name,
            slot.start_time.isoformat(),
            slot.end_time.isoformat(),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{digest}@{_ICAL_UID_DOMAIN}"


def _build_vevent(
    plan_title: str | None,
    day_index: int,
    slot: Slot,
    dtstamp: datetime,
) -> Event:
    event = Event()
    event.add("uid", _slot_uid(plan_title, day_index, slot))
    event.add("dtstamp", dtstamp)
    event.add("dtstart", slot.start_time)
    event.add("dtend", slot.end_time)
    event.add("summary", slot.name)

    description_parts: list[str] = []
    if slot.description:
        description_parts.append(slot.description)
    if slot.cost is not None:
        description_parts.append(f"Cost: €{slot.cost:.2f}")
    description_parts.append(f"Category: {slot.category}")
    if slot.notes:
        description_parts.append(f"Notes: {slot.notes}")
    if description_parts:
        # icalendar handles \n escaping inside TEXT values for us.
        event.add("description", "\n".join(description_parts))

    if slot.location:
        event.add("location", slot.location)
    return event
