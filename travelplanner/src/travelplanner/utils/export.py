"""Export utilities for TravelPlanner outputs.

Converts ``CalenderModel`` (itinerary timetable) into a human-readable Markdown
file for easy review.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

from travelplanner.schema.calender import (
    ActivityModel,
    CalenderModel,
    DayModel,
    TripSummaryModel,
)


def _fmt_time(t: time | None) -> str:
    return t.strftime("%H:%M") if t else "—"


def calender_to_markdown(calendar: CalenderModel) -> str:
    """Convert a ``CalenderModel`` into a complete Markdown document."""
    lines: list[str] = []
    ts: TripSummaryModel = calendar.trip_summary

    # ── Trip Summary ────────────────────────────────────────────────────
    destination = ts.destination or "(not set)"
    lines.append(f"# ✈️  Travel Itinerary: {destination}")
    lines.append("")

    if ts.origin:
        lines.append(f"**From:** {ts.origin}")
    if ts.start_date:
        lines.append(f"**Start:** {ts.start_date}")
    if ts.end_date:
        lines.append(f"**End:** {ts.end_date}")
    if ts.traveler_count:
        lines.append(f"**Travelers:** {ts.traveler_count}")
    if ts.total_budget is not None:
        currency = ts.currency or "EUR"
        lines.append(f"**Budget:** {ts.total_budget:.2f} {currency}")
    if ts.total_cost_estimate is not None:
        currency = ts.currency or "EUR"
        lines.append(f"**Estimated Cost:** {ts.total_cost_estimate:.2f} {currency}")
    lines.append("")

    # ── Days ────────────────────────────────────────────────────────────
    if not calendar.days:
        lines.append("_No days have been scheduled yet._")
        lines.append("")
    else:
        for day in calendar.days:
            lines.extend(_day_to_markdown(day))

    # ── Unscheduled Tasks ───────────────────────────────────────────────
    if calendar.unscheduled_tasks:
        lines.append("## ⚠️  Unscheduled Tasks")
        lines.append("")
        for task_name in calendar.unscheduled_tasks:
            lines.append(f"- {task_name}")
        lines.append("")

    return "\n".join(lines)


def _day_to_markdown(day: DayModel) -> list[str]:
    lines: list[str] = []
    date_str = day.date.strftime("%A, %d %B %Y") if day.date else f"Day {day.day_number}"
    lines.append(f"## 📅 {date_str} (Day {day.day_number})")
    lines.append("")

    if day.daily_budget is not None:
        lines.append(f"**Daily Budget:** {day.daily_budget:.2f}")
        lines.append("")

    if day.notes:
        lines.append(f"*{day.notes}*")
        lines.append("")

    if not day.activities:
        lines.append("_No activities scheduled._")
        lines.append("")
        return lines

    lines.append("| Time | Activity | Type | Location | Duration | Cost | Notes |")
    lines.append("|------|----------|------|----------|----------|------|-------|")

    for act in day.activities:
        lines.append(_activity_to_table_row(act))

    lines.append("")
    return lines


def _activity_to_table_row(act: ActivityModel) -> str:
    time_str = f"{_fmt_time(act.start_time)} – {_fmt_time(act.end_time)}"
    name = act.name or "—"
    act_type = act.activity_type or "—"
    location = act.location or "—"
    duration = f"{act.duration_minutes} min" if act.duration_minutes else "—"
    cost = f"{act.cost_estimate:.2f} {act.currency}" if act.cost_estimate is not None else "—"
    notes = act.notes or ""
    confirmed_badge = " ✅" if act.confirmed else ""
    return f"| {time_str} | {name}{confirmed_badge} | {act_type} | {location} | {duration} | {cost} | {notes} |"


def save_itinerary_markdown(
    calendar: CalenderModel,
    output_path: Path,
) -> Path:
    """Save the itinerary as a Markdown file and return the resolved path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(calender_to_markdown(calendar), encoding="utf-8")
    return output_path.resolve()
