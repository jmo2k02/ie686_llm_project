from __future__ import annotations

from datetime import date as _date
from datetime import datetime

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from travelplanner.travelplan.errors import TravelPlanError
from travelplanner.travelplan.plan import TravelPlan
from travelplanner.travelplan.slot import Slot, SlotCategory
from travelplanner.travelplan.tool_args import (
    AddDayArgs,
    AddSlotArgs,
    CostSummaryArgs,
    DeleteSlotArgs,
    InitPlanArgs,
    InsertSlotArgs,
    RemoveDayArgs,
    ViewPlanArgs,
)


def _err(exc: Exception) -> str:
    return f"Error: {exc}"


def make_travelplan_tools(plan: TravelPlan) -> list[BaseTool]:
    """Build the agent-callable tool list for a given TravelPlan instance.

    The plan is closure-bound into each tool — every returned tool mutates
    that single shared instance. Domain errors (overlap, bad index, malformed
    ISO string) are caught and returned as ``Error: ...`` strings so the
    calling agent reads the failure as a tool message and self-corrects
    without crashing the loop.
    """

    def _build_slot(
        name: str,
        description: str,
        start_time_iso: str,
        end_time_iso: str,
        category: SlotCategory,
        location: str | None,
        cost: float | None,
        links: list[str],
        notes: str | None,
    ) -> Slot:
        return Slot(
            name=name,
            description=description,
            start_time=datetime.fromisoformat(start_time_iso),
            end_time=datetime.fromisoformat(end_time_iso),
            category=category,
            location=location,
            cost=cost,
            links=links,
            notes=notes,
        )

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def init_plan(title: str | None = None) -> str:
        plan.title = title
        plan.days.clear()
        if title:
            return f"Plan initialized with title '{title}'. 0 day(s)."
        return "Plan initialized (untitled). 0 day(s)."

    # ── Days ────────────────────────────────────────────────────────────────

    def add_day(label: str | None = None, calendar_date_iso: str | None = None) -> str:
        try:
            cd = _date.fromisoformat(calendar_date_iso) if calendar_date_iso else None
            day = plan.add_day(label=label, calendar_date=cd)
            return f"Added day {day.index}. Plan now has {len(plan.days)} day(s)."
        except (TravelPlanError, ValueError, ValidationError) as e:
            return _err(e)

    def remove_day(day_index: int) -> str:
        try:
            removed = plan.remove_day(day_index)
            return (
                f"Removed day {day_index} (had {len(removed.slots)} slot(s)). "
                f"Plan now has {len(plan.days)} day(s); remaining days renumbered."
            )
        except TravelPlanError as e:
            return _err(e)

    # ── Slots ───────────────────────────────────────────────────────────────

    def add_slot(
        day_index: int,
        name: str,
        start_time_iso: str,
        end_time_iso: str,
        description: str = "",
        category: SlotCategory = "other",
        location: str | None = None,
        cost: float | None = None,
        links: list[str] | None = None,
        notes: str | None = None,
    ) -> str:
        try:
            slot = _build_slot(
                name, description, start_time_iso, end_time_iso,
                category, location, cost, links or [], notes,
            )
            position = plan.add_slot(day_index, slot)
            day = plan.get_day(day_index)
            return (
                f"Added slot '{name}' at Day {day_index} position {position}. "
                f"Day total now €{day.total_cost():.2f}."
            )
        except (TravelPlanError, ValueError, ValidationError) as e:
            return _err(e)

    def insert_slot(
        day_index: int,
        position: int,
        name: str,
        start_time_iso: str,
        end_time_iso: str,
        description: str = "",
        category: SlotCategory = "other",
        location: str | None = None,
        cost: float | None = None,
        links: list[str] | None = None,
        notes: str | None = None,
    ) -> str:
        try:
            slot = _build_slot(
                name, description, start_time_iso, end_time_iso,
                category, location, cost, links or [], notes,
            )
            actual = plan.insert_slot(day_index, position, slot)
            day = plan.get_day(day_index)
            return (
                f"Inserted slot '{name}' at Day {day_index} position {actual}. "
                f"Day total now €{day.total_cost():.2f}."
            )
        except (TravelPlanError, ValueError, ValidationError) as e:
            return _err(e)

    def delete_slot(day_index: int, position: int) -> str:
        try:
            removed = plan.delete_slot(day_index, position)
            day = plan.get_day(day_index)
            return (
                f"Deleted slot '{removed.name}' from Day {day_index} position {position}. "
                f"Day {day_index} now has {len(day.slots)} slot(s); "
                f"day total €{day.total_cost():.2f}."
            )
        except TravelPlanError as e:
            return _err(e)

    # ── Read-only ───────────────────────────────────────────────────────────

    def view_plan() -> str:
        return plan.to_markdown()

    def cost_summary() -> str:
        summary = plan.cost_summary()
        if not summary.per_day:
            return f"Total estimated cost: €{summary.total:.2f}. (Plan has no days yet.)"
        per_day = ", ".join(
            f"Day {idx}: €{cost:.2f}" for idx, cost in summary.per_day.items()
        )
        return f"Total estimated cost: €{summary.total:.2f} ({per_day})."

    # ── Wrap as StructuredTools ─────────────────────────────────────────────

    return [
        StructuredTool.from_function(
            func=init_plan,
            name="init_plan",
            description=(
                "Reset the travel plan to an empty state with an optional title. "
                "Clears all days and slots. Call this when starting a fresh plan "
                "or when you want to wipe and re-plan from scratch. Safe to call "
                "multiple times."
            ),
            args_schema=InitPlanArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=add_day,
            name="add_day",
            description=(
                "Append a new (empty) day to the travel plan. Days are 1-based and "
                "assigned an auto-incrementing index. Use this to grow the plan one "
                "day at a time before adding slots."
            ),
            args_schema=AddDayArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=remove_day,
            name="remove_day",
            description=(
                "Remove a day by 1-based index. Remaining days are renumbered to "
                "stay contiguous (e.g. removing day 2 of [1,2,3] yields [1,2])."
            ),
            args_schema=RemoveDayArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=add_slot,
            name="add_slot",
            description=(
                "Append a slot to the END of a day. Times are ISO-8601 datetimes "
                "(e.g. '2026-06-01T08:00'). Slots within a day must not overlap; "
                "boundary-touching is allowed (one ends exactly when the next "
                "begins). Returns 'Error: ...' on overlap, bad index, or malformed "
                "datetime — read it and try again with corrected args."
            ),
            args_schema=AddSlotArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=insert_slot,
            name="insert_slot",
            description=(
                "Insert a slot at a specific 1-based position within a day. "
                "Position 1 inserts at the front; position N+1 appends. Same time "
                "and overlap rules as add_slot."
            ),
            args_schema=InsertSlotArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=delete_slot,
            name="delete_slot",
            description=(
                "Delete the slot at the given 1-based position on the given "
                "1-based day. Returns 'Error: ...' if either index is out of range."
            ),
            args_schema=DeleteSlotArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=view_plan,
            name="view_plan",
            description=(
                "Render the full travel plan as a markdown table with one column "
                "per day. Read-only. Call when you need the current layout to make "
                "a decision; otherwise rely on the short confirmation strings the "
                "mutation tools return."
            ),
            args_schema=ViewPlanArgs,
            handle_validation_error=True,
        ),
        StructuredTool.from_function(
            func=cost_summary,
            name="cost_summary",
            description=(
                "Return total estimated cost across all days plus a per-day "
                "breakdown. Read-only. Cost is in EUR; slots without a cost "
                "contribute zero."
            ),
            args_schema=CostSummaryArgs,
            handle_validation_error=True,
        ),
    ]
