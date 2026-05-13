"""Unit tests for the TravelPlan model.

Run from travelplanner/ directory:
    uv run python -m pytest tests/travelplan/test_plan.py -v
"""
from __future__ import annotations

import unittest
from datetime import date, datetime

from travelplanner.travelplan import (
    DayNotFoundError,
    Slot,
    SlotOverlapError,
    TravelPlan,
)


def _slot(name: str, start: str, end: str, cost: float | None = None, **kwargs) -> Slot:
    return Slot(
        name=name,
        start_time=datetime.fromisoformat(start),
        end_time=datetime.fromisoformat(end),
        cost=cost,
        **kwargs,
    )


class TestTravelPlanDays(unittest.TestCase):
    def test_add_day_assigns_one_based_index(self):
        plan = TravelPlan()
        d1 = plan.add_day()
        d2 = plan.add_day(label="Arrival", calendar_date=date(2026, 6, 1))
        self.assertEqual(d1.index, 1)
        self.assertEqual(d2.index, 2)
        self.assertEqual(d2.label, "Arrival")
        self.assertEqual(d2.calendar_date, date(2026, 6, 1))

    def test_remove_day_renumbers(self):
        plan = TravelPlan()
        plan.add_day(label="A")
        plan.add_day(label="B")
        plan.add_day(label="C")
        plan.remove_day(2)
        self.assertEqual([d.label for d in plan.days], ["A", "C"])
        self.assertEqual([d.index for d in plan.days], [1, 2])

    def test_get_day_out_of_range_raises(self):
        plan = TravelPlan()
        with self.assertRaises(DayNotFoundError):
            plan.get_day(1)
        plan.add_day()
        with self.assertRaises(DayNotFoundError):
            plan.get_day(2)
        with self.assertRaises(DayNotFoundError):
            plan.get_day(0)


class TestTravelPlanSlots(unittest.TestCase):
    def test_add_slot_delegates(self):
        plan = TravelPlan()
        plan.add_day()
        pos = plan.add_slot(1, _slot("Breakfast", "2026-06-01T08:00", "2026-06-01T10:00"))
        self.assertEqual(pos, 1)
        self.assertEqual(plan.days[0].slots[0].name, "Breakfast")

    def test_overlap_propagates(self):
        plan = TravelPlan()
        plan.add_day()
        plan.add_slot(1, _slot("A", "2026-06-01T08:00", "2026-06-01T10:00"))
        with self.assertRaises(SlotOverlapError):
            plan.add_slot(1, _slot("B", "2026-06-01T09:00", "2026-06-01T11:00"))

    def test_delete_slot_returns_removed(self):
        plan = TravelPlan()
        plan.add_day()
        plan.add_slot(1, _slot("A", "2026-06-01T08:00", "2026-06-01T10:00"))
        removed = plan.delete_slot(1, 1)
        self.assertEqual(removed.name, "A")
        self.assertEqual(plan.days[0].slots, [])


class TestTravelPlanCost(unittest.TestCase):
    def test_total_and_daily_costs(self):
        plan = TravelPlan()
        plan.add_day()
        plan.add_day()
        plan.add_slot(1, _slot("A", "2026-06-01T08:00", "2026-06-01T09:00", cost=10.0))
        plan.add_slot(1, _slot("B", "2026-06-01T09:00", "2026-06-01T10:00", cost=5.0))
        plan.add_slot(2, _slot("C", "2026-06-02T08:00", "2026-06-02T09:00", cost=20.0))

        self.assertAlmostEqual(plan.total_cost(), 35.0)
        self.assertEqual(plan.daily_costs(), {1: 15.0, 2: 20.0})

        summary = plan.cost_summary()
        self.assertAlmostEqual(summary.total, 35.0)
        self.assertEqual(summary.per_day, {1: 15.0, 2: 20.0})

    def test_empty_plan_costs_zero(self):
        plan = TravelPlan()
        self.assertEqual(plan.total_cost(), 0.0)
        self.assertEqual(plan.daily_costs(), {})


class TestTravelPlanMarkdown(unittest.TestCase):
    def test_empty_plan_renders_placeholder(self):
        md = TravelPlan().to_markdown()
        self.assertIn("# TravelPlan", md)
        self.assertIn("_No days yet._", md)

    def test_titled_plan_uses_title(self):
        md = TravelPlan(title="Rome 3-day").to_markdown()
        self.assertIn("# TravelPlan: Rome 3-day", md)

    def test_populated_plan_renders_table(self):
        plan = TravelPlan(title="Demo")
        plan.add_day(label="Arrival", calendar_date=date(2026, 6, 1))
        plan.add_day(calendar_date=date(2026, 6, 2))
        plan.add_slot(
            1,
            _slot(
                "Breakfast",
                "2026-06-01T08:00",
                "2026-06-01T10:00",
                cost=15.0,
                category="meal",
                location="Cafe Roma",
                links=["https://example.com/cafe"],
            ),
        )
        plan.add_slot(
            1,
            _slot(
                "Museum",
                "2026-06-01T11:00",
                "2026-06-01T14:00",
                cost=20.0,
                category="attraction",
            ),
        )
        plan.add_slot(
            2,
            _slot("Hike", "2026-06-02T09:00", "2026-06-02T13:00", category="attraction"),
        )

        md = plan.to_markdown()
        # Header row contains both day labels
        self.assertIn("Day 1 — 2026-06-01 — Arrival", md)
        self.assertIn("Day 2 — 2026-06-02", md)
        # Slot bodies
        self.assertIn("**1. Breakfast** [meal] 08:00–10:00 @ Cafe Roma (€15.00)", md)
        self.assertIn("Links: [link 1](https://example.com/cafe)", md)
        self.assertIn("**2. Museum** [attraction] 11:00–14:00 (€20.00)", md)
        self.assertIn("**1. Hike** [attraction] 09:00–13:00", md)
        # Cost summary
        self.assertIn("Total estimated cost: €35.00", md)
        self.assertIn("Day 1: €35.00", md)
        self.assertIn("Day 2: €0.00", md)

    def test_cross_midnight_slot_renders_full_dates(self):
        plan = TravelPlan()
        plan.add_day()
        plan.add_slot(
            1,
            _slot("Party", "2026-06-01T18:00", "2026-06-02T01:00", category="leisure"),
        )
        md = plan.to_markdown()
        self.assertIn("2026-06-01 18:00", md)
        self.assertIn("2026-06-02 01:00", md)

    def test_compact_render_drops_category_location_cost_description(self):
        plan = TravelPlan(title="Compact demo")
        plan.add_day(label="Arrival", calendar_date=date(2026, 6, 1))
        plan.add_slot(
            1,
            _slot(
                "Breakfast",
                "2026-06-01T08:00",
                "2026-06-01T10:00",
                cost=15.0,
                category="meal",
                location="Cafe Roma",
                links=["https://example.com/cafe"],
            ),
        )
        md = plan.to_markdown_compact()
        # Required: position, name, time range
        self.assertIn("**1. Breakfast** 08:00–10:00", md)
        self.assertIn("Links: [link 1](https://example.com/cafe)", md)
        self.assertIn("Day 1 — 2026-06-01 — Arrival", md)
        # Excluded: category, location, cost, cost summary line
        self.assertNotIn("[meal]", md)
        self.assertNotIn("Cafe Roma", md)
        self.assertNotIn("€15.00", md)
        self.assertNotIn("Total estimated cost", md)

    def test_compact_render_empty_plan(self):
        md = TravelPlan().to_markdown_compact()
        self.assertIn("# TravelPlan", md)
        self.assertIn("_No days yet._", md)


class TestTravelPlanIcal(unittest.TestCase):
    def test_empty_plan_renders_calendar_envelope(self):
        ics = TravelPlan().to_ical()
        self.assertIn("BEGIN:VCALENDAR", ics)
        self.assertIn("END:VCALENDAR", ics)
        self.assertIn("VERSION:2.0", ics)
        self.assertIn("PRODID:-//TravelPlanner//TravelPlan//EN", ics)
        self.assertNotIn("BEGIN:VEVENT", ics)

    def test_populated_plan_emits_vevents(self):
        plan = TravelPlan(title="Rome demo")
        plan.add_day(label="Arrival", calendar_date=date(2026, 6, 1))
        plan.add_slot(
            1,
            _slot(
                "Breakfast",
                "2026-06-01T08:00",
                "2026-06-01T10:00",
                cost=15.0,
                category="meal",
                location="Cafe Roma",
                links=["https://example.com/cafe"],
            ),
        )
        plan.add_slot(
            1,
            _slot("Museum", "2026-06-01T11:00", "2026-06-01T14:00", category="attraction"),
        )

        ics = plan.to_ical()
        self.assertEqual(ics.count("BEGIN:VEVENT"), 2)
        self.assertEqual(ics.count("END:VEVENT"), 2)
        self.assertIn("X-WR-CALNAME:Rome demo", ics)
        self.assertIn("SUMMARY:Breakfast", ics)
        self.assertIn("SUMMARY:Museum", ics)
        self.assertIn("DTSTART:20260601T080000", ics)
        self.assertIn("DTEND:20260601T100000", ics)
        self.assertIn("LOCATION:Cafe Roma", ics)
        self.assertIn("Category: meal", ics)
        self.assertIn("Link 1: https://example.com/cafe", ics)
        # Cost should be formatted with EUR prefix.
        self.assertIn(r"Cost: €15.00", ics)

    def test_uid_is_deterministic(self):
        def build():
            plan = TravelPlan(title="UID stability")
            plan.add_day()
            plan.add_slot(
                1,
                _slot("Breakfast", "2026-06-01T08:00", "2026-06-01T10:00"),
            )
            return plan

        ics_a = build().to_ical()
        ics_b = build().to_ical()
        # DTSTAMP differs (file generation time), but UID lines must be identical.
        uids_a = [line for line in ics_a.splitlines() if line.startswith("UID:")]
        uids_b = [line for line in ics_b.splitlines() if line.startswith("UID:")]
        self.assertEqual(uids_a, uids_b)
        self.assertEqual(len(uids_a), 1)

    def test_special_characters_are_escaped(self):
        plan = TravelPlan()
        plan.add_day()
        plan.add_slot(
            1,
            _slot(
                "Dinner, with friends",
                "2026-06-01T19:00",
                "2026-06-01T21:00",
                description="line one\nline two; with semi",
            ),
        )
        ics = plan.to_ical()
        # Comma and semicolon in TEXT must be backslash-escaped.
        self.assertIn(r"SUMMARY:Dinner\, with friends", ics)
        self.assertIn(r"semi", ics)
        # Newline becomes literal backslash-n.
        self.assertIn(r"line one\nline two", ics)


if __name__ == "__main__":
    unittest.main()
