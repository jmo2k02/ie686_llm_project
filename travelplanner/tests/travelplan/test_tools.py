"""Unit tests for the TravelPlan agent tools.

Run from travelplanner/ directory:
    uv run python -m pytest tests/travelplan/test_tools.py -v
"""
from __future__ import annotations

import unittest

from travelplanner.travelplan import TravelPlan, make_travelplan_tools


class _ToolsFixture(unittest.TestCase):
    def setUp(self):
        self.plan = TravelPlan()
        self.tools = {t.name: t for t in make_travelplan_tools(self.plan)}


class TestToolNames(unittest.TestCase):
    def test_factory_returns_expected_tool_names(self):
        names = {t.name for t in make_travelplan_tools(TravelPlan())}
        self.assertEqual(
            names,
            {
                "init_plan",
                "add_day",
                "remove_day",
                "add_slot",
                "insert_slot",
                "delete_slot",
                "view_plan",
                "cost_summary",
            },
        )


class TestInitPlan(_ToolsFixture):
    def test_init_plan_sets_title_and_clears_days(self):
        self.tools["add_day"].invoke({})
        self.tools["add_day"].invoke({})
        result = self.tools["init_plan"].invoke({"title": "Rome"})
        self.assertIn("Rome", result)
        self.assertEqual(self.plan.title, "Rome")
        self.assertEqual(self.plan.days, [])

    def test_init_plan_without_title(self):
        self.plan.title = "leftover"
        result = self.tools["init_plan"].invoke({})
        self.assertIn("untitled", result.lower())
        self.assertIsNone(self.plan.title)


class TestAddDay(_ToolsFixture):
    def test_add_day_basic(self):
        result = self.tools["add_day"].invoke({"label": "Arrival"})
        self.assertIn("Added day 1", result)
        self.assertEqual(len(self.plan.days), 1)
        self.assertEqual(self.plan.days[0].label, "Arrival")

    def test_add_day_with_iso_date(self):
        result = self.tools["add_day"].invoke({"calendar_date_iso": "2026-06-01"})
        self.assertIn("Added day 1", result)
        self.assertEqual(self.plan.days[0].calendar_date.isoformat(), "2026-06-01")

    def test_add_day_bad_iso_returns_error(self):
        result = self.tools["add_day"].invoke({"calendar_date_iso": "not-a-date"})
        self.assertTrue(result.startswith("Error:"), result)
        self.assertEqual(len(self.plan.days), 0)


class TestRemoveDay(_ToolsFixture):
    def test_remove_day_renumbers(self):
        self.tools["add_day"].invoke({"label": "A"})
        self.tools["add_day"].invoke({"label": "B"})
        self.tools["add_day"].invoke({"label": "C"})
        result = self.tools["remove_day"].invoke({"day_index": 2})
        self.assertIn("Removed day 2", result)
        self.assertEqual([d.label for d in self.plan.days], ["A", "C"])
        self.assertEqual([d.index for d in self.plan.days], [1, 2])

    def test_remove_day_out_of_range(self):
        result = self.tools["remove_day"].invoke({"day_index": 99})
        self.assertTrue(result.startswith("Error:"), result)


class TestAddSlot(_ToolsFixture):
    def _seed_day(self):
        self.tools["add_day"].invoke({})

    def test_add_slot_happy_path(self):
        self._seed_day()
        result = self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "Breakfast",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
                "cost": 15.0,
                "category": "meal",
            }
        )
        self.assertIn("Added slot 'Breakfast'", result)
        self.assertIn("position 1", result)
        self.assertIn("€15.00", result)
        self.assertEqual(self.plan.days[0].slots[0].name, "Breakfast")
        self.assertEqual(self.plan.days[0].slots[0].category, "meal")

    def test_add_slot_overlap_returns_error_and_does_not_mutate(self):
        self._seed_day()
        self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "A",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
            }
        )
        result = self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "B",
                "start_time_iso": "2026-06-01T09:00",
                "end_time_iso": "2026-06-01T11:00",
            }
        )
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("overlaps", result.lower())
        self.assertEqual(len(self.plan.days[0].slots), 1)

    def test_add_slot_bad_iso(self):
        self._seed_day()
        result = self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "X",
                "start_time_iso": "totally-not-iso",
                "end_time_iso": "2026-06-01T10:00",
            }
        )
        self.assertTrue(result.startswith("Error:"), result)
        self.assertEqual(len(self.plan.days[0].slots), 0)

    def test_add_slot_bad_day(self):
        result = self.tools["add_slot"].invoke(
            {
                "day_index": 99,
                "name": "X",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
            }
        )
        self.assertTrue(result.startswith("Error:"), result)

    def test_add_slot_end_before_start(self):
        self._seed_day()
        result = self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "Backwards",
                "start_time_iso": "2026-06-01T10:00",
                "end_time_iso": "2026-06-01T08:00",
            }
        )
        self.assertTrue(result.startswith("Error:"), result)


class TestInsertSlot(_ToolsFixture):
    def test_insert_slot_at_front(self):
        self.tools["add_day"].invoke({})
        self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "Lunch",
                "start_time_iso": "2026-06-01T12:00",
                "end_time_iso": "2026-06-01T13:00",
            }
        )
        result = self.tools["insert_slot"].invoke(
            {
                "day_index": 1,
                "position": 1,
                "name": "Breakfast",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
            }
        )
        self.assertIn("Inserted slot 'Breakfast'", result)
        self.assertEqual(
            [s.name for s in self.plan.days[0].slots], ["Breakfast", "Lunch"]
        )

    def test_insert_slot_out_of_range(self):
        self.tools["add_day"].invoke({})
        result = self.tools["insert_slot"].invoke(
            {
                "day_index": 1,
                "position": 5,
                "name": "Late",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
            }
        )
        self.assertTrue(result.startswith("Error:"), result)


class TestDeleteSlot(_ToolsFixture):
    def test_delete_slot(self):
        self.tools["add_day"].invoke({})
        self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "X",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
                "cost": 5.0,
            }
        )
        result = self.tools["delete_slot"].invoke({"day_index": 1, "position": 1})
        self.assertIn("Deleted slot 'X'", result)
        self.assertIn("0 slot(s)", result)
        self.assertEqual(self.plan.days[0].slots, [])

    def test_delete_slot_out_of_range(self):
        self.tools["add_day"].invoke({})
        result = self.tools["delete_slot"].invoke({"day_index": 1, "position": 99})
        self.assertTrue(result.startswith("Error:"), result)


class TestViewPlan(_ToolsFixture):
    def test_view_empty(self):
        result = self.tools["view_plan"].invoke({})
        self.assertIn("# TravelPlan", result)
        self.assertIn("_No days yet._", result)

    def test_view_populated(self):
        self.tools["init_plan"].invoke({"title": "Demo"})
        self.tools["add_day"].invoke(
            {"label": "Arrival", "calendar_date_iso": "2026-06-01"}
        )
        self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "Breakfast",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T10:00",
                "cost": 15.0,
                "category": "meal",
                "location": "Cafe Roma",
            }
        )
        result = self.tools["view_plan"].invoke({})
        self.assertIn("# TravelPlan: Demo", result)
        self.assertIn("Day 1 — 2026-06-01 — Arrival", result)
        self.assertIn("Breakfast", result)
        self.assertIn("[meal]", result)
        self.assertIn("Cafe Roma", result)
        self.assertIn("€15.00", result)


class TestCostSummary(_ToolsFixture):
    def test_cost_summary_empty(self):
        result = self.tools["cost_summary"].invoke({})
        self.assertIn("€0.00", result)
        self.assertIn("no days", result.lower())

    def test_cost_summary_populated(self):
        self.tools["add_day"].invoke({})
        self.tools["add_day"].invoke({})
        self.tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "A",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T09:00",
                "cost": 10.0,
            }
        )
        self.tools["add_slot"].invoke(
            {
                "day_index": 2,
                "name": "B",
                "start_time_iso": "2026-06-02T08:00",
                "end_time_iso": "2026-06-02T09:00",
                "cost": 25.0,
            }
        )
        result = self.tools["cost_summary"].invoke({})
        self.assertIn("€35.00", result)
        self.assertIn("Day 1: €10.00", result)
        self.assertIn("Day 2: €25.00", result)


class TestSharedClosure(unittest.TestCase):
    def test_tools_share_the_closure_bound_plan(self):
        plan = TravelPlan()
        tools = {t.name: t for t in make_travelplan_tools(plan)}
        tools["add_day"].invoke({"label": "A"})
        tools["add_slot"].invoke(
            {
                "day_index": 1,
                "name": "Z",
                "start_time_iso": "2026-06-01T08:00",
                "end_time_iso": "2026-06-01T09:00",
            }
        )
        self.assertEqual(len(plan.days), 1)
        self.assertEqual(plan.days[0].slots[0].name, "Z")


if __name__ == "__main__":
    unittest.main()
