"""Unit tests for the Day model.

Run from travelplanner/ directory:
    uv run python -m pytest tests/travelplan/test_day.py -v
"""
from __future__ import annotations

import unittest
from datetime import datetime

from travelplanner.travelplan import Day, Slot, SlotNotFoundError, SlotOverlapError


def _slot(name: str, start: str, end: str, cost: float | None = None) -> Slot:
    return Slot(
        name=name,
        start_time=datetime.fromisoformat(start),
        end_time=datetime.fromisoformat(end),
        cost=cost,
    )


class TestDaySlotOps(unittest.TestCase):
    def test_append_returns_one_based_position(self):
        day = Day(index=1)
        pos = day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))
        self.assertEqual(pos, 1)
        pos = day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T10:00"))
        self.assertEqual(pos, 2)
        self.assertEqual(len(day.slots), 2)

    def test_append_rejects_overlap(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T10:00"))
        with self.assertRaises(SlotOverlapError):
            day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T11:00"))

    def test_append_allows_boundary_touching(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T10:00"))
        day.append_slot(_slot("B", "2026-06-01T10:00", "2026-06-01T11:00"))
        self.assertEqual(len(day.slots), 2)

    def test_insert_at_front(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))
        day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T10:00"))
        day.insert_slot(1, _slot("C", "2026-06-01T07:00", "2026-06-01T08:00"))
        self.assertEqual([s.name for s in day.slots], ["C", "A", "B"])

    def test_insert_at_end(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))
        day.insert_slot(2, _slot("B", "2026-06-01T09:00", "2026-06-01T10:00"))
        self.assertEqual([s.name for s in day.slots], ["A", "B"])

    def test_insert_out_of_range_raises(self):
        day = Day(index=1)
        with self.assertRaises(SlotNotFoundError):
            day.insert_slot(2, _slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))
        with self.assertRaises(SlotNotFoundError):
            day.insert_slot(0, _slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))

    def test_insert_rejects_overlap(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T10:00"))
        with self.assertRaises(SlotOverlapError):
            day.insert_slot(1, _slot("B", "2026-06-01T09:00", "2026-06-01T11:00"))

    def test_delete_returns_slot_and_shrinks(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00"))
        day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T10:00"))
        removed = day.delete_slot(1)
        self.assertEqual(removed.name, "A")
        self.assertEqual([s.name for s in day.slots], ["B"])

    def test_delete_out_of_range_raises(self):
        day = Day(index=1)
        with self.assertRaises(SlotNotFoundError):
            day.delete_slot(1)

    def test_sorted_slots_does_not_mutate(self):
        day = Day(index=1)
        day.append_slot(_slot("Late", "2026-06-01T18:00", "2026-06-01T19:00"))
        day.append_slot(_slot("Early", "2026-06-01T08:00", "2026-06-01T09:00"))
        sorted_view = day.sorted_slots()
        self.assertEqual([s.name for s in sorted_view], ["Early", "Late"])
        self.assertEqual([s.name for s in day.slots], ["Late", "Early"])


class TestDayCost(unittest.TestCase):
    def test_total_cost_sums_known_costs(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00", cost=10.0))
        day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T10:00", cost=5.5))
        self.assertAlmostEqual(day.total_cost(), 15.5)

    def test_total_cost_treats_none_as_zero(self):
        day = Day(index=1)
        day.append_slot(_slot("A", "2026-06-01T08:00", "2026-06-01T09:00", cost=None))
        day.append_slot(_slot("B", "2026-06-01T09:00", "2026-06-01T10:00", cost=20.0))
        self.assertAlmostEqual(day.total_cost(), 20.0)

    def test_total_cost_empty_day_is_zero(self):
        self.assertEqual(Day(index=1).total_cost(), 0.0)


if __name__ == "__main__":
    unittest.main()