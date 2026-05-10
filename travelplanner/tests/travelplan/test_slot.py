"""Unit tests for the Slot model.

Run from travelplanner/ directory:
    uv run python -m pytest tests/travelplan/test_slot.py -v
"""
from __future__ import annotations

import unittest
from datetime import datetime

from pydantic import ValidationError

from travelplanner.travelplan import Slot


def _slot(name: str, start: str, end: str, **kwargs) -> Slot:
    return Slot(
        name=name,
        start_time=datetime.fromisoformat(start),
        end_time=datetime.fromisoformat(end),
        **kwargs,
    )


class TestSlotValidation(unittest.TestCase):
    def test_valid_slot(self):
        s = _slot("Breakfast", "2026-06-01T08:00", "2026-06-01T10:00", cost=15.0)
        self.assertEqual(s.name, "Breakfast")
        self.assertEqual(s.cost, 15.0)
        self.assertEqual(s.category, "other")

    def test_end_before_start_rejected(self):
        with self.assertRaises(ValidationError):
            _slot("Bad", "2026-06-01T10:00", "2026-06-01T08:00")

    def test_zero_length_rejected(self):
        with self.assertRaises(ValidationError):
            _slot("Zero", "2026-06-01T10:00", "2026-06-01T10:00")

    def test_empty_name_rejected(self):
        with self.assertRaises(ValidationError):
            _slot("", "2026-06-01T08:00", "2026-06-01T10:00")

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValidationError):
            _slot("X", "2026-06-01T08:00", "2026-06-01T10:00", cost=-1.0)


class TestSlotOverlap(unittest.TestCase):
    def test_disjoint_no_overlap(self):
        a = _slot("A", "2026-06-01T08:00", "2026-06-01T09:00")
        b = _slot("B", "2026-06-01T10:00", "2026-06-01T11:00")
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))

    def test_boundary_touching_no_overlap(self):
        a = _slot("A", "2026-06-01T08:00", "2026-06-01T10:00")
        b = _slot("B", "2026-06-01T10:00", "2026-06-01T11:00")
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))

    def test_partial_overlap(self):
        a = _slot("A", "2026-06-01T08:00", "2026-06-01T10:00")
        b = _slot("B", "2026-06-01T09:00", "2026-06-01T11:00")
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

    def test_contained_overlap(self):
        outer = _slot("Outer", "2026-06-01T08:00", "2026-06-01T18:00")
        inner = _slot("Inner", "2026-06-01T10:00", "2026-06-01T12:00")
        self.assertTrue(outer.overlaps(inner))
        self.assertTrue(inner.overlaps(outer))

    def test_cross_midnight_overlap(self):
        party = _slot("Party", "2026-06-01T18:00", "2026-06-02T01:00")
        nightcap = _slot("Cap", "2026-06-02T00:00", "2026-06-02T02:00")
        self.assertTrue(party.overlaps(nightcap))


if __name__ == "__main__":
    unittest.main()
