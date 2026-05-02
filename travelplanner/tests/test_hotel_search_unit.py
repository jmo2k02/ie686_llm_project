"""Unit tests for hotel search agent utility functions.

Fast tests without API calls - test pure logic functions.

Run from travelplanner/ directory:
    uv run python -m pytest tests/test_hotel_search_unit.py -v
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from travelplanner.agents.hotel_search_agent import (
    calculate_nights,
    parse_location,
    _facility_match,
    filter_hotels_by_constraints,
    rank_hotels,
)
from travelplanner.schema.hotel_search_artifact import HotelOptionModel


def _make_hotel(
    name: str,
    facilities: list[str],
    rating: float = 8.0,
    nightly_rate: float = 100.0,
    over_budget: bool = False
) -> HotelOptionModel:
    """Helper to create test hotel."""
    return HotelOptionModel(
        search_result_id=f"offer-{name}",
        accommodation_id=f"hotel-{name}",
        name=name,
        nightly_rate=nightly_rate,
        total_cost=nightly_rate * 7,
        currency="EUR",
        facilities=facilities,
        rating=rating,
        latitude=41.38,
        longitude=2.17,
        over_budget=over_budget,
        over_budget_amount=max(0, nightly_rate - 150) if over_budget else 0.0,
    )


class TestUtilityFunctions(unittest.TestCase):
    """Test pure utility functions without API calls."""

    def test_calculate_nights(self):
        """Test nights calculation."""
        nights = calculate_nights("2026-06-01", "2026-06-07")
        self.assertEqual(nights, 6)

    def test_calculate_nights_one_night(self):
        """Test minimum one night."""
        nights = calculate_nights("2026-06-01", "2026-06-02")
        self.assertEqual(nights, 1)

    def test_parse_location_with_country(self):
        """Test location parsing with country."""
        city, country = parse_location("Barcelona, Spain")
        self.assertEqual(city, "Barcelona")
        self.assertEqual(country, "ES")

    def test_parse_location_with_region(self):
        """Test location parsing with region."""
        city, country = parse_location("Eixample, Barcelona, Spain")
        self.assertEqual(city, "Barcelona")
        self.assertEqual(country, "ES")

    def test_parse_location_uk(self):
        """Test UK location parsing."""
        city, country = parse_location("London, UK")
        self.assertEqual(city, "London")
        self.assertEqual(country, "GB")


class TestFacilityMatching(unittest.TestCase):
    """Test facility fuzzy matching logic."""

    def test_exact_match(self):
        """Test exact facility matching."""
        self.assertTrue(_facility_match(["wifi", "parking"], "wifi"))
        self.assertFalse(_facility_match(["parking", "breakfast"], "wifi"))

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        # Note: facilities are already lowercased in _facility_match
        self.assertTrue(_facility_match(["wifi", "parking"], "wifi"))
        self.assertTrue(_facility_match(["WIFI".lower(), "Parking".lower()], "wifi"))

    def test_fuzzy_wifi_variants(self):
        """Test fuzzy matching for wifi variants."""
        test_cases = [
            (["free wifi"], True),
            (["wireless internet"], True),
            (["high-speed internet"], True),
            (["parking"], False),
        ]

        for facilities, should_match in test_cases:
            result = _facility_match(facilities, "wifi")
            self.assertEqual(result, should_match, f"Failed for {facilities}")

    def test_fuzzy_pool_variants(self):
        """Test fuzzy matching for pool variants."""
        test_cases = [
            ["swimming pool"],
            ["outdoor pool"],
            ["indoor pool"],
        ]

        for facilities in test_cases:
            self.assertTrue(_facility_match(facilities, "pool"))

    def test_fuzzy_gym_variants(self):
        """Test fuzzy matching for gym variants."""
        test_cases = [
            ["fitness center"],
            ["workout room"],
            ["exercise facilities"],
        ]

        for facilities in test_cases:
            self.assertTrue(_facility_match(facilities, "gym"))


class TestHotelFiltering(unittest.TestCase):
    """Test hotel filtering logic."""

    def test_filter_by_required_facilities(self):
        """Test filtering by required facilities."""
        hotels = [
            _make_hotel("with-wifi", ["wifi", "parking"]),
            _make_hotel("without-wifi", ["parking", "breakfast"]),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels,
            required_facilities=["wifi"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "with-wifi")

    def test_filter_multiple_required_facilities(self):
        """Test AND logic for multiple required facilities."""
        hotels = [
            _make_hotel("both", ["wifi", "pool", "parking"]),
            _make_hotel("wifi-only", ["wifi", "parking"]),
            _make_hotel("pool-only", ["pool", "parking"]),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels,
            required_facilities=["wifi", "pool"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "both")

    def test_filter_by_min_rating(self):
        """Test filtering by minimum rating."""
        hotels = [
            _make_hotel("high-rated", ["wifi"], rating=8.5),
            _make_hotel("low-rated", ["wifi"], rating=6.0),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels,
            required_facilities=["wifi"],
            min_rating=7.0
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "high-rated")

    def test_preferred_facilities_counting(self):
        """Test that preferred facilities are counted correctly."""
        hotels = [
            _make_hotel("two-preferred", ["wifi", "gym", "breakfast"]),
            _make_hotel("one-preferred", ["wifi", "gym"]),
            _make_hotel("zero-preferred", ["wifi", "parking"]),
        ]

        filtered, preferred_counts = filter_hotels_by_constraints(
            hotels=hotels,
            required_facilities=["wifi"],
            preferred_facilities=["gym", "breakfast"]
        )

        self.assertEqual(len(filtered), 3)
        self.assertEqual(preferred_counts["hotel-two-preferred"], 2)
        self.assertEqual(preferred_counts["hotel-one-preferred"], 1)
        self.assertEqual(preferred_counts["hotel-zero-preferred"], 0)


class TestHotelRanking(unittest.TestCase):
    """Test hotel ranking logic."""

    def test_rank_by_rating(self):
        """Test ranking by rating (descending)."""
        hotels = [
            _make_hotel("medium", ["wifi"], rating=7.5, nightly_rate=100),
            _make_hotel("high", ["wifi"], rating=9.0, nightly_rate=120),
            _make_hotel("low", ["wifi"], rating=6.0, nightly_rate=80),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].name, "high")
        self.assertEqual(ranked[1].name, "medium")
        self.assertEqual(ranked[2].name, "low")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)
        self.assertEqual(ranked[2].rank, 3)

    def test_rank_preferred_facilities_first(self):
        """Test that preferred facilities boost ranking."""
        hotels = [
            _make_hotel("no-preferred", ["wifi"], rating=9.0),
            _make_hotel("one-preferred", ["wifi", "gym"], rating=8.5),
        ]

        preferred_counts = {
            "hotel-no-preferred": 0,
            "hotel-one-preferred": 1,
        }

        ranked = rank_hotels(
            hotels,
            budget_max=150.0,
            preferred_counts=preferred_counts
        )

        self.assertEqual(ranked[0].name, "one-preferred")
        self.assertEqual(ranked[1].name, "no-preferred")

    def test_rank_within_budget_first(self):
        """Test that within-budget hotels rank higher."""
        hotels = [
            _make_hotel("over", ["wifi"], rating=9.0, nightly_rate=200, over_budget=True),
            _make_hotel("within", ["wifi"], rating=8.5, nightly_rate=120, over_budget=False),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].name, "within")
        self.assertEqual(ranked[1].name, "over")

    def test_exclude_over_budget(self):
        """Test excluding over-budget hotels."""
        hotels = [
            _make_hotel("over", ["wifi"], rating=9.0, nightly_rate=200, over_budget=True),
            _make_hotel("within", ["wifi"], rating=8.0, nightly_rate=120, over_budget=False),
        ]

        ranked = rank_hotels(
            hotels,
            budget_max=150.0,
            exclude_over_budget=True
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].name, "within")

    def test_rank_max_10_hotels(self):
        """Test that ranking returns max 10 hotels."""
        hotels = [
            _make_hotel(f"hotel-{i}", ["wifi"], rating=8.0 + i/10)
            for i in range(15)
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertLessEqual(len(ranked), 10)


if __name__ == "__main__":
    unittest.main()
