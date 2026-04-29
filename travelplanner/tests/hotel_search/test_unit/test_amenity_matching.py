"""Unit tests for amenity matching logic."""
from __future__ import annotations

import unittest

from travelplanner.agents.hotel_search_agent import filter_hotels_by_constraints
from travelplanner.schema.hotel_search_artifact import HotelOptionModel


def _make_hotel(name: str, amenities: list[str], rating: float = 8.0) -> HotelOptionModel:
    """Helper to create test hotel."""
    return HotelOptionModel(
        search_result_id=f"offer-{name}",
        accommodation_id=f"hotel-{name}",
        name=name,
        nightly_rate=100.0,
        total_cost=700.0,
        currency="EUR",
        amenities=amenities,
        rating=rating,
        latitude=41.38,
        longitude=2.17,
        over_budget=False,
        over_budget_amount=0.0,
    )


class TestAmenityMatching(unittest.TestCase):
    """Test amenity filtering and fuzzy matching logic."""

    def test_exact_match(self):
        """Test exact amenity matching."""
        hotels = [
            _make_hotel("with-wifi", ["wifi", "parking"]),
            _make_hotel("without-wifi", ["parking", "breakfast"]),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels, required_amenities=["wifi"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "with-wifi")

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        hotels = [_make_hotel("uppercase", ["WiFi", "Parking"])]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels, required_amenities=["wifi"]
        )

        self.assertEqual(len(filtered), 1)

    def test_fuzzy_wifi_variants(self):
        """Test fuzzy matching for wifi variants."""
        test_cases = [
            (["free-wifi"], True),
            (["wireless internet"], True),
            (["internet access"], True),
            (["high-speed internet"], True),
            (["parking"], False),  # Unrelated amenity
        ]

        for amenities, should_match in test_cases:
            hotels = [_make_hotel("test", amenities)]
            filtered, _ = filter_hotels_by_constraints(
                hotels=hotels, required_amenities=["wifi"]
            )

            matched = len(filtered) == 1
            self.assertEqual(
                matched,
                should_match,
                f"Failed for amenities: {amenities}, expected {should_match}, got {matched}",
            )

    def test_fuzzy_pool_variants(self):
        """Test fuzzy matching for pool variants."""
        test_cases = [
            ["swimming-pool"],
            ["outdoor-pool"],
            ["indoor-pool"],
            ["pool"],
        ]

        for amenities in test_cases:
            hotels = [_make_hotel("test", amenities)]
            filtered, _ = filter_hotels_by_constraints(
                hotels=hotels, required_amenities=["pool"]
            )

            self.assertEqual(
                len(filtered), 1, f"Failed to match pool in: {amenities}"
            )

    def test_fuzzy_gym_variants(self):
        """Test fuzzy matching for gym variants."""
        test_cases = [
            ["fitness center"],
            ["workout room"],
            ["exercise facilities"],
            ["gym"],
        ]

        for amenities in test_cases:
            hotels = [_make_hotel("test", amenities)]
            filtered, _ = filter_hotels_by_constraints(
                hotels=hotels, required_amenities=["gym"]
            )

            self.assertEqual(
                len(filtered), 1, f"Failed to match gym in: {amenities}"
            )

    def test_multiple_required_amenities(self):
        """Test that ALL required amenities must be present (AND logic)."""
        hotels = [
            _make_hotel("both", ["wifi", "pool", "parking"]),
            _make_hotel("wifi-only", ["wifi", "parking"]),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels, required_amenities=["wifi", "pool"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "both")

    def test_preferred_amenities_scoring(self):
        """Test that preferred amenities are counted correctly."""
        hotels = [
            _make_hotel("two-preferred", ["wifi", "gym", "breakfast"]),
            _make_hotel("one-preferred", ["wifi", "gym"]),
            _make_hotel("zero-preferred", ["wifi", "parking"]),
        ]

        filtered, preferred_counts = filter_hotels_by_constraints(
            hotels=hotels,
            required_amenities=["wifi"],
            preferred_amenities=["gym", "breakfast"],
        )

        self.assertEqual(len(filtered), 3)
        self.assertEqual(preferred_counts["hotel-two-preferred"], 2)
        self.assertEqual(preferred_counts["hotel-one-preferred"], 1)
        self.assertEqual(preferred_counts["hotel-zero-preferred"], 0)

    def test_min_rating_filter(self):
        """Test minimum rating filtering."""
        hotels = [
            _make_hotel("high-rated", ["wifi"], rating=8.5),
            _make_hotel("low-rated", ["wifi"], rating=6.0),
        ]

        filtered, _ = filter_hotels_by_constraints(
            hotels=hotels, required_amenities=["wifi"], min_rating=7.0
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "high-rated")

    def test_no_filters(self):
        """Test that no filters returns all hotels."""
        hotels = [
            _make_hotel("hotel1", ["wifi"]),
            _make_hotel("hotel2", ["pool"]),
        ]

        filtered, _ = filter_hotels_by_constraints(hotels=hotels)

        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
