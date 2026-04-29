"""Unit tests for hotel filtering and ranking logic."""
from __future__ import annotations

import unittest

from travelplanner.agents.hotel_search_agent import rank_hotels
from travelplanner.schema.hotel_search_artifact import HotelOptionModel


def _make_hotel(
    name: str,
    nightly_rate: float,
    rating: float = 8.0,
    amenities: list[str] = None,
) -> HotelOptionModel:
    """Helper to create test hotel."""
    return HotelOptionModel(
        search_result_id=f"offer-{name}",
        accommodation_id=f"hotel-{name}",
        name=name,
        nightly_rate=nightly_rate,
        total_cost=nightly_rate * 7,
        currency="EUR",
        amenities=amenities or [],
        rating=rating,
        latitude=41.38,
        longitude=2.17,
        over_budget=nightly_rate > 150.0,
        over_budget_amount=max(0, nightly_rate - 150.0),
    )


class TestHotelRanking(unittest.TestCase):
    """Test hotel ranking logic."""

    def test_within_budget_ranked_first(self):
        """Test that within-budget hotels are ranked before over-budget."""
        hotels = [
            _make_hotel("expensive", 300.0, rating=9.0),
            _make_hotel("cheap", 100.0, rating=7.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].name, "cheap")
        self.assertEqual(ranked[1].name, "expensive")

    def test_within_budget_sorted_by_rating(self):
        """Test that within-budget hotels are sorted by rating descending."""
        hotels = [
            _make_hotel("mid-rating", 100.0, rating=8.0),
            _make_hotel("high-rating", 120.0, rating=9.0),
            _make_hotel("low-rating", 80.0, rating=7.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].name, "high-rating")
        self.assertEqual(ranked[1].name, "mid-rating")
        self.assertEqual(ranked[2].name, "low-rating")

    def test_within_budget_sorted_by_price_when_rating_equal(self):
        """Test that within-budget hotels with same rating are sorted by price."""
        hotels = [
            _make_hotel("expensive", 140.0, rating=8.0),
            _make_hotel("cheap", 100.0, rating=8.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].name, "cheap")
        self.assertEqual(ranked[1].name, "expensive")

    def test_preferred_amenities_affect_ranking(self):
        """Test that preferred amenities affect ranking within budget."""
        hotels = [
            _make_hotel("basic", 100.0, rating=8.0, amenities=["wifi"]),
            _make_hotel("premium", 100.0, rating=8.0, amenities=["wifi", "gym", "pool"]),
        ]

        preferred_counts = {
            "hotel-basic": 0,
            "hotel-premium": 2,
        }

        ranked = rank_hotels(
            hotels, budget_max=150.0, preferred_counts=preferred_counts
        )

        self.assertEqual(ranked[0].name, "premium")
        self.assertEqual(ranked[1].name, "basic")

    def test_over_budget_sorted_by_price(self):
        """Test that over-budget hotels are sorted by price ascending."""
        hotels = [
            _make_hotel("very-expensive", 300.0),
            _make_hotel("expensive", 200.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].name, "expensive")
        self.assertEqual(ranked[1].name, "very-expensive")

    def test_exclude_over_budget(self):
        """Test that over-budget hotels can be excluded."""
        hotels = [
            _make_hotel("within", 100.0),
            _make_hotel("over", 200.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0, exclude_over_budget=True)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].name, "within")

    def test_max_results_limit(self):
        """Test that max_results limits output."""
        hotels = [_make_hotel(f"hotel-{i}", 100.0) for i in range(20)]

        ranked = rank_hotels(hotels, budget_max=150.0, max_results=5)

        self.assertEqual(len(ranked), 5)

    def test_rank_field_is_set(self):
        """Test that rank field is set sequentially."""
        hotels = [
            _make_hotel("first", 100.0, rating=9.0),
            _make_hotel("second", 120.0, rating=8.0),
        ]

        ranked = rank_hotels(hotels, budget_max=150.0)

        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)

    def test_empty_list(self):
        """Test that empty list returns empty."""
        ranked = rank_hotels([], budget_max=150.0)
        self.assertEqual(len(ranked), 0)


if __name__ == "__main__":
    unittest.main()
