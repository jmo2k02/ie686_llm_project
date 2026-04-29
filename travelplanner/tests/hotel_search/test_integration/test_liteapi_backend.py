"""Integration tests for LiteAPI backend."""
from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")

from travelplanner.agents.hotel_search_agent import (
    get_hotel_details,
    parse_location,
    search_hotels_via_api,
    search_places,
)


@unittest.skipIf(
    not os.environ.get("LITEAPI_API_KEY"),
    "LITEAPI_API_KEY not set - skipping live API tests",
)
class TestLiteAPIBackend(unittest.TestCase):
    """Test LiteAPI integration (requires API key)."""

    def test_search_places(self):
        """Test place search returns valid place ID."""
        result = search_places("Barcelona, Spain")

        self.assertIn("placeId", result)
        self.assertIn("displayName", result)
        self.assertIsNotNone(result["placeId"])

    def test_search_hotels_via_api(self):
        """Test hotel search returns results."""
        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        city_name, country_code = parse_location("Barcelona, Spain")

        response = search_hotels_via_api(
            place_id=None,
            city_name=city_name,
            country_code=country_code,
            check_in_date=check_in,
            check_out_date=check_out,
            guest_count=2,
        )

        self.assertEqual(response.get("status"), "success")
        self.assertIn("data", response)
        self.assertIn("hotels", response)
        self.assertGreater(len(response["hotels"]), 0)

        # Check hotel structure
        first_hotel = response["hotels"][0]
        self.assertIn("id", first_hotel)
        self.assertIn("name", first_hotel)
        self.assertIn("rating", first_hotel)

    def test_get_hotel_details(self):
        """Test fetching hotel details with amenities."""
        # First get a hotel ID from search
        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        city_name, country_code = parse_location("Barcelona, Spain")

        search_response = search_hotels_via_api(
            place_id=None,
            city_name=city_name,
            country_code=country_code,
            check_in_date=check_in,
            check_out_date=check_out,
            guest_count=2,
        )

        self.assertEqual(search_response.get("status"), "success")
        hotels = search_response.get("hotels", [])
        self.assertGreater(len(hotels), 0)

        # Fetch details for first hotel
        hotel_id = hotels[0]["id"]
        details_response = get_hotel_details(hotel_id, timeout=4)

        self.assertEqual(details_response.get("status"), "success")
        hotel_data = details_response.get("hotel", {})

        self.assertIn("id", hotel_data)
        self.assertIn("name", hotel_data)
        self.assertIn("hotelFacilities", hotel_data)

        # Check that facilities are returned
        facilities = hotel_data.get("hotelFacilities", [])
        self.assertIsInstance(facilities, list)
        # Most hotels should have some facilities
        if len(facilities) > 0:
            self.assertIsInstance(facilities[0], str)

    def test_search_with_place_id(self):
        """Test hotel search using placeId."""
        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        # Get place ID first
        place_result = search_places("Barcelona, Spain")
        place_id = place_result.get("placeId")

        self.assertIsNotNone(place_id)

        # Search with place ID
        response = search_hotels_via_api(
            place_id=place_id,
            city_name=None,
            country_code=None,
            check_in_date=check_in,
            check_out_date=check_out,
            guest_count=2,
        )

        self.assertEqual(response.get("status"), "success")
        self.assertGreater(len(response.get("hotels", [])), 0)


if __name__ == "__main__":
    unittest.main()
