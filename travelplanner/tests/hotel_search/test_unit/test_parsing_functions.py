"""Unit tests for parsing functions (dates, location, nights)."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from travelplanner.agents.hotel_search_agent import (
    calculate_nights,
    parse_date_range,
    parse_location,
)


class TestParsingFunctions(unittest.TestCase):
    """Test date and location parsing utilities."""

    def test_parse_date_range_with_to_separator(self):
        """Test parsing date range with 'to' separator."""
        check_in, check_out = parse_date_range("2026-06-01 to 2026-06-07")
        self.assertEqual(check_in, "2026-06-01")
        self.assertEqual(check_out, "2026-06-07")

    def test_parse_date_range_with_dash_separator(self):
        """Test parsing date range with '-' separator."""
        check_in, check_out = parse_date_range("2026-06-01 - 2026-06-07")
        self.assertEqual(check_in, "2026-06-01")
        self.assertEqual(check_out, "2026-06-07")

    def test_parse_date_range_rejects_invalid_format(self):
        """Test that invalid date formats are rejected."""
        check_in, check_out = parse_date_range("2026/06/01 to 2026/06/07")
        self.assertIsNone(check_in)
        self.assertIsNone(check_out)

    def test_parse_date_range_rejects_past_dates(self):
        """Test that past dates are rejected."""
        check_in, check_out = parse_date_range("2020-01-01 to 2020-01-07")
        self.assertIsNone(check_in)
        self.assertIsNone(check_out)

    def test_parse_date_range_rejects_invalid_order(self):
        """Test that check_out before check_in is rejected."""
        check_in, check_out = parse_date_range("2026-06-07 to 2026-06-01")
        self.assertIsNone(check_in)
        self.assertIsNone(check_out)

    def test_parse_location_full_address(self):
        """Test parsing full address with neighborhood."""
        city, country = parse_location("Eixample, Barcelona, Spain")
        self.assertEqual(city, "Barcelona")
        self.assertEqual(country, "ES")

    def test_parse_location_city_country(self):
        """Test parsing simple city, country format."""
        city, country = parse_location("Barcelona, Spain")
        self.assertEqual(city, "Barcelona")
        self.assertEqual(country, "ES")

    def test_parse_location_handles_uk(self):
        """Test UK is converted to GB code."""
        city, country = parse_location("London, UK")
        self.assertEqual(city, "London")
        self.assertEqual(country, "GB")

    def test_parse_location_invalid_format(self):
        """Test that invalid location format returns None."""
        city, country = parse_location("Barcelona")
        self.assertIsNone(city)
        self.assertIsNone(country)

    def test_calculate_nights(self):
        """Test nights calculation."""
        nights = calculate_nights("2026-06-01", "2026-06-07")
        self.assertEqual(nights, 6)

    def test_calculate_nights_single_night(self):
        """Test minimum 1 night is returned."""
        nights = calculate_nights("2026-06-01", "2026-06-01")
        self.assertEqual(nights, 1)


if __name__ == "__main__":
    unittest.main()
