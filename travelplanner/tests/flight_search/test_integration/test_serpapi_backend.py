from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from travelplanner.agents.flight_search_agent import (
    FlightSearchConfig,
    _search_flights,
    _normalize_flight_option,
    _compute_status,
)
from travelplanner.schema.flight_search_artifact import FlightParamsModel, FlightSegmentParams


_SAMPLE_SERPAPI_RESPONSE = {
    "search_metadata": {"status": "Success"},
    "best_flights": [
        {
            "flights": [
                {
                    "departure_airport": {"name": "Frankfurt Airport", "id": "FRA", "time": "2026-06-10 07:00"},
                    "arrival_airport": {"name": "Heathrow Airport", "id": "LHR", "time": "2026-06-10 07:40"},
                    "duration": 100,
                    "airplane": "Airbus A320neo",
                    "airline": "Lufthansa City Airlines",
                    "flight_number": "VL 924",
                    "travel_class": "Economy",
                    "legroom": "29 in",
                    "extensions": ["In-seat USB outlet"],
                }
            ],
            "layovers": [],
            "total_duration": 100,
            "carbon_emissions": {"this_flight": 67000, "typical_for_this_route": 69000},
            "price": 165,
            "type": "Round trip",
            "departure_token": "abc123",
        }
    ],
    "other_flights": [],
    "price_insights": {
        "lowest_price": 165,
        "price_level": "typical",
        "typical_price_range": [160, 280],
    },
}


class TestSerpApiBackend(unittest.TestCase):
    def test_missing_api_key_returns_error_payload(self) -> None:
        config = FlightSearchConfig(api_key="")
        params = FlightParamsModel(
            trip_type=1,
            segments=[FlightSegmentParams(departure_id="FRA", arrival_id="LHR", outbound_date="2026-06-10")],
            return_date="2026-06-15",
        )
        result = _search_flights(params, config)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_api_key")

    def test_success_response_is_normalized(self) -> None:
        config = FlightSearchConfig(api_key="test-key")
        params = FlightParamsModel(
            trip_type=1,
            segments=[FlightSegmentParams(departure_id="FRA", arrival_id="LHR", outbound_date="2026-06-10")],
            return_date="2026-06-15",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = _SAMPLE_SERPAPI_RESPONSE
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.flight_search_agent.requests.get", return_value=mock_response):
            result = _search_flights(params, config)

        self.assertTrue(result["ok"])
        self.assertIn("best_flights", result["raw"])
        self.assertEqual(len(result["raw"]["best_flights"]), 1)

    def test_normalize_flight_option_maps_fields(self) -> None:
        raw_option = _SAMPLE_SERPAPI_RESPONSE["best_flights"][0]
        option = _normalize_flight_option(raw_option, "EUR")
        self.assertEqual(option.price, 165.0)
        self.assertEqual(option.currency, "EUR")
        self.assertEqual(option.total_duration_minutes, 100)
        self.assertEqual(option.carbon_emissions_kg, 67)
        self.assertEqual(len(option.legs), 1)
        self.assertEqual(option.legs[0].departure_airport.id, "FRA")
        self.assertEqual(option.legs[0].arrival_airport.id, "LHR")

    def test_compute_status_variants(self) -> None:
        self.assertEqual(_compute_status(True, 3, 0), "success")
        self.assertEqual(_compute_status(True, 0, 2), "success")
        self.assertEqual(_compute_status(True, 0, 0), "partial")
        self.assertEqual(_compute_status(False, 0, 0), "failed")

    def test_timeout_returns_error_payload(self) -> None:
        import requests as req_lib
        config = FlightSearchConfig(api_key="test-key")
        params = FlightParamsModel(
            trip_type=2,
            segments=[FlightSegmentParams(departure_id="FRA", arrival_id="LHR", outbound_date="2026-06-10")],
        )

        with patch("travelplanner.agents.flight_search_agent.requests.get", side_effect=req_lib.Timeout):
            result = _search_flights(params, config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout_error")


if __name__ == "__main__":
    unittest.main()
