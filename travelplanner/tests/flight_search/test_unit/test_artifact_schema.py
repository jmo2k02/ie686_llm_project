from __future__ import annotations

import unittest

from travelplanner.schema.flight_search_artifact import (
    FlightSearchArtifactContentModel,
)


class TestFlightArtifactSchema(unittest.TestCase):
    def _make_leg(self) -> dict:
        return {
            "departure_airport": {"name": "Frankfurt Airport", "id": "FRA", "time": "2026-06-10 07:00"},
            "arrival_airport": {"name": "Heathrow Airport", "id": "LHR", "time": "2026-06-10 07:40"},
            "duration_minutes": 100,
            "airline": "Lufthansa",
            "flight_number": "LH 918",
            "airplane": "Airbus A320neo",
            "travel_class": "Economy",
            "legroom": "29 in",
            "extensions": ["In-seat USB outlet"],
        }

    def _make_option(self) -> dict:
        return {
            "legs": [self._make_leg()],
            "layovers": [],
            "total_duration_minutes": 100,
            "price": 165.0,
            "currency": "EUR",
            "type": "Round trip",
            "carbon_emissions_kg": 67,
            "departure_token": "abc123",
        }

    def test_required_fields_and_enums(self) -> None:
        payload = {
            "task_ref": "flight-fra-lhr",
            "status": "success",
            "provider": "serpapi_google_flights",
            "departure_id": "FRA",
            "arrival_id": "LHR",
            "outbound_date": "2026-06-10",
            "return_date": "2026-06-15",
            "adults": 1,
            "currency": "EUR",
            "best_flights": [self._make_option()],
            "other_flights": [],
            "price_insights": {
                "lowest_price": 165.0,
                "price_level": "typical",
                "typical_price_range": [160.0, 280.0],
            },
            "errors": [],
            "config": {},
        }
        parsed = FlightSearchArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.provider, "serpapi_google_flights")
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.departure_id, "FRA")
        self.assertEqual(parsed.arrival_id, "LHR")
        self.assertEqual(len(parsed.best_flights), 1)
        self.assertEqual(parsed.best_flights[0].price, 165.0)
        self.assertEqual(parsed.best_flights[0].legs[0].flight_number, "LH 918")

    def test_one_way_trip_no_return_date(self) -> None:
        payload = {
            "task_ref": "flight-one-way",
            "status": "success",
            "provider": "serpapi_google_flights",
            "departure_id": "MUC",
            "arrival_id": "CDG",
            "outbound_date": "2026-07-01",
            "return_date": None,
            "adults": 2,
            "currency": "EUR",
            "best_flights": [],
            "other_flights": [],
            "errors": [],
            "config": {},
        }
        parsed = FlightSearchArtifactContentModel.model_validate(payload)
        self.assertIsNone(parsed.return_date)
        self.assertEqual(parsed.adults, 2)

    def test_failed_status_with_errors(self) -> None:
        payload = {
            "task_ref": "flight-failed",
            "status": "failed",
            "provider": "serpapi_google_flights",
            "departure_id": "",
            "arrival_id": "",
            "outbound_date": "",
            "adults": 1,
            "currency": "EUR",
            "errors": [{"code": "missing_api_key", "message": "SERPAPI_API_KEY is not set"}],
            "config": {},
        }
        parsed = FlightSearchArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "failed")
        self.assertEqual(len(parsed.errors), 1)
        self.assertEqual(parsed.errors[0].code, "missing_api_key")


if __name__ == "__main__":
    unittest.main()
