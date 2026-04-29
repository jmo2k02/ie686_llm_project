from __future__ import annotations

import unittest

from travelplanner.schema.hotel_search_artifact import (
    HotelSearchArtifactContentModel,
    HotelSearchCoordinatesModel,
    HotelSearchMetadataModel,
    HotelSearchParametersModel,
    HotelSearchErrorModel,
    HotelOptionModel,
)


class TestHotelSearchArtifactSchema(unittest.TestCase):
    def test_required_fields_and_enums(self) -> None:
        payload = {
            "task_ref": "hotel-search-1",
            "status": "success",
            "attempt": 1,
            "search_parameters": {
                "location": "Barcelona, Spain",
                "check_in_date": "2026-06-01",
                "check_out_date": "2026-06-07",
                "nights": 6,
                "budget_max": 150.0,
                "guest_count": 2,
                "rooms": 1,
            },
            "options": [],
            "metadata": {
                "total_results": 0,
                "returned_results": 0,
            },
            "errors": [],
            "config": {},
        }
        parsed = HotelSearchArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.task_ref, "hotel-search-1")
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.attempt, 1)

    def test_coordinates_model(self) -> None:
        coords = HotelSearchCoordinatesModel(latitude=41.3851, longitude=2.1734)
        self.assertEqual(coords.latitude, 41.3851)
        self.assertEqual(coords.longitude, 2.1734)

    def test_search_parameters_model(self) -> None:
        params = HotelSearchParametersModel(
            location="Barcelona, Spain",
            check_in_date="2026-06-01",
            check_out_date="2026-06-07",
            nights=6,
            budget_max=150.0,
            guest_count=2,
        )
        self.assertEqual(params.location, "Barcelona, Spain")
        self.assertEqual(params.nights, 6)

    def test_hotel_option_model(self) -> None:
        option = HotelOptionModel(
            search_result_id="offer-123",
            accommodation_id="hotel-456",
            name="Test Hotel",
            nightly_rate=120.0,
            total_cost=720.0,
            currency="EUR",
            rating=4.5,
            latitude=41.3851,
            longitude=2.1734,
            over_budget=False,
        )
        self.assertEqual(option.name, "Test Hotel")
        self.assertFalse(option.over_budget)

    def test_metadata_model(self) -> None:
        metadata = HotelSearchMetadataModel(total_results=50, returned_results=10)
        self.assertEqual(metadata.total_results, 50)
        self.assertEqual(metadata.returned_results, 10)
        self.assertEqual(metadata.search_radius_km, 5.0)

    def test_error_model(self) -> None:
        error = HotelSearchErrorModel(code="timeout_error", message="Request timed out")
        self.assertEqual(error.code, "timeout_error")
        self.assertEqual(error.message, "Request timed out")


if __name__ == "__main__":
    unittest.main()