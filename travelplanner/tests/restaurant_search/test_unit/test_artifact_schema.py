from __future__ import annotations

import unittest

from travelplanner.schema.restaurant_search_artifact import (
    RestaurantArtifactContentModel,
    RestaurantCandidateModel,
    RestaurantItemModel,
    RestaurantLocationModel,
    RestaurantParamsModel,
    RestaurantSearchErrorModel,
)


class TestRestaurantArtifactSchema(unittest.TestCase):
    def test_params_model_defaults(self) -> None:
        payload = {
            "city": "Barcelona",
            "cuisine": "Italian",
            "budget": "medium",
            "meal_type": "dinner",
            "dietary_restrictions": ["vegetarian"],
            "min_rating": 4.0,
            "num_people": 2,
            "preferred_time": "19:30",
            "additional_notes": "outdoor seating",
        }
        parsed = RestaurantParamsModel.model_validate(payload)
        self.assertEqual(parsed.city, "Barcelona")
        self.assertEqual(parsed.cuisine, "Italian")
        self.assertEqual(parsed.budget, "medium")
        self.assertEqual(parsed.meal_type, "dinner")
        self.assertEqual(parsed.dietary_restrictions, ["vegetarian"])
        self.assertEqual(parsed.min_rating, 4.0)
        self.assertEqual(parsed.num_people, 2)
        self.assertEqual(parsed.preferred_time, "19:30")
        self.assertEqual(parsed.additional_notes, "outdoor seating")

    def test_params_model_defaults_unset(self) -> None:
        parsed = RestaurantParamsModel(city="Paris")
        self.assertIsNone(parsed.cuisine)
        self.assertIsNone(parsed.budget)
        self.assertIsNone(parsed.meal_type)
        self.assertEqual(parsed.dietary_restrictions, [])
        self.assertIsNone(parsed.min_rating)
        self.assertEqual(parsed.num_people, 1)
        self.assertIsNone(parsed.preferred_time)
        self.assertIsNone(parsed.additional_notes)

    def test_candidate_model(self) -> None:
        payload = {
            "place_id": "ChIJ_abc123",
            "name": "La Pepita",
            "address": "Carrer de Còrsega 343, Barcelona",
            "types": ["restaurant", "food", "establishment"],
            "rating": 4.5,
            "price_level": "$$",
            "phone": "+34 123 456 789",
            "website": "https://lapepita.com",
            "opening_hours": "Mon-Sun 13:00-23:00",
            "location": {"lat": 41.3851, "lng": 2.1734},
            "photos": ["photo_ref_1"],
            "raw": {"extra": "data"},
        }
        parsed = RestaurantCandidateModel.model_validate(payload)
        self.assertEqual(parsed.place_id, "ChIJ_abc123")
        self.assertEqual(parsed.name, "La Pepita")
        self.assertEqual(parsed.rating, 4.5)
        self.assertEqual(parsed.location.lat, 41.3851) if parsed.location else None
        self.assertEqual(parsed.raw.get("extra"), "data")

    def test_item_model_with_candidate(self) -> None:
        payload = {
            "name": "La Pepita",
            "address": "Carrer de Còrsega 343, Barcelona",
            "place_id": "ChIJ_abc123",
            "cuisine": "Italian",
            "meal_type": "dinner",
            "rating": 4.5,
            "price_level": "$$",
            "price_range": "$$",
            "phone": "+34 123 456 789",
            "website": "https://lapepita.com",
            "opening_hours": "Mon-Sun 13:00-23:00",
            "location": {"lat": 41.3851, "lng": 2.1734},
            "dietary_suitability": ["vegetarian"],
            "selection_reason": "Highly rated and fits the budget.",
            "provenance": "google_places_api_new",
        }
        parsed = RestaurantItemModel.model_validate(payload)
        self.assertEqual(parsed.name, "La Pepita")
        self.assertEqual(parsed.price_range, "$$")
        self.assertEqual(parsed.provenance, "google_places_api_new")

    def test_item_model_fallback(self) -> None:
        payload = {
            "name": "Restaurant in Barcelona",
            "cuisine": "Italian",
            "meal_type": "dinner",
            "price_range": "$$",
            "dietary_suitability": ["vegetarian"],
            "provenance": "fallback_llm_suggestion",
        }
        parsed = RestaurantItemModel.model_validate(payload)
        self.assertIsNone(parsed.place_id)
        self.assertIsNone(parsed.rating)
        self.assertEqual(parsed.provenance, "fallback_llm_suggestion")

    def test_artifact_content_success(self) -> None:
        payload = {
            "task_ref": "rest-1",
            "status": "success",
            "provider": "google_places_api_new",
            "query": '{"city":"Barcelona","cuisine":"Italian"}',
            "city": "Barcelona",
            "cuisine": "Italian",
            "budget": "medium",
            "meal_type": "dinner",
            "items": [
                {
                    "name": "La Pepita",
                    "address": "Carrer de Còrsega 343, Barcelona",
                    "place_id": "ChIJ_abc123",
                    "rating": 4.5,
                    "price_range": "$$",
                    "provenance": "google_places_api_new",
                }
            ],
            "errors": [],
            "config": {"api_key_set": True, "max_results": 5},
        }
        parsed = RestaurantArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.city, "Barcelona")
        self.assertEqual(len(parsed.items), 1)
        self.assertEqual(parsed.items[0].name, "La Pepita")

    def test_artifact_content_failed(self) -> None:
        payload = {
            "task_ref": "rest-fail",
            "status": "failed",
            "provider": "google_places_api_new",
            "query": '{"city":"Barcelona"}',
            "city": "Barcelona",
            "items": [],
            "errors": [{"code": "missing_api_key", "message": "GOOGLE_PLACES_API_KEY is not set"}],
        }
        parsed = RestaurantArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "failed")
        self.assertEqual(len(parsed.errors), 1)
        self.assertEqual(parsed.errors[0].code, "missing_api_key")

    def test_error_model_invalid_code_fallback(self) -> None:
        payload = {"code": "not_a_real_code", "message": "Something weird happened"}
        # Pydantic Literal will reject invalid code at validation time
        with self.assertRaises(Exception):
            RestaurantSearchErrorModel.model_validate(payload)

    def test_location_model(self) -> None:
        loc = RestaurantLocationModel(lat=41.3851, lng=2.1734)
        self.assertAlmostEqual(loc.lat, 41.3851)
        self.assertAlmostEqual(loc.lng, 2.1734)


if __name__ == "__main__":
    unittest.main()
