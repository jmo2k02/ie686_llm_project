from __future__ import annotations

import unittest

from travelplanner.schema.attraction_search_artifact import (
    AttractionArtifactContentModel,
    AttractionCandidateModel,
    AttractionItemModel,
    AttractionParamsModel,
    AttractionSearchErrorModel,
)


def _make_item(place_found: bool = True) -> dict:
    base = {
        "day": 1,
        "time_slot": "morning",
        "title": "Sprint at a local hacker space",
        "description": "A working session alongside local makers.",
        "local_touchpoint": "Regular members who build here weekly.",
        "estimated_duration_hours": 3.0,
        "has_specific_location": True,
        "location_name": "Fablab Barcelona" if place_found else "Barcelona",
        "place_found": place_found,
        "estimated_price_range": "$$",
        "selected_archetype": "digital_nomad",
        "provenance": "LLM activity | SERPAPI google_maps" if place_found else "LLM activity | no place found",
    }
    if place_found:
        base.update({
            "location_address": "Carrer dels Almogàvers 165",
            "coordinates": {"lat": 41.4015, "lng": 2.1916},
            "place_id": "ChIJ_abc123",
            "place_rating": 4.7,
            "place_review_count": 312,
            "place_price_level": "$$",
            "place_type": "Coworking space",
            "place_hours": "Mon-Fri 9am-8pm",
            "selection_reason": "Locally embedded maker community, not tourist-facing.",
        })
    return base


class TestAttractionParamsSchema(unittest.TestCase):
    def test_params_model_validates(self) -> None:
        payload = {
            "budget": 80.0,
            "destination": "Barcelona",
            "traveller_profile": "solo digital nomad",
            "day": 2,
            "previous_activities": "Day 1 - ceramics workshop.",
            "orchestrator_hint": "Focus on tech community",
        }
        params = AttractionParamsModel.model_validate(payload)
        self.assertEqual(params.budget, 80.0)
        self.assertEqual(params.destination, "Barcelona")
        self.assertEqual(params.day, 2)
        self.assertEqual(params.orchestrator_hint, "Focus on tech community")

    def test_params_defaults(self) -> None:
        params = AttractionParamsModel(budget=50.0, destination="Berlin", traveller_profile="active traveler")
        self.assertEqual(params.day, 1)
        self.assertEqual(params.previous_activities, "")
        self.assertIsNone(params.orchestrator_hint)


class TestAttractionArtifactSchema(unittest.TestCase):
    def test_required_fields_and_status(self) -> None:
        payload = {
            "task_ref": "attraction-1",
            "status": "success",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Barcelona",
            "budget": 80.0,
            "selected_archetype": "digital_nomad",
            "item": _make_item(place_found=True),
            "top_candidates": [],
            "errors": [],
            "config": {"openai_api_key_set": True},
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.destination, "Barcelona")
        self.assertEqual(parsed.budget, 80.0)
        self.assertEqual(parsed.selected_archetype, "digital_nomad")
        self.assertIsNotNone(parsed.item)
        self.assertEqual(parsed.item.place_rating, 4.7)
        self.assertEqual(parsed.item.provenance, "LLM activity | SERPAPI google_maps")

    def test_null_location_item_validates(self) -> None:
        item_payload = _make_item(place_found=False)
        item = AttractionItemModel.model_validate(item_payload)
        self.assertFalse(item.place_found)
        self.assertIsNone(item.location_address)
        self.assertIsNone(item.coordinates)
        self.assertIsNone(item.place_rating)
        self.assertEqual(item.location_name, "Barcelona")
        self.assertEqual(item.provenance, "LLM activity | no place found")

    def test_failed_status_with_errors(self) -> None:
        payload = {
            "task_ref": "attraction-fail",
            "status": "failed",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Barcelona",
            "budget": 80.0,
            "selected_archetype": "",
            "errors": [{"code": "missing_api_key", "message": "OPENAI_API_KEY is not set"}],
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "failed")
        self.assertIsNone(parsed.item)
        self.assertEqual(len(parsed.errors), 1)
        self.assertEqual(parsed.errors[0].code, "missing_api_key")

    def test_top_candidates_stored_in_artifact(self) -> None:
        candidate = {
            "title": "Fablab Barcelona",
            "address": "Carrer dels Almogàvers 165",
            "gps_coordinates": {"lat": 41.4015, "lng": 2.1916},
            "rating": 4.7,
            "reviews": 312,
        }
        payload = {
            "task_ref": "attraction-1",
            "status": "success",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Barcelona",
            "budget": 80.0,
            "selected_archetype": "digital_nomad",
            "item": _make_item(place_found=True),
            "top_candidates": [candidate, candidate],
            "errors": [],
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(len(parsed.top_candidates), 2)
        self.assertEqual(parsed.top_candidates[0].title, "Fablab Barcelona")
        self.assertAlmostEqual(parsed.top_candidates[0].gps_coordinates["lat"], 41.4015)

    def test_partial_status_with_no_place_found(self) -> None:
        payload = {
            "task_ref": "attraction-partial",
            "status": "partial",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Berlin",
            "budget": 40.0,
            "selected_archetype": "active_traveler",
            "item": _make_item(False),
            "errors": [{"code": "http_error", "message": "SERPAPI returned 429"}],
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "partial")
        self.assertFalse(parsed.item.place_found)


if __name__ == "__main__":
    unittest.main()
