from __future__ import annotations

import unittest

from travelplanner.schema.attraction_search_artifact import (
    AttractionArtifactContentModel,
    AttractionItemModel,
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


class TestAttractionArtifactSchema(unittest.TestCase):
    def test_required_fields_and_status(self) -> None:
        payload = {
            "task_ref": "attraction-1",
            "status": "success",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Barcelona",
            "days": 3,
            "budget": "medium",
            "selected_archetype": "digital_nomad",
            "items": [_make_item(place_found=True)],
            "errors": [],
            "config": {"openai_api_key_set": True},
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.destination, "Barcelona")
        self.assertEqual(parsed.selected_archetype, "digital_nomad")
        self.assertEqual(len(parsed.items), 1)
        self.assertEqual(parsed.items[0].place_rating, 4.7)
        self.assertEqual(parsed.items[0].provenance, "LLM activity | SERPAPI google_maps")

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
            "days": 3,
            "budget": "medium",
            "selected_archetype": "",
            "items": [],
            "errors": [{"code": "missing_api_key", "message": "OPENAI_API_KEY is not set"}],
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "failed")
        self.assertEqual(len(parsed.errors), 1)
        self.assertEqual(parsed.errors[0].code, "missing_api_key")

    def test_partial_status_with_mixed_items(self) -> None:
        payload = {
            "task_ref": "attraction-partial",
            "status": "partial",
            "provider": "openai_embeddings+llm+serpapi_google_maps",
            "destination": "Berlin",
            "days": 2,
            "budget": "low",
            "selected_archetype": "active_traveler",
            "items": [_make_item(True), _make_item(False)],
            "errors": [{"code": "http_error", "message": "SERPAPI returned 429"}],
        }
        parsed = AttractionArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.status, "partial")
        self.assertTrue(parsed.items[0].place_found)
        self.assertFalse(parsed.items[1].place_found)


if __name__ == "__main__":
    unittest.main()
