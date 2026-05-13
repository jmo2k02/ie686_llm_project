from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests as req_lib

from travelplanner.agents.restaurant_search_agent import (
    RestaurantSearchConfig,
    _compute_status,
    _normalize_candidate,
    _normalize_candidates,
    _normalize_error,
    _search_restaurants,
    _select_candidate,
    load_config_from_env,
)
from travelplanner.schema.restaurant_search_artifact import (
    RestaurantCandidateModel,
    RestaurantParamsModel,
    RestaurantSearchErrorModel,
)


_SAMPLE_GOOGLE_PLACES_RESPONSE = {
    "places": [
        {
            "id": "ChIJ_abc123",
            "displayName": {"text": "La Pepita"},
            "formattedAddress": "Carrer de Còrsega 343, Barcelona",
            "types": ["restaurant", "food"],
            "rating": 4.5,
            "priceLevel": "MODERATE",
            "nationalPhoneNumber": "+34 123 456 789",
            "websiteUri": "https://lapepita.com",
            "regularOpeningHours": {"weekdayDescriptions": ["Monday: 1:00 PM – 11:00 PM", "Tuesday: 1:00 PM – 11:00 PM"]},
            "location": {"latitude": 41.3851, "longitude": 2.1734},
            "photos": [{"name": "photo_ref_1"}],
        },
        {
            "id": "ChIJ_def456",
            "displayName": {"text": "El Xampanyet"},
            "formattedAddress": "Carrer de Montcada 22, Barcelona",
            "types": ["bar", "restaurant"],
            "rating": 4.2,
            "priceLevel": "INEXPENSIVE",
            "location": {"latitude": 41.3845, "longitude": 2.1823},
        },
    ]
}


class TestLoadConfig(unittest.TestCase):
    def test_load_from_env(self) -> None:
        # We can't control env in these tests reliably, but we can verify the function
        # returns a RestaurantSearchConfig dataclass with expected types.
        cfg = load_config_from_env()
        self.assertIsInstance(cfg, RestaurantSearchConfig)
        self.assertIsInstance(cfg.timeout_seconds, int)
        self.assertIsInstance(cfg.max_results, int)
        self.assertGreaterEqual(cfg.timeout_seconds, 5)
        self.assertGreaterEqual(cfg.max_results, 1)


class TestSearchRestaurants(unittest.TestCase):
    def test_missing_api_key_returns_error(self) -> None:
        config = RestaurantSearchConfig(api_key="")
        params = RestaurantParamsModel(city="Barcelona")
        result = _search_restaurants(params, config)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_api_key")

    def test_successful_search_normalizes_fields(self) -> None:
        config = RestaurantSearchConfig(api_key="test-key")
        params = RestaurantParamsModel(city="Barcelona", cuisine="Italian")

        mock_response = MagicMock()
        mock_response.json.return_value = _SAMPLE_GOOGLE_PLACES_RESPONSE
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.restaurant_search_agent.requests.post", return_value=mock_response):
            result = _search_restaurants(params, config)

        self.assertTrue(result["ok"])
        self.assertIn("raw", result)
        self.assertEqual(len(result["raw"]["places"]), 2)

    def test_http_error_returns_error(self) -> None:
        config = RestaurantSearchConfig(api_key="test-key")
        params = RestaurantParamsModel(city="Barcelona")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("403")

        with patch("travelplanner.agents.restaurant_search_agent.requests.post", return_value=mock_response):
            result = _search_restaurants(params, config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "http_error")

    def test_timeout_returns_error(self) -> None:
        config = RestaurantSearchConfig(api_key="test-key")
        params = RestaurantParamsModel(city="Barcelona")

        with patch("travelplanner.agents.restaurant_search_agent.requests.post", side_effect=req_lib.Timeout):
            result = _search_restaurants(params, config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout_error")

    def test_request_params_include_query_and_budget(self) -> None:
        config = RestaurantSearchConfig(api_key="test-key")
        params = RestaurantParamsModel(city="Barcelona", cuisine="Italian", budget="medium")

        mock_response = MagicMock()
        mock_response.json.return_value = {"places": []}
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.restaurant_search_agent.requests.post", return_value=mock_response) as mock_post:
            _search_restaurants(params, config)

        call_args = mock_post.call_args
        self.assertEqual(call_args.kwargs["headers"]["X-Goog-Api-Key"], "test-key")
        request_body = call_args.kwargs["json"]
        self.assertEqual(request_body["textQuery"], "restaurant Italian in Barcelona")
        self.assertEqual(request_body["pageSize"], config.max_results)
        self.assertEqual(request_body["priceLevels"], ["PRICE_LEVEL_MODERATE"])


class TestNormalizeCandidates(unittest.TestCase):
    def test_normalize_candidate(self) -> None:
        raw = _SAMPLE_GOOGLE_PLACES_RESPONSE["places"][0]
        candidate = _normalize_candidate(raw)
        self.assertEqual(candidate.place_id, "ChIJ_abc123")
        self.assertEqual(candidate.name, "La Pepita")
        self.assertEqual(candidate.rating, 4.5)
        self.assertEqual(candidate.price_level, "MODERATE")
        self.assertEqual(candidate.phone, "+34 123 456 789")
        self.assertEqual(candidate.website, "https://lapepita.com")
        self.assertEqual(candidate.opening_hours, "Monday: 1:00 PM – 11:00 PM; Tuesday: 1:00 PM – 11:00 PM")
        self.assertIsNotNone(candidate.location)
        self.assertAlmostEqual(candidate.location.lat, 41.3851)
        self.assertEqual(candidate.photos, ["photo_ref_1"])
        self.assertIn("id", candidate.raw)

    def test_normalize_candidates_empty(self) -> None:
        result = _normalize_candidates({"places": []})
        self.assertEqual(result, [])

    def test_normalize_candidates_malformed_location(self) -> None:
        raw = {
            "id": "ChIJ_bad",
            "displayName": {"text": "Bad Loc"},
            "location": {"latitude": "not_a_number", "longitude": 2.0},
        }
        candidate = _normalize_candidate(raw)
        # Should gracefully handle bad lat
        self.assertIsNone(candidate.location)

    def test_normalize_candidates_non_list_data(self) -> None:
        result = _normalize_candidates({"places": "unexpected_string"})
        self.assertEqual(result, [])

    def test_normalize_candidate_plain_string_display_name(self) -> None:
        raw = {
            "id": "ChIJ_str",
            "displayName": "Plain String Restaurant",
            "formattedAddress": "Some Street 1",
        }
        candidate = _normalize_candidate(raw)
        self.assertEqual(candidate.name, "Plain String Restaurant")

    def test_opening_hours_none(self) -> None:
        raw = {
            "id": "ChIJ_no_hours",
            "displayName": {"text": "No Hours Cafe"},
        }
        candidate = _normalize_candidate(raw)
        self.assertIsNone(candidate.opening_hours)

    def test_photos_empty(self) -> None:
        raw = {
            "id": "ChIJ_no_photos",
            "displayName": {"text": "No Photos"},
        }
        candidate = _normalize_candidate(raw)
        self.assertEqual(candidate.photos, [])


class TestNormalizeError(unittest.TestCase):
    def test_known_code(self) -> None:
        err = _normalize_error("timeout_error", "Request timed out")
        self.assertEqual(err.code, "timeout_error")
        self.assertEqual(err.message, "Request timed out")

    def test_unknown_code_fallback(self) -> None:
        err = _normalize_error("weird_code", "Something odd")
        self.assertEqual(err.code, "unknown_error")


class TestComputeStatus(unittest.TestCase):
    def test_success(self) -> None:
        status = _compute_status(True, 3, [])
        self.assertEqual(status, "success")

    def test_failed_no_items(self) -> None:
        status = _compute_status(True, 0, [RestaurantSearchErrorModel(code="http_error", message="fail")])
        self.assertEqual(status, "failed")

    def test_partial_with_errors(self) -> None:
        status = _compute_status(True, 2, [RestaurantSearchErrorModel(code="timeout_error", message="slow")])
        self.assertEqual(status, "partial")

    def test_failed_not_ok(self) -> None:
        status = _compute_status(False, 0, [])
        self.assertEqual(status, "failed")


class TestSelectCandidate(unittest.TestCase):
    def test_fallback_when_llm_fails(self) -> None:
        candidates = [
            RestaurantCandidateModel(place_id="1", name="A", rating=3.0),
            RestaurantCandidateModel(place_id="2", name="B", rating=4.5),
        ]
        params = RestaurantParamsModel(city="Barcelona")

        with patch("travelplanner.agents.restaurant_search_agent.invoke_structured_model", side_effect=RuntimeError("LLM down")):
            selected, reason = _select_candidate(candidates, params, "openai:gpt-5-mini", 0.0)

        self.assertEqual(selected.place_id, "2")
        self.assertEqual(reason, "Fallback to highest-rated candidate.")

    def test_selects_within_bounds(self) -> None:
        candidates = [
            RestaurantCandidateModel(place_id="1", name="A", rating=4.0),
            RestaurantCandidateModel(place_id="2", name="B", rating=4.5),
        ]
        params = RestaurantParamsModel(city="Barcelona")

        class FakeSelection:
            selected_index = 5  # Out of bounds
            selection_reason = "I like this one"

        with patch(
            "travelplanner.agents.restaurant_search_agent.invoke_structured_model",
            return_value=(FakeSelection(), "", ""),
        ):
            selected, reason = _select_candidate(candidates, params, "openai:gpt-5-mini", 0.0)

        self.assertEqual(selected.place_id, "2")  # clamped to last index
        self.assertEqual(reason, "I like this one")





if __name__ == "__main__":
    unittest.main()
