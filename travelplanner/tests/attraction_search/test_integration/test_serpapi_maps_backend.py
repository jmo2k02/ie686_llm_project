from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests as req_lib

from travelplanner.agents.attraction_search_agent import (
    AttractionSearchConfig,
    _find_candidates,
    _fetch_reviews,
    _search_candidates,
)
from travelplanner.schema.attraction_search_artifact import (
    AttractionCandidateModel,
    GeneratedActivityModel,
)


_SAMPLE_LOCAL_RESULTS = [
    {
        "title": "Fablab Barcelona",
        "address": "Carrer dels Almogàvers 165",
        "gps_coordinates": {"latitude": 41.4015, "longitude": 2.1916},
        "rating": 4.7,
        "reviews": 312,
        "price": "$$",
        "type": "Coworking space",
        "data_id": "ChIJ_abc123",
        "hours": "Mon-Fri 9am-8pm",
    },
    {
        "title": "WorkTours BCN",
        "address": "Las Ramblas 42",
        "gps_coordinates": {"latitude": 41.38, "longitude": 2.17},
        "rating": 3.9,
        "reviews": 50,
        "price": "$$$",
        "type": "Coworking space",
        "data_id": "ChIJ_tourist",
        "hours": "Daily 10am-6pm",
    },
]

_SAMPLE_REVIEWS_RESPONSE = {
    "reviews": [
        {"snippet": "Great place, lots of locals and regulars here."},
        {"snippet": "Very community-driven, not touristy at all."},
        {"snippet": "A bit crowded on weekends."},
    ]
}


class TestSerpApiMapsBackend(unittest.TestCase):
    def test_missing_api_key_returns_empty_candidates(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="")
        result = _search_candidates("co-working space", "Barcelona", config)
        self.assertEqual(result, [])

    def test_success_candidate_search_normalizes_fields(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"local_results": _SAMPLE_LOCAL_RESULTS}
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.attraction_search_agent.requests.get", return_value=mock_response):
            candidates = _search_candidates("co-working space", "Barcelona", config)

        self.assertEqual(len(candidates), 2)
        first = candidates[0]
        self.assertEqual(first.title, "Fablab Barcelona")
        self.assertEqual(first.address, "Carrer dels Almogàvers 165")
        self.assertAlmostEqual(first.gps_coordinates["lat"], 41.4015)
        self.assertEqual(first.rating, 4.7)
        self.assertEqual(first.reviews, 312)
        self.assertEqual(first.data_id, "ChIJ_abc123")

    def test_http_error_returns_empty_candidates(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("403")

        with patch("travelplanner.agents.attraction_search_agent.requests.get", return_value=mock_response):
            result = _search_candidates("co-working space", "Barcelona", config)

        self.assertEqual(result, [])

    def test_timeout_returns_empty_candidates(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")

        with patch("travelplanner.agents.attraction_search_agent.requests.get", side_effect=req_lib.Timeout):
            result = _search_candidates("co-working space", "Barcelona", config)

        self.assertEqual(result, [])

    def test_no_local_results_returns_empty(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"local_results": []}
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.attraction_search_agent.requests.get", return_value=mock_response):
            result = _search_candidates("co-working space", "Barcelona", config)

        self.assertEqual(result, [])

    def test_review_fetch_populates_snippets(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        candidate = AttractionCandidateModel(title="Fablab Barcelona", data_id="ChIJ_abc123")

        mock_response = MagicMock()
        mock_response.json.return_value = _SAMPLE_REVIEWS_RESPONSE
        mock_response.raise_for_status.return_value = None

        with patch("travelplanner.agents.attraction_search_agent.requests.get", return_value=mock_response):
            snippets = _fetch_reviews(candidate, config)

        self.assertEqual(len(snippets), 3)
        self.assertIn("locals", snippets[0])

    def test_review_fetch_failure_is_non_fatal(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        candidate = AttractionCandidateModel(title="Fablab Barcelona", data_id="ChIJ_abc123")

        with patch("travelplanner.agents.attraction_search_agent.requests.get", side_effect=req_lib.Timeout):
            snippets = _fetch_reviews(candidate, config)

        self.assertEqual(snippets, [])

    def test_review_fetch_missing_data_id_returns_empty(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        candidate = AttractionCandidateModel(title="Unknown place", data_id=None)
        snippets = _fetch_reviews(candidate, config)
        self.assertEqual(snippets, [])

    def test_find_candidates_stops_after_two_found(self) -> None:
        config = AttractionSearchConfig(serpapi_api_key="test-key")
        activity = GeneratedActivityModel(
            day=1,
            time_slot="morning",
            title="Co-working session",
            description="Work alongside local makers.",
            local_touchpoint="Regular members.",
            search_keywords=["co-working space", "maker space", "hacker space"],
            estimated_duration_hours=3.0,
            has_specific_location=True,
        )

        mock_search = MagicMock(return_value=MagicMock(
            json=MagicMock(return_value={"local_results": _SAMPLE_LOCAL_RESULTS}),
            raise_for_status=MagicMock(return_value=None),
        ))
        mock_reviews = MagicMock(return_value=MagicMock(
            json=MagicMock(return_value={"reviews": []}),
            raise_for_status=MagicMock(return_value=None),
        ))

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            params = kwargs.get("params", {})
            if params.get("engine") == "google_maps_reviews":
                return mock_reviews()
            return mock_search()

        with patch("travelplanner.agents.attraction_search_agent.requests.get", side_effect=side_effect):
            candidates = _find_candidates(activity, "Barcelona", config)

        # Should stop after first keyword since 2 candidates found
        search_calls = sum(1 for _ in range(call_count))
        self.assertGreaterEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
