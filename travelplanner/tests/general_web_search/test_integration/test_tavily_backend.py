from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from travelplanner.agents.general_web_search_agent import _search_tavily


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class TestTavilyBackend(unittest.TestCase):
    def test_missing_api_key_returns_error_payload(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = _search_tavily(
                "rome weather",
                max_results=3,
                timeout=10,
                search_depth="basic",
                include_answer=True,
            )
        self.assertFalse(result["ok"])
        self.assertIn("TAVILY_API_KEY", result["error"])

    def test_success_payload_is_normalized(self) -> None:
        fake_tavily_response = {
            "results": [
                {
                    "title": "Weather",
                    "url": "https://example.com",
                    "content": "It is usually warm in Rome.",
                    "score": 0.95,
                }
            ],
            "answer": "It is usually warm.",
        }
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True):
            with patch(
                "travelplanner.agents.general_web_search_agent.TavilyClient"
            ) as mock_client_class:
                mock_client = mock_client_class.return_value
                mock_client.search.return_value = fake_tavily_response
                result = _search_tavily(
                    "rome weather",
                    max_results=3,
                    timeout=10,
                    search_depth="basic",
                    include_answer=True,
                )
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "It is usually warm.")
        self.assertEqual(len(result["results"]), 1)


if __name__ == "__main__":
    unittest.main()
