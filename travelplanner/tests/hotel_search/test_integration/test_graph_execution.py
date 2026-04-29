"""Integration tests for full graph execution."""
from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")

from travelplanner.agents.hotel_search_agent import make_graph


@unittest.skipIf(
    not os.environ.get("LITEAPI_API_KEY"),
    "LITEAPI_API_KEY not set - skipping live API tests",
)
class TestGraphExecution(unittest.TestCase):
    """Test full hotel search agent graph execution."""

    def test_basic_graph_execution(self):
        """Test basic graph execution without amenity requirements."""
        graph = make_graph()

        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        state = {
            "query": "Find hotels in Barcelona",
            "search_parameters": {
                "location": "Barcelona, Spain",
                "dates": f"{check_in} to {check_out}",
                "budget_max": 150.0,
                "guest_count": 2,
            },
            "task_id": 1,
        }

        result = graph.invoke(state)

        self.assertIn("hotel_artifact", result)
        artifact = result["hotel_artifact"]
        self.assertIsNotNone(artifact)

        content = artifact.content
        self.assertEqual(content["status"], "success")
        self.assertGreater(len(content["options"]), 0)
        self.assertLessEqual(len(content["options"]), 10)

        # Check first hotel structure
        first_hotel = content["options"][0]
        self.assertIn("name", first_hotel)
        self.assertIn("nightly_rate", first_hotel)
        self.assertIn("rating", first_hotel)
        self.assertIn("amenities", first_hotel)
        self.assertIn("over_budget", first_hotel)
        self.assertIn("rank", first_hotel)

    def test_graph_with_amenity_requirements(self):
        """Test graph execution with amenity filtering."""
        graph = make_graph()

        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        state = {
            "query": "Find hotels with WiFi and pool",
            "search_parameters": {
                "location": "Barcelona, Spain",
                "dates": f"{check_in} to {check_out}",
                "budget_max": 200.0,
                "guest_count": 2,
                "required_amenities": ["wifi", "pool"],
                "preferred_amenities": ["gym"],
            },
            "task_id": 1,
        }

        result = graph.invoke(state)

        artifact = result["hotel_artifact"]
        content = artifact.content

        self.assertEqual(content["status"], "success")

        # All hotels should have required amenities
        for hotel in content["options"]:
            amenities_lower = [a.lower() for a in hotel["amenities"]]

            # Check wifi (fuzzy match)
            has_wifi = any(
                kw in " ".join(amenities_lower)
                for kw in ["wifi", "internet", "wireless"]
            )
            self.assertTrue(has_wifi, f"Hotel {hotel['name']} missing wifi")

            # Check pool (fuzzy match)
            has_pool = any("pool" in a for a in amenities_lower)
            self.assertTrue(has_pool, f"Hotel {hotel['name']} missing pool")

    def test_graph_with_min_rating(self):
        """Test graph execution with minimum rating filter."""
        graph = make_graph()

        today = datetime.now().date()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=37)).isoformat()

        state = {
            "query": "Find highly rated hotels",
            "search_parameters": {
                "location": "Barcelona, Spain",
                "dates": f"{check_in} to {check_out}",
                "budget_max": 300.0,
                "guest_count": 2,
                "min_rating": 8.5,
            },
            "task_id": 1,
        }

        result = graph.invoke(state)

        artifact = result["hotel_artifact"]
        content = artifact.content

        if content["status"] == "success":
            # All hotels should meet min rating
            for hotel in content["options"]:
                self.assertGreaterEqual(
                    hotel["rating"],
                    8.5,
                    f"Hotel {hotel['name']} has rating {hotel['rating']} < 8.5",
                )

    def test_graph_handles_invalid_dates(self):
        """Test that graph returns failed artifact for invalid dates."""
        graph = make_graph()

        state = {
            "query": "Find hotels",
            "search_parameters": {
                "location": "Barcelona, Spain",
                "dates": "invalid date format",
                "budget_max": 150.0,
                "guest_count": 2,
            },
            "task_id": 1,
        }

        # Graph should complete but return failed artifact
        result = graph.invoke(state)

        self.assertIn("hotel_artifact", result)
        artifact = result["hotel_artifact"]
        content = artifact.content

        self.assertEqual(content["status"], "failed")
        self.assertGreater(len(content.get("errors", [])), 0)
        self.assertEqual(content["errors"][0]["code"], "parse_error")


if __name__ == "__main__":
    unittest.main()
