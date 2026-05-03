from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch
from pydantic import ValidationError

from travelplanner.agents.attraction_search_agent import (
    AttractionSearchConfig,
    _generate_activities,
    run_attraction_search,
)
from travelplanner.schema.attraction_search_artifact import (
    GeneratedActivitiesResponse,
    GeneratedActivityModel,
)
import travelplanner.agents.attraction_search_agent as agent_module


def _make_activities(n: int, has_specific_location: bool = True) -> GeneratedActivitiesResponse:
    return GeneratedActivitiesResponse(
        activities=[
            GeneratedActivityModel(
                day=i + 1,
                time_slot="morning",
                title=f"Activity {i + 1}",
                description="A rich, locally embedded experience.",
                local_touchpoint="Local makers who work here weekly.",
                search_keywords=["maker space", "community workshop"],
                estimated_duration_hours=3.0,
                has_specific_location=has_specific_location,
            )
            for i in range(n)
        ]
    )


_DIGITAL_NOMAD_ARCHETYPE = next(
    a for a in agent_module._EXPERIENCE_POOL if a["archetype"] == "digital_nomad"
)


class TestExperienceGeneration(unittest.TestCase):
    def setUp(self) -> None:
        agent_module._EMBEDDING_CACHE = None

    def tearDown(self) -> None:
        agent_module._EMBEDDING_CACHE = None

    def test_llm_generates_correct_activity_count(self) -> None:
        activities_resp = _make_activities(3)

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activities_resp, "prompt", "raw"),
        ):
            activities = _generate_activities(
                destination="Barcelona",
                days=3,
                budget="medium",
                traveller_profile="digital nomad",
                archetype=_DIGITAL_NOMAD_ARCHETYPE,
                model_name="test-model",
                temperature=0.0,
            )

        self.assertEqual(len(activities), 3)
        self.assertEqual(activities[0].day, 1)
        self.assertEqual(activities[2].day, 3)

    def test_parse_retry_on_validation_error(self) -> None:
        activities_resp = _make_activities(2)
        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError.from_exception_data(
                    title="GeneratedActivitiesResponse",
                    input_type="python",
                    line_errors=[],
                )
            return (activities_resp, "prompt", "raw")

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            side_effect=invoke_side_effect,
        ):
            activities = _generate_activities(
                destination="Barcelona",
                days=2,
                budget="medium",
                traveller_profile="digital nomad",
                archetype=_DIGITAL_NOMAD_ARCHETYPE,
                model_name="test-model",
                temperature=0.0,
            )

        self.assertEqual(call_count, 2)
        self.assertEqual(len(activities), 2)

    def test_has_specific_location_false_skips_serpapi(self) -> None:
        activities_resp = _make_activities(1, has_specific_location=False)

        # Seed embedding cache so _select_archetype doesn't call OpenAI
        import numpy as np
        agent_module._EMBEDDING_CACHE = {
            "embeddings": [np.array([1.0, 0.0, 0.0, 0.0])] * 4,
            "names": ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"],
        }

        mock_query_emb = MagicMock()
        mock_query_emb.data = [MagicMock(embedding=[0.0, 0.0, 1.0, 0.0])]

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activities_resp, "prompt", "raw"),
        ), patch(
            "travelplanner.agents.attraction_search_agent.openai.OpenAI"
        ) as mock_openai, patch(
            "travelplanner.agents.attraction_search_agent.requests.get"
        ) as mock_get:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_query_emb

            config = AttractionSearchConfig(
                openai_api_key="test-key", serpapi_api_key="test-key"
            )
            result = run_attraction_search(
                destination="Barcelona",
                days=1,
                budget="medium",
                traveller_profile="running club enthusiast",
                model_name="test-model",
                temperature=0.0,
                config=config,
            )

        mock_get.assert_not_called()
        self.assertEqual(len(result.items), 1)
        self.assertFalse(result.items[0].place_found)

    def test_missing_openai_key_returns_failed_artifact(self) -> None:
        config = AttractionSearchConfig(openai_api_key="", serpapi_api_key="test-key")
        result = run_attraction_search(
            destination="Barcelona",
            days=3,
            budget="medium",
            traveller_profile="any profile",
            model_name="test-model",
            temperature=0.0,
            config=config,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "missing_api_key")

    def test_missing_serpapi_key_sets_place_not_found(self) -> None:
        activities_resp = _make_activities(1, has_specific_location=True)

        import numpy as np
        agent_module._EMBEDDING_CACHE = {
            "embeddings": [np.array([1.0, 0.0, 0.0, 0.0])] * 4,
            "names": ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"],
        }
        mock_query_emb = MagicMock()
        mock_query_emb.data = [MagicMock(embedding=[0.0, 0.0, 1.0, 0.0])]

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activities_resp, "prompt", "raw"),
        ), patch(
            "travelplanner.agents.attraction_search_agent.openai.OpenAI"
        ) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_query_emb

            config = AttractionSearchConfig(openai_api_key="test-key", serpapi_api_key="")
            result = run_attraction_search(
                destination="Barcelona",
                days=1,
                budget="medium",
                traveller_profile="nomad",
                model_name="test-model",
                temperature=0.0,
                config=config,
            )

        self.assertEqual(len(result.items), 1)
        self.assertFalse(result.items[0].place_found)
        self.assertTrue(any(e.code == "missing_api_key" for e in result.errors))


if __name__ == "__main__":
    unittest.main()
