from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from travelplanner.agents.attraction_search_agent import (
    AttractionSearchConfig,
    _extract_attraction_params,
    _generate_activity,
    run_attraction_search,
)
from travelplanner.schema.attraction_search_artifact import (
    AttractionParamsModel,
    GeneratedActivityModel,
)
import travelplanner.agents.attraction_search_agent as agent_module


def _make_activity(has_specific_location: bool = True) -> GeneratedActivityModel:
    return GeneratedActivityModel(
        day=1,
        time_slot="morning",
        title="Sprint at a local hacker space",
        description="A rich, locally embedded experience.",
        local_touchpoint="Local makers who work here weekly.",
        search_keywords=["maker space", "community workshop"],
        estimated_duration_hours=3.0,
        has_specific_location=has_specific_location,
    )


def _make_params() -> AttractionParamsModel:
    return AttractionParamsModel(
        budget=80.0,
        destination="Barcelona",
        traveller_profile="digital nomad",
        day=1,
        previous_activities="",
        orchestrator_hint=None,
    )


_DIGITAL_NOMAD_ARCHETYPE = next(
    a for a in agent_module._EXPERIENCE_POOL if a["archetype"] == "digital_nomad"
)


class TestExperienceGeneration(unittest.TestCase):
    def setUp(self) -> None:
        agent_module._EMBEDDING_CACHE = None

    def tearDown(self) -> None:
        agent_module._EMBEDDING_CACHE = None

    def test_llm_generates_one_activity(self) -> None:
        activity = _make_activity()

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activity, "prompt", "raw"),
        ):
            result = _generate_activity(
                params=_make_params(),
                archetype=_DIGITAL_NOMAD_ARCHETYPE,
                model_name="test-model",
                temperature=0.0,
            )

        self.assertEqual(result.day, 1)
        self.assertEqual(result.title, "Sprint at a local hacker space")
        self.assertTrue(result.has_specific_location)

    def test_parse_retry_on_validation_error(self) -> None:
        activity = _make_activity()
        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError.from_exception_data(
                    title="GeneratedActivityModel",
                    input_type="python",
                    line_errors=[],
                )
            return (activity, "prompt", "raw")

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            side_effect=invoke_side_effect,
        ):
            result = _generate_activity(
                params=_make_params(),
                archetype=_DIGITAL_NOMAD_ARCHETYPE,
                model_name="test-model",
                temperature=0.0,
            )

        self.assertEqual(call_count, 2)
        self.assertEqual(result.title, "Sprint at a local hacker space")

    def test_has_specific_location_false_skips_serpapi(self) -> None:
        activity = _make_activity(has_specific_location=False)

        import numpy as np
        agent_module._EMBEDDING_CACHE = {
            "embeddings": [np.array([1.0, 0.0, 0.0, 0.0])] * 4,
            "names": ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"],
        }

        mock_query_emb = MagicMock()
        mock_query_emb.data = [MagicMock(embedding=[0.0, 0.0, 1.0, 0.0])]

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activity, "prompt", "raw"),
        ), patch(
            "travelplanner.agents.attraction_search_agent.openai.OpenAI"
        ) as mock_openai, patch(
            "travelplanner.agents.attraction_search_agent.requests.get"
        ) as mock_get:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_query_emb

            config = AttractionSearchConfig(openai_api_key="test-key", serpapi_api_key="test-key")
            result = run_attraction_search(
                params=_make_params(),
                model_name="test-model",
                temperature=0.0,
                config=config,
            )

        mock_get.assert_not_called()
        self.assertIsNotNone(result.item)
        self.assertFalse(result.item.place_found)

    def test_missing_openai_key_returns_failed_artifact(self) -> None:
        config = AttractionSearchConfig(openai_api_key="", serpapi_api_key="test-key")
        result = run_attraction_search(
            params=_make_params(),
            model_name="test-model",
            temperature=0.0,
            config=config,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "missing_api_key")

    def test_missing_serpapi_key_sets_place_not_found(self) -> None:
        activity = _make_activity(has_specific_location=True)

        import numpy as np
        agent_module._EMBEDDING_CACHE = {
            "embeddings": [np.array([1.0, 0.0, 0.0, 0.0])] * 4,
            "names": ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"],
        }
        mock_query_emb = MagicMock()
        mock_query_emb.data = [MagicMock(embedding=[0.0, 0.0, 1.0, 0.0])]

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(activity, "prompt", "raw"),
        ), patch(
            "travelplanner.agents.attraction_search_agent.openai.OpenAI"
        ) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_query_emb

            config = AttractionSearchConfig(openai_api_key="test-key", serpapi_api_key="")
            result = run_attraction_search(
                params=_make_params(),
                model_name="test-model",
                temperature=0.0,
                config=config,
            )

        self.assertIsNotNone(result.item)
        self.assertFalse(result.item.place_found)
        self.assertTrue(any(e.code == "missing_api_key" for e in result.errors))

    def test_param_extraction(self) -> None:
        expected_params = AttractionParamsModel(
            budget=80.0,
            destination="Barcelona",
            traveller_profile="solo digital nomad",
            day=2,
            previous_activities="Day 1 - ceramics workshop.",
            orchestrator_hint="Focus on tech community",
        )

        with patch(
            "travelplanner.agents.attraction_search_agent.invoke_structured_model",
            return_value=(expected_params, "prompt", "raw"),
        ):
            params = _extract_attraction_params(
                "Budget: 80.0, Destination: Barcelona, ...",
                model_name="test-model",
                temperature=0.0,
            )

        self.assertEqual(params.budget, 80.0)
        self.assertEqual(params.destination, "Barcelona")
        self.assertEqual(params.day, 2)
        self.assertEqual(params.orchestrator_hint, "Focus on tech community")


if __name__ == "__main__":
    unittest.main()
