from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from travelplanner.agents.attraction_search_agent import (
    AttractionSearchConfig,
    _cosine_similarity,
    _select_archetype,
    _serialize_profile,
)
import travelplanner.agents.attraction_search_agent as agent_module


class TestSerializeProfile(unittest.TestCase):
    def test_digital_nomad_format(self) -> None:
        profile = {
            "travel_style": "slow travel",
            "party_type": "solo",
            "pace": "slow",
            "budget": "medium",
            "interests": ["remote work", "startup scene", "coffee culture"],
            "engagement_depth": "high",
        }
        result = _serialize_profile(profile)
        self.assertIn("slow travel", result)
        self.assertIn("solo", result)
        self.assertIn("slow pace", result)
        self.assertIn("medium budget", result)
        self.assertIn("remote work, startup scene, coffee culture", result)
        self.assertIn("engagement depth: high", result)

    def test_empty_interests(self) -> None:
        profile = {
            "travel_style": "active",
            "party_type": "group",
            "pace": "fast",
            "budget": "low",
            "interests": [],
            "engagement_depth": "medium",
        }
        result = _serialize_profile(profile)
        self.assertIn("interested in ,", result)  # empty join, comma follows


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(_cosine_similarity(a, b), -1.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        self.assertEqual(_cosine_similarity(a, b), 0.0)


class TestArchetypeSelection(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the module-level cache before each test
        agent_module._EMBEDDING_CACHE = None

    def tearDown(self) -> None:
        agent_module._EMBEDDING_CACHE = None

    def test_selects_correct_archetype_by_embedding(self) -> None:
        # 4 orthogonal unit vectors — one per archetype
        family_emb      = np.array([1.0, 0.0, 0.0, 0.0])
        active_emb      = np.array([0.0, 1.0, 0.0, 0.0])
        nomad_emb       = np.array([0.0, 0.0, 1.0, 0.0])
        culture_emb     = np.array([0.0, 0.0, 0.0, 1.0])
        archetype_embs  = [family_emb, active_emb, nomad_emb, culture_emb]
        archetype_names = ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"]

        # Pre-seed the cache with our mock embeddings
        agent_module._EMBEDDING_CACHE = {
            "embeddings": archetype_embs,
            "names": archetype_names,
        }

        config = AttractionSearchConfig(openai_api_key="test-key")

        # Query that should match digital_nomad (third vector)
        nomad_query_emb = np.array([0.0, 0.0, 1.0, 0.0])
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=nomad_query_emb.tolist())]

        with patch("travelplanner.agents.attraction_search_agent.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_resp
            name, archetype = _select_archetype("solo nomad looking for startup scene", config)

        self.assertEqual(name, "digital_nomad")
        self.assertEqual(archetype["archetype"], "digital_nomad")

    def test_selects_family_archetype(self) -> None:
        family_emb      = np.array([1.0, 0.0, 0.0, 0.0])
        active_emb      = np.array([0.0, 1.0, 0.0, 0.0])
        nomad_emb       = np.array([0.0, 0.0, 1.0, 0.0])
        culture_emb     = np.array([0.0, 0.0, 0.0, 1.0])

        agent_module._EMBEDDING_CACHE = {
            "embeddings": [family_emb, active_emb, nomad_emb, culture_emb],
            "names": ["family_with_kids", "active_traveler", "digital_nomad", "culture_seeking_couple"],
        }

        config = AttractionSearchConfig(openai_api_key="test-key")
        query_emb = np.array([1.0, 0.0, 0.0, 0.0])
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=query_emb.tolist())]

        with patch("travelplanner.agents.attraction_search_agent.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = mock_resp
            name, _ = _select_archetype("family trip with two kids", config)

        self.assertEqual(name, "family_with_kids")


if __name__ == "__main__":
    unittest.main()
