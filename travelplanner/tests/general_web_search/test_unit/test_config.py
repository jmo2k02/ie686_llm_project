from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from travelplanner.agents.general_web_search_agent import load_config_from_env


class TestGeneralWebSearchConfig(unittest.TestCase):
    def test_defaults_include_openrouter_minimax_answer_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config_from_env()
        self.assertEqual(cfg.max_results, 5)
        self.assertEqual(cfg.timeout_seconds, 30)
        self.assertEqual(cfg.max_retries, 1)
        self.assertEqual(cfg.max_searches, 3)
        self.assertEqual(cfg.answer_model_name, "openrouter:minimax/minimax-m2.5")

    def test_env_overrides_take_effect(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RESULTS": "8",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_TIMEOUT_SECONDS": "45",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES": "3",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_SEARCHES": "2",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_DEPTH": "advanced",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_INCLUDE_ANSWER": "false",
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL": "none",
            },
            clear=True,
        ):
            cfg = load_config_from_env()
        self.assertEqual(cfg.max_results, 8)
        self.assertEqual(cfg.timeout_seconds, 45)
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.max_searches, 2)
        self.assertEqual(cfg.search_depth, "advanced")
        self.assertFalse(cfg.include_answer)
        self.assertIsNone(cfg.answer_model_name)


if __name__ == "__main__":
    unittest.main()
