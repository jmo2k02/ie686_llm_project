from __future__ import annotations

import unittest

from run_search_agent import SEARCH_AGENT_SPECS


class TestSearchAgentRegistry(unittest.TestCase):
    def test_general_web_search_is_registered(self) -> None:
        self.assertIn("general_web_search", SEARCH_AGENT_SPECS)
        spec = SEARCH_AGENT_SPECS["general_web_search"]
        self.assertEqual(spec.task_type, "general-web-search")
        self.assertEqual(spec.artifact_key, "general_web_search_agent")


if __name__ == "__main__":
    unittest.main()
