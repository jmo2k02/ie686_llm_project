from __future__ import annotations

import unittest

from travelplanner.schema.general_web_search_artifact import (
    GeneralWebSearchArtifactContentModel,
)


class TestArtifactSchema(unittest.TestCase):
    def test_required_fields_and_enums(self) -> None:
        payload = {
            "task_ref": "general-search-1",
            "query": "test query",
            "provider": "tavily",
            "status": "success",
            "attempt": 1,
            "result": {"ok": True, "query": "q"},
            "answer": "Test answer text",
            "model": "openrouter:x",
            "proof_points": [],
            "errors": [],
            "config": {},
        }
        parsed = GeneralWebSearchArtifactContentModel.model_validate(payload)
        self.assertEqual(parsed.provider, "tavily")
        self.assertEqual(parsed.status, "success")


if __name__ == "__main__":
    unittest.main()
