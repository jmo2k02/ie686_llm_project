from __future__ import annotations

import unittest

from travelplanner.schema.system_state import TaskModel, get_allowed_task_types


class TestTaskModel(unittest.TestCase):
    def test_get_allowed_task_types_matches_task_model_literal(self) -> None:
        allowed = get_allowed_task_types()

        self.assertEqual(
            allowed,
            (
                "flight",
                "hotel",
                "restaurant",
                "attraction",
                "opening_times",
                "routing-check",
                "general-web-search",
            ),
        )

        for task_type in allowed:
            TaskModel(
                name="test",
                type=task_type,
                text="test task text",
                is_valid=True
            )


if __name__ == "__main__":
    unittest.main()
