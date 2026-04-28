from __future__ import annotations

import unittest

from travelplanner.agents.general_web_search_agent import _extract_search_tasks
from travelplanner.schema.system_state import TaskModel


class TestTaskFiltering(unittest.TestCase):
    def test_only_valid_general_web_search_tasks_are_selected(self) -> None:
        tasks = [
            TaskModel(
                name="a",
                type="general-web-search",
                text="find weather in rome",
                is_valid=True,
                validation_comment=None,
            ),
            TaskModel(
                name="b",
                type="general-web-search",
                text="find event dates",
                is_valid=False,
                validation_comment="needs detail",
            ),
            TaskModel(
                name="c",
                type="hotel",
                text="book hotel",
                is_valid=True,
                validation_comment=None,
            ),
        ]
        selected = _extract_search_tasks(tasks)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "a")


if __name__ == "__main__":
    unittest.main()
