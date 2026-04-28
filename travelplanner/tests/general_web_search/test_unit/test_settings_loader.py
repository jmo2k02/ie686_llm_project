from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from travelplanner.config.settings import load_settings


class TestSettingsLoader(unittest.TestCase):
    def test_local_config_overrides_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            global_cfg = tmp_path / "global.yaml"
            local_cfg = tmp_path / "local.yaml"
            global_cfg.write_text(
                "agents:\n  general_web_search:\n    max_results: 5\n",
                encoding="utf-8",
            )
            local_cfg.write_text(
                "agents:\n  general_web_search:\n    max_results: 9\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "TRAVELPLANNER_GLOBAL_CONFIG_PATH": str(global_cfg),
                    "TRAVELPLANNER_LOCAL_CONFIG_PATH": str(local_cfg),
                },
                clear=False,
            ):
                settings = load_settings(force_reload=True)
        self.assertEqual(settings["agents"]["general_web_search"]["max_results"], 9)


if __name__ == "__main__":
    unittest.main()
