from __future__ import annotations

import unittest
from pathlib import Path

from slam_icp.config import load_slam_config


class SlamConfigModeTests(unittest.TestCase):
    def test_tuning_config_enables_auto_mode(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_slam_config(repository_root / "config" / "slam_tuning_01.yaml")

        self.assertEqual(config.data["loop_closure"]["mode"], "auto")
        self.assertEqual(config.data["loop_closure"]["manual_pairs"], [])
        self.assertEqual(config.data["loop_closure"]["maximum_accepted_loops"], 1)

    def test_preserved_config_is_loaded_as_manual_mode(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_slam_config(repository_root / "config" / "slam.yaml")

        self.assertEqual(config.data["loop_closure"]["mode"], "manual")
        self.assertEqual(
            config.data["loop_closure"]["manual_pairs"],
            [{"source_id": 143, "target_id": 596}],
        )


if __name__ == "__main__":
    unittest.main()
