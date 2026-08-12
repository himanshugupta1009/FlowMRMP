"""Tests for Franka physical path-time quality metrics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FRANKA_TESTING = ROOT / "scripts" / "FrankaTesting"
if str(FRANKA_TESTING) not in sys.path:
    sys.path.insert(0, str(FRANKA_TESTING))

from franka_solution_quality import (  # noqa: E402
    both_success_path_quality,
    path_motion_time_from_states,
    solution_quality_summary,
)


class FrankaSolutionQualityTests(unittest.TestCase):
    def test_dense_path_time_counts_integration_intervals(self):
        states = np.zeros((51, 14), dtype=np.float64)
        self.assertAlmostEqual(path_motion_time_from_states(states, 0.02), 1.0)

    def test_quality_excludes_failures_and_compares_witness(self):
        rows = [
            {
                "success": True,
                "path_motion_time_seconds": 1.0,
                "witness_duration_seconds": 2.0,
            },
            {
                "success": "true",
                "path_motion_time_seconds": 3.0,
                "witness_duration_seconds": 2.0,
            },
            {
                "success": False,
                "path_motion_time_seconds": np.nan,
                "witness_duration_seconds": 2.0,
            },
        ]
        summary = solution_quality_summary(rows)
        self.assertEqual(summary["successful_paths"]["count"], 2)
        self.assertAlmostEqual(summary["successful_paths"]["mean"], 2.0)
        self.assertAlmostEqual(
            summary["path_time_over_feasible_witness"]["median"], 1.0
        )

    def test_paired_quality_uses_only_both_successes(self):
        first = [
            {"success": True, "path_motion_time_seconds": 1.0},
            {"success": True, "path_motion_time_seconds": 2.0},
        ]
        second = [
            {"success": True, "path_motion_time_seconds": 1.5},
            {"success": False, "path_motion_time_seconds": np.nan},
        ]
        summary = both_success_path_quality(
            first, second, first_name="vanilla", second_name="flow"
        )
        self.assertEqual(summary["paired_both_success_count"], 1)
        self.assertAlmostEqual(summary["flow_minus_vanilla_seconds"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
