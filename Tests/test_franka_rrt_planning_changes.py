"""Focused regression tests for the Franka RRT/FlowEBRRT planning changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT / "mrmp_with_kite_extend" / "src"
for path in (ROOT, MRMP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Agents.FrankaPanda import FrankaPanda, Q_LOWER, Q_UPPER  # noqa: E402
from rrt import RRT  # noqa: E402
from src.flow_eb_rrt import (  # noqa: E402
    FlowEBRRT,
    GeneratedSequenceEdgeBundle,
)


class _NoCollision:
    @staticmethod
    def in_collision(_q):
        return False


class _DummyEnvironment:
    size = np.ones(1, dtype=np.float64)
    obstacle_buffer = 0.0
    boundary_buffer = 0.0
    static_circular_obstacles = np.empty((0, 3), dtype=np.float64)
    static_rectangular_obstacles = np.empty((0, 4), dtype=np.float64)


class _CrossingAgent:
    radius = 0.0
    state_length = 1
    action_length = 1
    distance_metric_state_size = 1
    disable_dynamic_collision_check = True
    dynamic_limit_indices = np.array([0], dtype=np.int64)
    dynamic_limit_values = np.array([10.0], dtype=np.float64)
    id = 0

    @staticmethod
    def get_distance_metric_state(state):
        return np.asarray(state, dtype=np.float64)

    @staticmethod
    def get_distance(state, target):
        return float(np.linalg.norm(np.asarray(state) - np.asarray(target)))

    @staticmethod
    def get_random_action(_rng):
        return np.array([1.0], dtype=np.float64)

    @staticmethod
    def get_random_point(*_args):
        return np.array([5.0], dtype=np.float64)

    @staticmethod
    def get_next_state(_state, _action, _duration, num_steps):
        path = np.arange(1, num_steps + 1, dtype=np.float64)[:, None]
        return path[-1], path

    @staticmethod
    def reached(state, goal, radius, _agent):
        distance = float(np.linalg.norm(np.asarray(state) - np.asarray(goal)))
        return distance <= radius, distance

    @staticmethod
    def cost(*_args):
        return 0.0


def _make_crossing_rrt() -> RRT:
    agent = _CrossingAgent()
    return RRT(
        start=np.array([0.0]),
        goal=np.array([2.0]),
        goal_radius=0.01,
        env=_DummyEnvironment(),
        agent=agent,
        use_fixed_sampling_time=True,
        sampling_time_step=1.0,
        minimum_time_step=0.2,
        max_iter=1,
        planning_time=1.0,
        num_extension_trials=1,
        isvalid_function=lambda *_args: True,
        cost_function=agent.cost,
        reached_goal_function=agent.reached,
        random_point_function=agent.get_random_point,
        udf_seed=4,
    )


class FrankaPlanningChangeTests(unittest.TestCase):
    def test_acceleration_and_intra_edge_jerk_limits(self):
        valid = np.zeros((3, 7), dtype=np.float64)
        self.assertIsNone(FrankaPanda.action_sequence_violation(valid, 0.02))

        excessive_acceleration = valid.copy()
        excessive_acceleration[0, 0] = 15.1
        self.assertEqual(
            FrankaPanda.action_sequence_violation(excessive_acceleration, 0.02),
            "acceleration",
        )

        excessive_jerk = valid.copy()
        excessive_jerk[1, 0] = 10.1
        self.assertEqual(
            FrankaPanda.action_sequence_violation(excessive_jerk, 0.02),
            "jerk",
        )

    def test_relative_outcome_sort_does_not_propagate_controls(self):
        starts = np.zeros((3, 14), dtype=np.float64)
        relative = np.zeros((3, 14), dtype=np.float64)
        relative[:, 0] = [2.0, 0.25, 1.0]
        bundle = GeneratedSequenceEdgeBundle(
            action_sequences=[np.zeros((50, 7)) for _ in range(3)],
            action_dt=0.02,
            start_states=starts,
            final_states=starts + relative,
            relative_changes=relative,
        )
        fake_planner = SimpleNamespace(
            distance_array=np.zeros(3, dtype=np.float64),
            _distance_metric_state=lambda state: np.asarray(
                state, dtype=np.float64
            ),
        )
        order, count = FlowEBRRT._sort_sequence_edges(
            fake_planner,
            bundle,
            np.zeros(14, dtype=np.float64),
            np.arange(3, dtype=np.int64),
            np.zeros(3, dtype=bool),
        )
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(order, np.array([1, 2, 0]))

    def test_goal_crossing_is_truncated_even_when_full_endpoint_is_far(self):
        planner = _make_crossing_rrt()
        parent = SimpleNamespace(state=np.array([0.0]), time_elapsed=0.0)
        candidate = planner._select_best_extension_candidate(
            parent, np.array([5.0])
        )
        self.assertIsNotNone(candidate)
        state, path, _action, duration = candidate
        np.testing.assert_allclose(state, np.array([2.0]))
        np.testing.assert_allclose(path[:, 0], np.array([1.0, 2.0]))
        self.assertAlmostEqual(duration, 0.4)

    def test_random_durations_are_integer_integration_steps(self):
        planner = _make_crossing_rrt()
        planner.use_fixed_sampling_time = False
        planner.minimum_time_step = 0.02
        planner.max_sample_T = 1.0
        planner.roundoff_digits = 2
        samples = np.array([planner.get_random_time() for _ in range(1000)])
        self.assertGreaterEqual(samples.min(), 0.02)
        self.assertLessEqual(samples.max(), 1.0)
        np.testing.assert_allclose(samples / 0.02, np.rint(samples / 0.02))

    def test_franka_goal_uses_only_position_while_rrt_distance_remains_14d(self):
        agent = FrankaPanda(
            state_bank=np.zeros((1, 14), dtype=np.float64),
            collision_checker=_NoCollision(),
        )
        state = np.zeros(14, dtype=np.float64)
        goal = state.copy()
        goal[7] = 1.0
        reached, distance = agent.agent_reached_goal(state, goal, 0.0, agent)
        self.assertTrue(reached)
        self.assertEqual(distance, 0.0)
        self.assertGreater(agent.get_distance(state, goal), 0.0)

        goal[0] = 0.2 * (Q_UPPER[0] - Q_LOWER[0])
        reached, distance = agent.agent_reached_goal(state, goal, 0.25, agent)
        self.assertFalse(reached)
        self.assertAlmostEqual(distance, 0.4)


if __name__ == "__main__":
    unittest.main()
