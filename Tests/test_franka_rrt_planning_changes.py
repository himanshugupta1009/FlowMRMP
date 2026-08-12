"""Focused regression tests for the Franka RRT/FlowEBRRT planning changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT / "mrmp_with_kite_extend" / "src"
MAIN_SCRIPTS = ROOT / "scripts"
for path in (ROOT, MAIN_SCRIPTS, MRMP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from FrankaPanda import FrankaPanda, Q_LOWER, Q_UPPER  # noqa: E402
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

    def test_relative_outcome_sort_uses_position_not_velocity(self):
        starts = np.zeros((3, 14), dtype=np.float64)
        relative = np.zeros((3, 14), dtype=np.float64)
        relative[:, 0] = [0.10, 0.20, 0.30]
        # Edge 0 must remain first despite a very poor predicted velocity.
        # This also confirms that sorting uses predictions without propagating
        # any of the zero control sequences below.
        relative[0, 7] = 10.0
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
        np.testing.assert_array_equal(order, np.array([0, 1, 2]))

    def test_sequence_bundle_can_pool_four_independent_generations(self):
        def bundle(offset):
            starts = np.zeros((2, 14), dtype=np.float64)
            relative = np.zeros((2, 14), dtype=np.float64)
            relative[:, 0] = [offset, offset + 0.1]
            return GeneratedSequenceEdgeBundle(
                action_sequences=[np.zeros((2, 7)), np.zeros((3, 7))],
                action_dt=0.02,
                start_states=starts,
                final_states=starts + relative,
                relative_changes=relative,
            )

        pooled = bundle(0.0)
        for offset in (1.0, 2.0, 3.0):
            pooled.extend(bundle(offset))
        self.assertEqual(pooled.num_edges, 8)
        self.assertEqual(len(pooled.action_sequences), 8)
        self.assertEqual(pooled.final_states.shape, (8, 14))
        self.assertEqual(pooled.relative_changes.shape, (8, 14))

    def test_goal_pool_ranks_physical_goal_crossing_by_earliest_entry(self):
        sequences = [
            np.array([[1.0], [1.0], [1.0]]),
            np.array([[2.0], [1.0]]),
            np.array([[0.5], [0.5]]),
        ]
        starts = np.zeros((3, 1), dtype=np.float64)
        bundle = GeneratedSequenceEdgeBundle(
            action_sequences=sequences,
            action_dt=0.02,
            start_states=starts,
            final_states=np.array([[100.0], [200.0], [300.0]]),
            relative_changes=np.array([[100.0], [200.0], [300.0]]),
        )

        class _PhysicalSequenceAgent:
            @staticmethod
            def get_next_state_sequence(state, actions, _dt):
                path = np.asarray(state)[None, :] + np.cumsum(actions, axis=0)
                return path[-1], path

        fake_planner = SimpleNamespace(
            agent=_PhysicalSequenceAgent(),
            goal=np.array([2.0]),
            goal_radius=0.1,
            profile={
                "goal_pool_propagated_edges": 0,
                "goal_pool_goal_crossing_edges": 0,
                "sequence_jerk_rejections": 0,
                "sequence_acceleration_rejections": 0,
            },
            _deadline_reached=lambda: False,
            reached_goal=lambda state, goal, radius, _agent: (
                abs(float(state[0] - goal[0])) <= radius,
                abs(float(state[0] - goal[0])),
            ),
        )
        parent = SimpleNamespace(
            state=np.array([0.0]),
            flow_edge_bundle=bundle,
            edge_bundle_indices=np.arange(3, dtype=np.int64),
            edge_bundle_mask=np.zeros(3, dtype=bool),
        )
        goals = FlowEBRRT._rank_goal_pool_candidates(fake_planner, parent)
        self.assertEqual([item["edge_index"] for item in goals], [1, 0])
        self.assertEqual([item["goal_index"] for item in goals], [0, 1])
        self.assertEqual(fake_planner.profile["goal_pool_propagated_edges"], 3)
        self.assertEqual(fake_planner.profile["goal_pool_goal_crossing_edges"], 2)

    def test_franka_agent_goal_policy_uses_q_and_ordinary_policy_uses_q_dq(self):
        agent = FrankaPanda(
            state_bank=np.zeros((1, 14), dtype=np.float64),
            collision_checker=_NoCollision(),
        )
        self.assertEqual(agent.get_parent_selection_metric_dims(True), 7)
        self.assertEqual(agent.get_parent_selection_metric_dims(False), 14)

        target = np.zeros(14, dtype=np.float64)
        position_close_velocity_far = target.copy()
        position_close_velocity_far[0] = 0.10
        position_close_velocity_far[7] = 2.0
        position_far_velocity_close = target.copy()
        position_far_velocity_close[0] = 0.50

        self.assertLess(
            agent.get_extension_score(position_close_velocity_far, target, True),
            agent.get_extension_score(position_far_velocity_close, target, True),
        )
        self.assertGreater(
            agent.get_extension_score(position_close_velocity_far, target, False),
            agent.get_extension_score(position_far_velocity_close, target, False),
        )

    def test_rrt_uses_agent_goal_parent_policy(self):
        agent = FrankaPanda(
            state_bank=np.zeros((1, 14), dtype=np.float64),
            collision_checker=_NoCollision(),
        )
        states = np.zeros((2, 14), dtype=np.float64)
        states[0, 0] = 0.10
        states[0, 7] = 2.0
        states[1, 0] = 0.50

        class _Matrix:
            count = 2
            ids = np.array([10, 11], dtype=np.int64)

            @staticmethod
            def get_valid_matrix():
                return states

        planner = SimpleNamespace(
            agent=agent,
            _last_sample_was_goal=True,
            distance_metric_state_size=14,
            _node_matrix=_Matrix(),
            _distance_metric_state=lambda state: np.asarray(state, dtype=np.float64),
            tree=SimpleNamespace(
                nodes={
                    10: {"value": SimpleNamespace(state=states[0])},
                    11: {"value": SimpleNamespace(state=states[1])},
                }
            ),
            debug_flag=False,
        )
        node_id, _node = RRT.get_nearest_node(planner, np.zeros(14))
        self.assertEqual(node_id, 10)

        planner._last_sample_was_goal = False
        node_id, _node = RRT.get_nearest_node(planner, np.zeros(14))
        self.assertEqual(node_id, 11)

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

    def test_start_inside_goal_requires_safe_future_parking(self):
        planner = _make_crossing_rrt()
        planner.goal = planner.start.copy()
        planner.goal_radius = 0.01
        planner.max_iter = -1
        planner.dynamic_col_checker_to_end = lambda *_args, **_kwargs: True
        planner.plan_path()
        self.assertFalse(planner.path_found)
        self.assertIsNone(planner.goal_node_id)

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
