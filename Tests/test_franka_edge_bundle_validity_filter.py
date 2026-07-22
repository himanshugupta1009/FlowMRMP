from pathlib import Path
import importlib.util

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "create_franka_edge_bundle_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("create_franka_edge_bundle_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_raw() -> dict[str, object]:
    accelerations = np.zeros((8, 7), dtype=np.float32)
    accelerations[2:4, 0] = 20.0
    accelerations[4, 0] = -15.0
    accelerations[5, 0] = 15.0
    return {
        "trajectory_index": np.zeros(4, dtype=np.int32),
        "start_index": np.array([0, 2, 4, 6], dtype=np.int32),
        "num_steps": np.full(4, 2, dtype=np.int32),
        "acceleration_offsets": np.array([0], dtype=np.int64),
        "accelerations": accelerations,
        "dt": 0.02,
        "max_duration": 0.04,
    }


def test_complete_dynamic_filter_rejects_acceleration_jerk_and_limit_failures():
    raw = synthetic_raw()
    center = np.zeros(14, dtype=np.float32)
    valid, _, _, _ = MODULE.integrate_candidate_rollouts(
        center, np.arange(4), raw
    )
    assert valid.tolist() == [True, False, False, True]

    near_upper_with_positive_velocity = np.zeros(14, dtype=np.float32)
    near_upper_with_positive_velocity[0] = 0.999
    near_upper_with_positive_velocity[7] = 1.0
    valid, _, _, _ = MODULE.integrate_candidate_rollouts(
        near_upper_with_positive_velocity, np.array([0]), raw
    )
    assert not bool(valid[0])


class ThresholdCollisionChecker:
    def in_collision(self, q: np.ndarray) -> bool:
        return bool(q[0] > 0.5)


def test_collision_filter_selects_only_complete_collision_free_rollouts():
    candidate_ids = np.array([10, 11, 12], dtype=np.int64)
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    q_paths = np.zeros((3, 3, 7), dtype=np.float64)
    q_paths[0, 1, 0] = 0.75
    counts = np.full(3, 3, dtype=np.int64)
    selected, stats = MODULE.select_collision_free_fps(
        candidate_ids,
        features,
        q_paths,
        counts,
        set_size=2,
        collision_checker=ThresholdCollisionChecker(),
        rng=np.random.default_rng(11),
    )
    assert selected is not None
    assert set(selected.tolist()) == {11, 12}
    assert stats["self_collision_candidate_rejections"] == 1
