from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for module_path in (
    ROOT / "scripts",
    ROOT / "scripts" / "FrankaObstacle",
    ROOT / "mrmp_with_kite_extend" / "src",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from franka_prioritized_adapter import (  # noqa: E402
    BASE_TRANSFORMS,
    FrankaPrioritizedCollisionAdapter,
    rotation_z,
    static_scene_for_robot,
)
from prioritized_planning import PrioritizedPlanning  # noqa: E402


def test_world_base_transform_round_trip_and_inward_axes():
    points = np.asarray([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0]])
    for transform in BASE_TRANSFORMS:
        restored = transform.points_to_base(transform.points_to_world(points))
        assert np.allclose(restored, points, atol=1e-12)

    north_forward = rotation_z(np.deg2rad(-90.0)) @ np.asarray([1.0, 0.0, 0.0])
    south_forward = rotation_z(np.deg2rad(90.0)) @ np.asarray([1.0, 0.0, 0.0])
    assert np.allclose(north_forward, [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(south_forward, [0.0, 1.0, 0.0], atol=1e-12)


def test_static_scene_excludes_only_own_fixture():
    for robot_index in range(4):
        cuboids = static_scene_for_robot(robot_index)["cuboid"]
        assert "shared_table" in cuboids
        assert len(cuboids) == 7
        own_name = BASE_TRANSFORMS[robot_index].name
        assert f"{own_name}_pedestal" not in cuboids
        assert f"{own_name}_mount" not in cuboids


def test_batched_sphere_overlap_mask():
    torch = pytest.importorskip("torch")
    adapter = FrankaPrioritizedCollisionAdapter.__new__(
        FrankaPrioritizedCollisionAdapter
    )
    adapter.torch = torch
    adapter.dynamic_clearance = 0.0
    current = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.2]], [[0.0, 0.0, 0.0, 0.2]]]
    )
    obstacle = torch.tensor(
        [[[0.3, 0.0, 0.0, 0.2]], [[1.0, 0.0, 0.0, 0.2]]]
    )
    collision = adapter._sphere_overlap_mask(
        current, obstacle, pair_counter="test"
    )
    assert collision.tolist() == [True, False]


class _FakeAgent:
    id = 0
    distance_metric_state_size = 2
    radius = 0.1


class _FakePlanner:
    def __init__(self):
        self.agent = _FakeAgent()
        self.dynamic_agent_obstacles = None
        self.planning_time = None
        self.path_found = True
        self.path_cost = 2.5

    def plan_path(self):
        return None

    def get_high_resolution_path_numpy_array(self):
        return np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)


class _FakeAdapter:
    def __init__(self):
        self.events = []

    def reset(self):
        self.events.append("reset")

    def prepare_planner(self, planner, index):
        self.events.append(("prepare", index))

    def finish_planner(self, planner, index, dense_path, *, success):
        self.events.append(("finish", index, success, dense_path.shape))


def test_prioritized_planning_optional_adapter_hook():
    adapter = _FakeAdapter()
    solved, _, cost = PrioritizedPlanning.plan_multi(
        planners=[_FakePlanner()],
        planning_time=1.0,
        dynamic_obstacle_adapter=adapter,
    )
    assert solved
    assert cost == 2.5
    assert adapter.events == ["reset", ("prepare", 0), ("finish", 0, True, (2, 2))]
