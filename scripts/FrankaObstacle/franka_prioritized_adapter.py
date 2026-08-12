#!/usr/bin/env python3
"""cuRobo/GPU collision adapter for prioritized four-Franka planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from FrankaPanda import (
    DEFAULT_CUROBO_CONFIG,
    Q_LOWER,
    Q_UPPER,
    FrankaCuroboCollisionChecker,
    FrankaPanda,
)
from four_franka_table_scene import (
    MOUNT_DIMS,
    PEDESTAL_DIMS,
    ROBOT_LAYOUT,
    TABLE_DIMS,
    TABLE_POSE,
)


ROBOT_NAMES = tuple(robot["name"] for robot in ROBOT_LAYOUT)


def rotation_z(yaw_radians: float) -> np.ndarray:
    """Return the active local-to-world rotation around +Z."""
    c = math.cos(yaw_radians)
    s = math.sin(yaw_radians)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


@dataclass(frozen=True)
class FrankaBaseTransform:
    """Fixed transform between one Franka base and the shared world frame."""

    name: str
    translation: np.ndarray
    yaw_radians: float
    rotation_world_from_base: np.ndarray

    @classmethod
    def from_layout(cls, robot: dict[str, object]) -> "FrankaBaseTransform":
        yaw = math.radians(float(robot["yaw_deg"]))
        return cls(
            name=str(robot["name"]),
            translation=np.asarray(robot["position"], dtype=np.float64),
            yaw_radians=yaw,
            rotation_world_from_base=rotation_z(yaw),
        )

    def points_to_world(self, points_base: np.ndarray) -> np.ndarray:
        points = np.asarray(points_base, dtype=np.float64)
        return points @ self.rotation_world_from_base.T + self.translation

    def points_to_base(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        return (points - self.translation) @ self.rotation_world_from_base

    def world_cuboid_pose_to_base(
        self, position_world: tuple[float, float, float], yaw_world: float = 0.0
    ) -> list[float]:
        center = self.points_to_base(np.asarray(position_world, dtype=np.float64))
        local_yaw = float(yaw_world) - self.yaw_radians
        half = 0.5 * local_yaw
        # cuRobo scene poses use [x, y, z, qw, qx, qy, qz].
        return [
            *center.tolist(),
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ]


BASE_TRANSFORMS = tuple(
    FrankaBaseTransform.from_layout(robot) for robot in ROBOT_LAYOUT
)


def static_scene_for_robot(robot_index: int) -> dict[str, object]:
    """Build the table/other-pedestal scene in one arm's base frame."""
    transform = BASE_TRANSFORMS[robot_index]
    cuboids: dict[str, dict[str, object]] = {
        "shared_table": {
            "pose": transform.world_cuboid_pose_to_base(TABLE_POSE),
            "dims": list(TABLE_DIMS),
        }
    }
    for other_index, robot in enumerate(ROBOT_LAYOUT):
        if other_index == robot_index:
            # The owning pedestal terminates at the robot mounting plane. It is
            # deliberately excluded so link0 is not tested against its fixture.
            continue
        x, y, _ = robot["position"]
        cuboids[f"{robot['name']}_pedestal"] = {
            "pose": transform.world_cuboid_pose_to_base(
                (x, y, PEDESTAL_DIMS[2] * 0.5)
            ),
            "dims": list(PEDESTAL_DIMS),
        }
        cuboids[f"{robot['name']}_mount"] = {
            "pose": transform.world_cuboid_pose_to_base(
                (x, y, PEDESTAL_DIMS[2] + MOUNT_DIMS[2] * 0.5)
            ),
            "dims": list(MOUNT_DIMS),
        }
    return {"cuboid": cuboids}


@dataclass
class CachedSphereTrajectory:
    """One higher-priority robot trajectory held in GPU world coordinates."""

    robot_index: int
    spheres_world: Any
    integration_dt: float


class PrioritizedFrankaPanda(FrankaPanda):
    """Franka agent whose edge validity includes higher-priority arms."""

    def __init__(self, *, collision_adapter, robot_index: int, **kwargs):
        super().__init__(agent_id=robot_index, **kwargs)
        self.collision_adapter = collision_adapter
        self.robot_index = int(robot_index)
        self.inter_robot_rejections = 0

    def is_new_node_valid(self, path_to_new_state: np.ndarray, *args) -> bool:
        states = np.asarray(path_to_new_state, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 14:
            raise ValueError(f"Expected a waypoint path shaped (N, 14), got {states.shape}")
        if len(states) == 0:
            return False

        start_time = float(args[-3])
        step_time = float(args[-1])
        limit_valid = np.asarray(
            [self.is_state_within_limits(state) for state in states], dtype=bool
        )
        q = np.clip(states[:, :7], Q_LOWER, Q_UPPER)
        static_valid, dynamic_valid = self.collision_adapter.validate_path(
            self.robot_index,
            q,
            start_time=start_time,
            step_time=step_time,
        )
        combined = limit_valid & static_valid & dynamic_valid
        invalid = np.flatnonzero(~combined)
        if invalid.size:
            first = int(invalid[0])
            self.checked_waypoints += first + 1
            if not limit_valid[first]:
                self.limit_rejections += 1
            elif not static_valid[first]:
                self.self_collision_rejections += 1
            else:
                self.inter_robot_rejections += 1
            return False
        self.checked_waypoints += len(states)
        return True


class FrankaPrioritizedCollisionAdapter:
    """Bridge prioritized planning to 295-sphere articulated GPU collision."""

    def __init__(
        self,
        *,
        state_bank: np.ndarray,
        config_path: str | Path = DEFAULT_CUROBO_CONFIG,
        device: str = "cuda:0",
        dynamic_clearance: float = 0.0,
        acceleration_scale: float = 0.5,
    ):
        import torch

        self.torch = torch
        self.device = device
        self.dynamic_clearance = float(dynamic_clearance)
        self.state_bank = np.asarray(state_bank, dtype=np.float64)
        self.checkers = [
            FrankaCuroboCollisionChecker(
                config_path,
                scene_model=static_scene_for_robot(robot_index),
                device=device,
            )
            for robot_index in range(len(ROBOT_LAYOUT))
        ]
        self.agents = [
            PrioritizedFrankaPanda(
                state_bank=self.state_bank,
                collision_checker=self.checkers[robot_index],
                collision_adapter=self,
                robot_index=robot_index,
                acceleration_scale=acceleration_scale,
            )
            for robot_index in range(len(ROBOT_LAYOUT))
        ]
        self._rotations = [
            torch.as_tensor(
                transform.rotation_world_from_base,
                device=self.checkers[index].device_cfg.device,
                dtype=self.checkers[index].device_cfg.dtype,
            )
            for index, transform in enumerate(BASE_TRANSFORMS)
        ]
        self._translations = [
            torch.as_tensor(
                transform.translation,
                device=self.checkers[index].device_cfg.device,
                dtype=self.checkers[index].device_cfg.dtype,
            )
            for index, transform in enumerate(BASE_TRANSFORMS)
        ]
        self.trajectories: list[CachedSphereTrajectory] = []
        self.stage_records: list[dict[str, object]] = []
        self._stage_before: dict[int, dict[str, object]] = {}
        self.robot_stats = [self._empty_robot_stats() for _ in ROBOT_LAYOUT]

    @staticmethod
    def _empty_robot_stats() -> dict[str, float | int]:
        return {
            "collision_batches": 0,
            "collision_waypoints": 0,
            "sphere_fk_seconds": 0.0,
            "self_static_seconds": 0.0,
            "dynamic_collision_seconds": 0.0,
            "dynamic_sphere_pair_tests": 0,
            "dynamic_collision_batches_rejected": 0,
            "goal_parking_queries": 0,
            "goal_parking_sphere_pair_tests": 0,
        }

    def reset(self) -> None:
        self.trajectories.clear()
        self.stage_records.clear()
        self._stage_before.clear()
        self.robot_stats = [self._empty_robot_stats() for _ in ROBOT_LAYOUT]
        for agent in self.agents:
            agent.checked_waypoints = 0
            agent.limit_rejections = 0
            agent.self_collision_rejections = 0
            agent.inter_robot_rejections = 0

    def close(self) -> None:
        for checker in self.checkers:
            checker.close()

    def _joint_tensor(self, robot_index: int, q: np.ndarray):
        checker = self.checkers[robot_index]
        return self.torch.as_tensor(
            np.asarray(q, dtype=np.float32),
            device=checker.device_cfg.device,
            dtype=checker.device_cfg.dtype,
        ).unsqueeze(0)

    def _world_spheres(self, robot_index: int, local_spheres):
        world = local_spheres.clone()
        rotation = self._rotations[robot_index]
        translation = self._translations[robot_index]
        world[..., :3] = local_spheres[..., :3] @ rotation.T + translation
        return world

    def configuration_world_spheres(self, robot_index: int, q: np.ndarray):
        checker = self.checkers[robot_index]
        q_tensor = self._joint_tensor(robot_index, np.atleast_2d(q))
        with self.torch.inference_mode():
            state = checker.checker.get_kinematics(q_tensor)
            return self._world_spheres(robot_index, state.robot_spheres).squeeze(0)

    def _sphere_overlap_mask(self, current, obstacle, *, pair_counter: str):
        """Return one collision Boolean per aligned time sample."""
        if current.shape[0] != obstacle.shape[0]:
            raise ValueError("Sphere trajectories must have aligned time dimensions")
        result = self.torch.zeros(
            current.shape[0], device=current.device, dtype=self.torch.bool
        )
        chunk_size = 8
        for start in range(0, current.shape[0], chunk_size):
            stop = min(start + chunk_size, current.shape[0])
            a = current[start:stop]
            b = obstacle[start:stop]
            delta = a[:, :, None, :3] - b[:, None, :, :3]
            threshold = (
                a[:, :, None, 3]
                + b[:, None, :, 3]
                + self.dynamic_clearance
            )
            result[start:stop] = self.torch.any(
                self.torch.sum(delta * delta, dim=-1) < threshold * threshold,
                dim=(1, 2),
            )
        return result

    def validate_path(
        self,
        robot_index: int,
        q: np.ndarray,
        *,
        start_time: float,
        step_time: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        checker = self.checkers[robot_index]
        stats = self.robot_stats[robot_index]
        stats["collision_batches"] += 1
        stats["collision_waypoints"] += len(q)
        q_tensor = self._joint_tensor(robot_index, q)

        started = time.perf_counter()
        with self.torch.inference_mode():
            kin_state = checker.checker.get_kinematics(q_tensor)
        stats["sphere_fk_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        with self.torch.inference_mode():
            self_cost = checker.checker.get_self_collision(kin_state.robot_spheres)
            scene_cost = checker.checker.get_collision_constraint(kin_state)
            static_valid_tensor = (
                (self_cost.sum(dim=-1) == 0.0)
                & (scene_cost.sum(dim=-1) == 0.0)
            ).squeeze(0)
        static_valid = static_valid_tensor.detach().cpu().numpy().astype(bool)
        stats["self_static_seconds"] += time.perf_counter() - started

        if not self.trajectories:
            return static_valid, np.ones(len(q), dtype=bool)

        started = time.perf_counter()
        current_world = self._world_spheres(
            robot_index, kin_state.robot_spheres
        ).squeeze(0)
        times = start_time + step_time * np.arange(1, len(q) + 1, dtype=np.float64)
        dynamic_valid_tensor = self.torch.ones(
            len(q), device=current_world.device, dtype=self.torch.bool
        )
        for trajectory in self.trajectories:
            indices = np.rint(times / trajectory.integration_dt).astype(np.int64)
            indices = np.clip(indices, 0, trajectory.spheres_world.shape[0] - 1)
            index_tensor = self.torch.as_tensor(
                indices, device=current_world.device, dtype=self.torch.long
            )
            obstacle_world = trajectory.spheres_world.index_select(0, index_tensor)
            colliding = self._sphere_overlap_mask(
                current_world, obstacle_world, pair_counter="dynamic"
            )
            dynamic_valid_tensor &= ~colliding
            stats["dynamic_sphere_pair_tests"] += int(
                len(q) * current_world.shape[1] * obstacle_world.shape[1]
            )
        dynamic_valid = dynamic_valid_tensor.detach().cpu().numpy().astype(bool)
        if not bool(np.all(dynamic_valid)):
            stats["dynamic_collision_batches_rejected"] += 1
        stats["dynamic_collision_seconds"] += time.perf_counter() - started
        return static_valid, dynamic_valid

    def parking_blocked(
        self,
        robot_index: int,
        state: np.ndarray,
        arrival_time: float,
        timestep: float,
    ) -> bool:
        del timestep
        if not self.trajectories:
            return False
        stats = self.robot_stats[robot_index]
        stats["goal_parking_queries"] += 1
        current = self.configuration_world_spheres(robot_index, state[:7])[:1]
        for trajectory in self.trajectories:
            first = int(round(arrival_time / trajectory.integration_dt)) + 1
            if first >= trajectory.spheres_world.shape[0]:
                future = trajectory.spheres_world[-1:]
            else:
                future = trajectory.spheres_world[first:]
            for start in range(0, future.shape[0], 8):
                obstacle = future[start : start + 8]
                held = current.expand(obstacle.shape[0], -1, -1)
                collision = self._sphere_overlap_mask(
                    held, obstacle, pair_counter="parking"
                )
                stats["goal_parking_sphere_pair_tests"] += int(
                    obstacle.shape[0] * held.shape[1] * obstacle.shape[1]
                )
                if bool(self.torch.any(collision).item()):
                    return True
        return False

    def configuration_set_valid(
        self, states: np.ndarray
    ) -> tuple[bool, dict[str, object]]:
        """Validate one simultaneous four-arm configuration without PyBullet."""
        states = np.asarray(states, dtype=np.float64)
        if states.shape != (len(ROBOT_LAYOUT), 14):
            raise ValueError(f"Expected shape (4, 14), got {states.shape}")
        spheres = []
        static_valid = []
        for index, state in enumerate(states):
            q = np.clip(state[:7], Q_LOWER, Q_UPPER)
            mask, _ = self.validate_path(index, q[None, :], start_time=0.0, step_time=0.02)
            static_valid.append(bool(mask[0]) and self.agents[index].is_state_within_limits(state))
            spheres.append(self.configuration_world_spheres(index, q)[0])
        colliding_pairs = []
        for first in range(len(spheres)):
            for second in range(first + 1, len(spheres)):
                collision = self._sphere_overlap_mask(
                    spheres[first][None, ...],
                    spheres[second][None, ...],
                    pair_counter="endpoint",
                )
                if bool(collision.item()):
                    colliding_pairs.append([ROBOT_NAMES[first], ROBOT_NAMES[second]])
        return bool(all(static_valid) and not colliding_pairs), {
            "per_robot_self_static_valid": static_valid,
            "colliding_robot_pairs": colliding_pairs,
        }

    def joint_path_first_collision(
        self, joint_path: np.ndarray, start_index: int = 0
    ) -> tuple[bool, int, int, int]:
        """Return the first exact inter-arm collision in a flattened joint path.

        The expected path layout is ``[robot0_state14, ..., robot3_state14]``.
        Static table, fixture, self-collision, and limits remain the per-agent
        validity callbacks' responsibility; this method tests only inter-arm
        295-sphere overlap in the shared world frame.
        """
        path = np.asarray(joint_path, dtype=np.float64)
        expected_width = len(ROBOT_LAYOUT) * 14
        if path.ndim != 2 or path.shape[1] != expected_width:
            raise ValueError(
                f"Expected joint path shape (N, {expected_width}), got {path.shape}")
        first = max(0, int(start_index))
        if first >= len(path):
            return False, -1, -1, -1
        world_paths = [
            self.configuration_world_spheres(
                robot_index,
                path[first:, robot_index * 14 : robot_index * 14 + 7],
            )
            for robot_index in range(len(ROBOT_LAYOUT))
        ]
        best = None
        for first_robot in range(len(world_paths)):
            for second_robot in range(first_robot + 1, len(world_paths)):
                collision = self._sphere_overlap_mask(
                    world_paths[first_robot],
                    world_paths[second_robot],
                    pair_counter="joint_path",
                )
                hits = self.torch.nonzero(collision, as_tuple=False).view(-1)
                if hits.numel():
                    hit = first + int(hits[0].item())
                    candidate = (hit, first_robot, second_robot)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            return False, -1, -1, -1
        hit, first_robot, second_robot = best
        return True, first_robot, second_robot, hit

    def joint_path_collides(self, joint_path: np.ndarray, start_index: int = 0) -> bool:
        """Return whether a flattened simultaneous four-arm edge collides."""
        return bool(self.joint_path_first_collision(joint_path, start_index)[0])

    def first_synchronized_path_conflict(
        self, paths: list[np.ndarray], integration_dt: float
    ) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray]:
        """Return the first exact KCBS conflict interval between saved paths."""
        if len(paths) != len(ROBOT_LAYOUT):
            raise ValueError(f"Expected four paths, received {len(paths)}")
        arrays = [np.asarray(path, dtype=np.float64) for path in paths]
        if any(path.ndim != 2 or path.shape[1] != 14 or len(path) == 0 for path in arrays):
            raise ValueError("Every KCBS path must be a non-empty (N, 14) array")
        max_steps = max(len(path) for path in arrays)
        aligned = [
            path[np.minimum(np.arange(max_steps), len(path) - 1)] for path in arrays
        ]
        world = [
            self.configuration_world_spheres(index, path[:, :7])
            for index, path in enumerate(aligned)
        ]
        best = None
        for first_robot in range(len(world)):
            for second_robot in range(first_robot + 1, len(world)):
                collision = self._sphere_overlap_mask(
                    world[first_robot], world[second_robot], pair_counter="kcbs")
                hits = self.torch.nonzero(collision, as_tuple=False).view(-1)
                if hits.numel():
                    hit = int(hits[0].item())
                    candidate = (hit, first_robot, second_robot)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            return (
                np.empty(0, dtype=np.float64),
                -1,
                np.empty((0, 14), dtype=np.float64),
                -1,
                np.empty((0, 14), dtype=np.float64),
            )
        hit, first_robot, second_robot = best
        key = np.asarray([(hit + 1) * float(integration_dt)], dtype=np.float64)
        return (
            key,
            first_robot,
            aligned[first_robot][hit : hit + 1].copy(),
            second_robot,
            aligned[second_robot][hit : hit + 1].copy(),
        )

    def constraint_path_valid(
        self,
        robot_index: int,
        path: np.ndarray,
        constraints,
        *,
        start_time: float,
        step_time: float,
    ) -> bool:
        """Check a low-level KCBS edge against exact time-indexed arm constraints."""
        states = np.asarray(path, dtype=np.float64)
        if not constraints or len(states) == 0:
            return True
        times = start_time + step_time * np.arange(1, len(states) + 1)
        for keys, obstacle_states, obstacle_robot_value in constraints:
            keys = np.asarray(keys, dtype=np.float64)
            obstacle_states = np.asarray(obstacle_states, dtype=np.float64)
            if len(keys) == 0:
                continue
            obstacle_robot = int(round(float(obstacle_robot_value)))
            # Conflict keys are grid-aligned.  Use half a step of tolerance to
            # remain robust to floating-point roundoff in accumulated RRT time.
            matches = np.abs(times[:, None] - keys[None, :]) <= (0.51 * step_time)
            path_rows, constraint_rows = np.nonzero(matches)
            if len(path_rows) == 0:
                continue
            current = self.configuration_world_spheres(
                robot_index, states[path_rows, :7])
            obstacle = self.configuration_world_spheres(
                obstacle_robot, obstacle_states[constraint_rows, :7])
            if bool(self.torch.any(self._sphere_overlap_mask(
                current, obstacle, pair_counter="kcbs_constraint"
            )).item()):
                return False
        return True

    def constraint_parking_blocked(
        self,
        robot_index: int,
        state: np.ndarray,
        constraints,
        *,
        arrival_time: float,
        step_time: float,
    ) -> bool:
        """Check whether holding a KCBS arm at its goal violates a future constraint."""
        if not constraints:
            return False
        held = None
        for keys, obstacle_states, obstacle_robot_value in constraints:
            keys = np.asarray(keys, dtype=np.float64)
            obstacle_states = np.asarray(obstacle_states, dtype=np.float64)
            future = np.flatnonzero(keys >= arrival_time - 0.51 * step_time)
            if future.size == 0:
                continue
            obstacle_robot = int(round(float(obstacle_robot_value)))
            if held is None:
                held = self.configuration_world_spheres(
                    robot_index, np.asarray(state, dtype=np.float64)[:7]
                )[:1]
            obstacle = self.configuration_world_spheres(
                obstacle_robot, obstacle_states[future, :7])
            current = held.expand(obstacle.shape[0], -1, -1)
            if bool(self.torch.any(self._sphere_overlap_mask(
                current, obstacle, pair_counter="kcbs_parking"
            )).item()):
                return True
        return False

    @staticmethod
    def _numeric_profile(profile: dict[str, object]) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for key, value in profile.items():
            if isinstance(value, (bool, np.bool_)):
                result[key] = int(value)
            elif isinstance(value, (int, np.integer)):
                result[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                result[key] = float(value)
        return result

    @staticmethod
    def _delta(after: dict[str, object], before: dict[str, object]) -> dict[str, float | int]:
        delta: dict[str, float | int] = {}
        for key, after_value in after.items():
            before_value = before.get(key, 0)
            value = after_value - before_value
            delta[key] = float(value) if isinstance(value, float) else int(value)
        return delta

    def prepare_planner(self, planner, planner_index: int) -> None:
        robot_index = int(planner.agent.robot_index)
        planner.dynamic_col_checker_to_end = (
            lambda state, _radius, _obstacles, _clearance, arrival, dt,
            current=robot_index: self.parking_blocked(
                current, np.asarray(state), float(arrival), float(dt)
            )
        )
        generator = getattr(planner, "flow_edge_generator", None)
        self._stage_before[planner_index] = {
            "robot_stats": dict(self.robot_stats[robot_index]),
            "generator_profile": self._numeric_profile(
                getattr(generator, "profile", {})
            ),
        }

    def finish_planner(
        self,
        planner,
        planner_index: int,
        dense_path: np.ndarray | None,
        *,
        success: bool,
    ) -> None:
        robot_index = int(planner.agent.robot_index)
        before = self._stage_before[planner_index]
        generator = getattr(planner, "flow_edge_generator", None)
        generator_profile = self._numeric_profile(getattr(generator, "profile", {}))
        planner_profile = self._numeric_profile(getattr(planner, "profile", {}))
        record: dict[str, object] = {
            "generation": int(planner_index),
            "robot_index": robot_index,
            "robot_name": ROBOT_NAMES[robot_index],
            "success": bool(success),
            "planning_time_seconds": float(planner.last_plan_wall_time),
            "iterations": int(planner.last_plan_iterations),
            "nodes": int(planner.num_rrt_nodes()),
            "path_motion_time_seconds": float(planner.path_time) if success else None,
            "path_cost": float(planner.path_cost) if success else None,
            "checked_waypoints": int(planner.agent.checked_waypoints),
            "limit_rejections": int(planner.agent.limit_rejections),
            "self_or_static_collision_rejections": int(
                planner.agent.self_collision_rejections
            ),
            "inter_robot_collision_rejections": int(
                planner.agent.inter_robot_rejections
            ),
            "collision_profile": self._delta(
                self.robot_stats[robot_index], before["robot_stats"]
            ),
            "planner_profile": planner_profile,
            "flow_generator_profile": self._delta(
                generator_profile, before["generator_profile"]
            ),
        }
        self.stage_records.append(record)
        if success and dense_path is not None:
            q = np.asarray(dense_path, dtype=np.float64)[:, :7]
            world_spheres = self.configuration_world_spheres(robot_index, q)
            self.trajectories.append(
                CachedSphereTrajectory(
                    robot_index=robot_index,
                    spheres_world=world_spheres.detach(),
                    integration_dt=float(planner.minimum_time_step),
                )
            )
