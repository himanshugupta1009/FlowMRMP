"""Franka Panda adapter for the vanilla kinodynamic RRT.

The state is ``[q(7), dq(7)]`` and the control is a constant joint
acceleration ``ddq(7)``.  Limits match cuRobo's
``franka_no_attachment.yml``.  Rollout validity checks every recorded
waypoint for joint limits, velocity limits, and MorphIt-sphere self-collision.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .franka_redirect_stream import RedirectStream
except ImportError:
    from franka_redirect_stream import RedirectStream


Q_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
Q_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)
DQ_MAX = np.array(
    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    dtype=np.float64,
)
DDQ_MAX = np.full(7, 15.0, dtype=np.float64)
DDDQ_MAX = np.full(7, 500.0, dtype=np.float64)
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
DEFAULT_CUROBO_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "robots"
    / "panda"
    / "curobo"
    / "panda_morphit_d10_c1000_p1000.yml"
)


class EmptyFrankaEnvironment:
    """Compatibility environment for an empty-world single-arm RRT."""

    def __init__(self):
        self.size = np.ones(14, dtype=np.float64)
        self.env_start = -np.ones(14, dtype=np.float64)
        self.obstacle_buffer = 0.0
        self.boundary_buffer = 0.0
        self.static_circular_obstacles = np.empty((0, 3), dtype=np.float64)
        self.static_rectangular_obstacles = np.empty((0, 4), dtype=np.float64)


class FrankaPyBulletCollisionChecker:
    """Exact-mesh PyBullet checker retained for visualization and reference tests."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        visualize: bool = False,
        load_visuals: bool | None = None,
        suppress_output: bool = True,
    ):
        try:
            import pybullet as pb
            from pybullet_utils.bullet_client import BulletClient
        except ImportError as exc:
            raise RuntimeError(
                "PyBullet is required for Franka self-collision checking."
            ) from exc

        self._pb = pb
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)

        mode = pb.GUI if visualize else pb.DIRECT
        if load_visuals is None:
            load_visuals = visualize
        flags = pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        if not load_visuals:
            flags |= pb.URDF_IGNORE_VISUAL_SHAPES
        # PyBullet writes URDF diagnostics through native stdout/stderr rather
        # than Python logging. Keep those diagnostics out of benchmark output
        # using the repository-local stream redirector.
        if suppress_output:
            with RedirectStream(sys.stdout), RedirectStream(sys.stderr):
                self.client = BulletClient(connection_mode=mode)
                self.body_id = self.client.loadURDF(
                    str(self.urdf_path), useFixedBase=True, flags=flags
                )
        else:
            self.client = BulletClient(connection_mode=mode)
            self.body_id = self.client.loadURDF(
                str(self.urdf_path), useFixedBase=True, flags=flags
            )

        joint_info = [
            self.client.getJointInfo(self.body_id, index)
            for index in range(self.client.getNumJoints(self.body_id))
        ]
        self.joint_indices = {
            self._decode(info[1]): int(info[0]) for info in joint_info
        }
        self.link_indices = {
            self._decode(info[12]): int(info[0]) for info in joint_info
        }
        # The root link does not appear as a child link in getJointInfo.
        root_link = ET.parse(self.urdf_path).getroot().find("link")
        if root_link is not None:
            self.link_indices[root_link.attrib["name"]] = -1

        missing = [name for name in JOINT_NAMES if name not in self.joint_indices]
        if missing:
            raise ValueError(f"Panda URDF is missing joints: {missing}")
        self.arm_joint_indices = np.array(
            [self.joint_indices[name] for name in JOINT_NAMES], dtype=np.int32
        )
        self._apply_srdf_exclusions()

    @staticmethod
    def _decode(value):
        return value.decode() if isinstance(value, bytes) else value

    def _apply_srdf_exclusions(self) -> None:
        srdf_files = sorted(self.urdf_path.parent.glob("*.srdf"))
        if not srdf_files:
            raise FileNotFoundError(
                f"No SRDF found beside Panda URDF: {self.urdf_path}"
            )
        root = ET.parse(srdf_files[0]).getroot()
        for disabled in root.findall("disable_collisions"):
            link1 = disabled.attrib["link1"]
            link2 = disabled.attrib["link2"]
            if link1 not in self.link_indices or link2 not in self.link_indices:
                continue
            self.client.setCollisionFilterPair(
                self.body_id,
                self.body_id,
                self.link_indices[link1],
                self.link_indices[link2],
                0,
            )

    def set_configuration(self, q: np.ndarray) -> None:
        values = [[float(value)] for value in np.asarray(q, dtype=np.float64)]
        self.client.resetJointStatesMultiDof(
            self.body_id,
            self.arm_joint_indices.tolist(),
            values,
            [[0.0]] * len(values),
        )

    def in_collision(self, q: np.ndarray) -> bool:
        self.set_configuration(np.asarray(q, dtype=np.float64))
        self.client.performCollisionDetection()
        contacts = self.client.getContactPoints(self.body_id, self.body_id)
        return any(point[8] < 0.0 for point in contacts)

    def close(self) -> None:
        if getattr(self, "client", None) is not None:
            self.client.disconnect()
            self.client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FrankaCuroboCollisionChecker:
    """GPU collision checker backed by a MorphIt sphere robot model."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CUROBO_CONFIG,
        *,
        scene_model=None,
        device: str = "cuda:0",
    ):
        try:
            import torch
            import yaml

            from curobo.collision_checking import (
                RobotCollisionChecker,
                RobotCollisionCheckerCfg,
            )
            from curobo._src.types.device_cfg import DeviceCfg
        except ImportError as exc:
            raise RuntimeError(
                "cuRobo and its CUDA dependencies are required for Franka "
                "collision checking. Use the dedicated cuRobo environment."
            ) from exc

        requested_path = Path(config_path).resolve()
        # Preserve the old constructor call used throughout the repository:
        # callers historically supplied the URDF even though the active
        # backend now loads the generated cuRobo configuration.
        self.urdf_path = requested_path if requested_path.suffix.lower() == ".urdf" else None
        self.config_path = (
            DEFAULT_CUROBO_CONFIG.resolve()
            if self.urdf_path is not None
            else requested_path
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo collision checking requires CUDA")

        config_data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._torch = torch
        self.device_cfg = DeviceCfg(device=torch.device(device))
        checker_cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=config_data,
            scene_model=scene_model,
            device_cfg=self.device_cfg,
            collision_activation_distance=0.0,
            self_collision_activation_distance=0.0,
        )
        self.checker = RobotCollisionChecker(checker_cfg)
        actual_joint_names = tuple(self.checker.kinematics.all_articulated_joint_names)
        if actual_joint_names != JOINT_NAMES:
            raise ValueError(
                f"cuRobo joint order {actual_joint_names} does not match {JOINT_NAMES}"
            )

    def collision_free_mask(self, q: np.ndarray) -> np.ndarray:
        """Return a Boolean validity mask for joint positions shaped ``(N, 7)``."""
        configurations = np.asarray(q, dtype=np.float32)
        if configurations.ndim == 1:
            configurations = configurations[None, :]
        if configurations.ndim != 2 or configurations.shape[1] != 7:
            raise ValueError(f"Expected joint positions shaped (N, 7), got {configurations.shape}")
        joint_tensor = self._torch.as_tensor(
            configurations,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        ).unsqueeze(0)
        with self._torch.inference_mode():
            valid = self.checker.validate(joint_tensor)
        return valid.squeeze(0).detach().cpu().numpy().astype(bool, copy=False)

    def in_collision(self, q: np.ndarray) -> bool:
        return not bool(self.collision_free_mask(q)[0])

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self.device_cfg.device)

    def close(self) -> None:
        """Match the PyBullet checker lifecycle API; cuRobo owns no client handle."""


FrankaSelfCollisionChecker = FrankaCuroboCollisionChecker


class FrankaPanda:
    """14D Franka state / 7D acceleration agent for vanilla RRT."""

    def __init__(
        self,
        *,
        state_bank: np.ndarray,
        collision_checker: FrankaSelfCollisionChecker,
        agent_id: int = 0,
        acceleration_scale: float = 1.0,
    ):
        states = np.asarray(state_bank, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 14 or states.shape[0] == 0:
            raise ValueError("state_bank must have shape (N, 14) with N > 0")
        if not 0.0 < acceleration_scale <= 1.0:
            raise ValueError("acceleration_scale must be in (0, 1]")

        self.state_bank = states
        self.collision_checker = collision_checker
        self.acceleration_scale = float(acceleration_scale)
        self.id = agent_id
        self.radius = 0.0
        self.state_length = 14
        self.action_length = 7
        self.distance_metric_state_size = 14
        self.disable_dynamic_collision_check = True
        # These fields are required by the shared RRT validity callback API.
        self.dynamic_limit_indices = np.arange(14, dtype=np.int64)
        self.dynamic_limit_values = np.ones(14, dtype=np.float64)
        self.checked_waypoints = 0
        self.limit_rejections = 0
        self.acceleration_rejections = 0
        self.jerk_rejections = 0
        self.self_collision_rejections = 0

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        q_norm = 2.0 * (state[..., :7] - Q_LOWER) / (Q_UPPER - Q_LOWER) - 1.0
        dq_norm = state[..., 7:] / DQ_MAX
        return np.concatenate((q_norm, dq_norm), axis=-1)

    @staticmethod
    def normalize_configuration(state: np.ndarray) -> np.ndarray:
        """Return only the seven normalized joint positions."""
        state = np.asarray(state, dtype=np.float64)
        return 2.0 * (state[..., :7] - Q_LOWER) / (Q_UPPER - Q_LOWER) - 1.0

    def get_distance_metric_state(self, state: np.ndarray) -> np.ndarray:
        return self.normalize_state(state)

    def get_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        delta = self.normalize_state(state1) - self.normalize_state(state2)
        return float(np.linalg.norm(delta))

    def get_parent_selection_metric_dims(self, goal_biased: bool) -> int:
        """Return the Franka parent metric dimension for the current query.

        A goal-biased query uses only normalized joint position because task
        completion ignores goal velocity. Ordinary state-space exploration
        remains kinodynamic and therefore uses the complete ``[q, dq]`` state.
        """
        return 7 if goal_biased else self.distance_metric_state_size

    def get_extension_score(
        self,
        state: np.ndarray,
        target: np.ndarray,
        goal_biased: bool,
    ) -> float:
        """Score an actual VanillaRRT rollout under the shared Franka policy."""
        if goal_biased:
            delta = self.normalize_configuration(state) - self.normalize_configuration(
                target
            )
            return float(np.linalg.norm(delta))
        return self.get_distance(state, target)

    def get_flow_edge_ranking_distance(
        self,
        predicted_state: np.ndarray,
        target: np.ndarray,
        goal_biased: bool,
    ) -> float:
        """Score a predicted Flow edge endpoint using joint position only.

        Position-only Flow-edge ranking is the previously validated Franka
        policy for both goal and ordinary samples. ``goal_biased`` is accepted
        explicitly so future policy changes remain centralized in this agent.
        """
        del goal_biased
        delta = self.normalize_configuration(
            predicted_state
        ) - self.normalize_configuration(target)
        return float(np.linalg.norm(delta))

    @staticmethod
    def get_goal_distance(state: np.ndarray, goal: np.ndarray) -> float:
        """Normalized 7D joint-position distance used only for goal tests."""
        delta = (
            FrankaPanda.normalize_configuration(state)
            - FrankaPanda.normalize_configuration(goal)
        )
        return float(np.linalg.norm(delta))

    def get_random_action(self, rng: np.random.Generator) -> np.ndarray:
        limit = DDQ_MAX * self.acceleration_scale
        return rng.uniform(-limit, limit)

    def get_next_state(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
        num_steps: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate a constant acceleration exactly at uniform substeps."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        state = np.asarray(state, dtype=np.float64)
        acceleration = np.asarray(control, dtype=np.float64)
        if acceleration.shape != (7,):
            raise ValueError(
                f"control must have shape (7,), received {acceleration.shape}"
            )
        if not self.is_acceleration_within_limits(acceleration):
            raise ValueError("control is non-finite or exceeds Franka acceleration limits")
        step_dt = float(dt) / num_steps
        path = np.empty((num_steps, 14), dtype=np.float64)
        current = state.copy()
        for index in range(num_steps):
            q = current[:7]
            dq = current[7:]
            next_state = np.empty(14, dtype=np.float64)
            next_state[:7] = q + dq * step_dt + 0.5 * acceleration * step_dt**2
            next_state[7:] = dq + acceleration * step_dt
            path[index] = next_state
            current = next_state
        return current, path

    def get_next_state_sequence(
        self,
        state: np.ndarray,
        accelerations: np.ndarray,
        dt: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out one piecewise-constant 7D acceleration per time interval."""
        state = np.asarray(state, dtype=np.float64)
        accelerations = np.asarray(accelerations, dtype=np.float64)
        if state.shape != (14,):
            raise ValueError(f"state must have shape (14,), received {state.shape}")
        if accelerations.ndim != 2 or accelerations.shape[1] != 7:
            raise ValueError(
                "accelerations must have shape (number_of_intervals, 7), "
                f"received {accelerations.shape}"
            )
        if accelerations.shape[0] == 0:
            raise ValueError("an acceleration sequence must contain at least one interval")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if not self.is_action_sequence_within_limits(accelerations, dt):
            raise ValueError(
                "acceleration sequence is non-finite or exceeds Franka "
                "acceleration/intra-edge jerk limits"
            )
        path = np.empty((accelerations.shape[0], 14), dtype=np.float64)
        current = state.copy()
        step_dt = float(dt)
        for index, acceleration in enumerate(accelerations):
            q = current[:7]
            dq = current[7:]
            next_state = np.empty(14, dtype=np.float64)
            next_state[:7] = q + dq * step_dt + 0.5 * acceleration * step_dt**2
            next_state[7:] = dq + acceleration * step_dt
            path[index] = next_state
            current = next_state
        return current, path

    @staticmethod
    def is_acceleration_within_limits(acceleration: np.ndarray) -> bool:
        acceleration = np.asarray(acceleration, dtype=np.float64)
        return bool(
            acceleration.shape == (7,)
            and np.isfinite(acceleration).all()
            and np.all(np.abs(acceleration) <= DDQ_MAX + 1e-10)
        )

    @staticmethod
    def is_action_sequence_within_limits(
        accelerations: np.ndarray,
        dt: float,
    ) -> bool:
        """Check finite acceleration and intra-edge jerk limits.

        Boundary jerk between two RRT edges is intentionally not checked here
        because acceleration is not part of the planner's 14D state.
        """
        return FrankaPanda.action_sequence_violation(accelerations, dt) is None

    @staticmethod
    def action_sequence_violation(
        accelerations: np.ndarray,
        dt: float,
    ) -> str | None:
        """Return ``acceleration``, ``jerk``, or ``None`` for a sequence."""
        accelerations = np.asarray(accelerations, dtype=np.float64)
        if (
            accelerations.ndim != 2
            or accelerations.shape[0] == 0
            or accelerations.shape[1] != 7
            or not np.isfinite(accelerations).all()
            or not np.isfinite(dt)
            or dt <= 0.0
        ):
            return "acceleration"
        if np.any(np.abs(accelerations) > DDQ_MAX[None, :] + 1e-10):
            return "acceleration"
        if len(accelerations) > 1:
            jerk = np.diff(accelerations, axis=0) / float(dt)
            if np.any(np.abs(jerk) > DDDQ_MAX[None, :] + 1e-10):
                return "jerk"
        return None

    def is_state_within_limits(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float64)
        q = state[:7]
        dq = state[7:]
        return bool(
            np.all(q >= Q_LOWER)
            and np.all(q <= Q_UPPER)
            and np.all(np.abs(dq) <= DQ_MAX)
        )

    def is_state_valid(self, state: np.ndarray) -> bool:
        return self.is_state_within_limits(state) and not self.collision_checker.in_collision(
            state[:7]
        )

    def is_new_node_valid(self, path_to_new_state: np.ndarray, *args) -> bool:
        # The parent state was already validated.  Every newly integrated
        # waypoint is checked, not just the final state.
        states = np.asarray(path_to_new_state, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 14:
            raise ValueError(f"Expected a waypoint path shaped (N, 14), got {states.shape}")

        collision_free_mask = getattr(self.collision_checker, "collision_free_mask", None)
        if collision_free_mask is not None and len(states):
            limit_valid = np.array(
                [self.is_state_within_limits(state) for state in states], dtype=bool
            )
            # Collision results for strict-limit-invalid states do not affect
            # validity. Clipping keeps the GPU kinematics query well-defined
            # while preserving the first-invalid-waypoint accounting below.
            q_for_collision = np.clip(states[:, :7], Q_LOWER, Q_UPPER)
            collision_valid = collision_free_mask(q_for_collision)
            combined_valid = limit_valid & collision_valid
            invalid_indices = np.flatnonzero(~combined_valid)
            if invalid_indices.size:
                first_invalid = int(invalid_indices[0])
                self.checked_waypoints += first_invalid + 1
                if not limit_valid[first_invalid]:
                    self.limit_rejections += 1
                else:
                    self.self_collision_rejections += 1
                return False
            self.checked_waypoints += len(states)
            return True

        for state in path_to_new_state:
            self.checked_waypoints += 1
            if not self.is_state_within_limits(state):
                self.limit_rejections += 1
                return False
            if self.collision_checker.in_collision(state[:7]):
                self.self_collision_rejections += 1
                return False
        return True

    @staticmethod
    def get_cost(env, agent, parent_state, action, duration, path) -> float:
        return float(duration)

    def get_random_point(
        self, env, circular_obstacles, rectangular_obstacles, rng
    ) -> np.ndarray:
        index = int(rng.integers(self.state_bank.shape[0]))
        return self.state_bank[index].copy()

    @staticmethod
    def agent_reached_goal(
        state: np.ndarray, goal: np.ndarray, goal_radius: float, agent: "FrankaPanda"
    ) -> tuple[bool, float]:
        state = np.asarray(state, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if state.shape != (14,) or goal.shape != (14,):
            raise ValueError(
                f"Franka goal checks require two 14D states, got {state.shape} and "
                f"{goal.shape}"
            )
        if goal_radius < 0.0:
            raise ValueError("goal_radius must be nonnegative")
        if not np.isfinite(state).all() or not np.isfinite(goal).all():
            return False, float("inf")
        # Planning and nearest-neighbor selection remain kinodynamic (14D),
        # while task completion depends only on the seven joint positions.
        distance = agent.get_goal_distance(state, goal)
        return bool(distance <= goal_radius), distance

    @staticmethod
    def kd_tree_point_translate_function(
        base_point: np.ndarray,
        edge_start_point: np.ndarray,
        edge_end_point: np.ndarray,
    ) -> np.ndarray:
        """Generated Franka endpoints are already absolute 14D states."""
        del base_point, edge_start_point
        return np.asarray(edge_end_point, dtype=np.float64)

    @staticmethod
    def sort_kd_tree_edges(
        closest_tree_point: np.ndarray,
        random_point: np.ndarray,
        start_states: np.ndarray,
        final_states: np.ndarray,
        curr_edge_indices: np.ndarray,
        curr_edge_mask: np.ndarray,
        distance_array: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Sort untried generated edges by normalized 7D position distance.

        Target velocity is intentionally ignored because Franka task success
        is defined by joint configuration, while velocity is still enforced
        by propagation, state validity, and kinodynamic nearest-node queries.
        """
        del closest_tree_point, start_states
        count = len(curr_edge_indices)
        valid = 0
        target = FrankaPanda.normalize_configuration(random_point)
        for local_index in range(count):
            if curr_edge_mask[local_index]:
                distance_array[local_index] = np.inf
                continue
            edge_index = int(curr_edge_indices[local_index])
            endpoint = FrankaPanda.normalize_configuration(final_states[edge_index])
            distance_array[local_index] = np.linalg.norm(endpoint - target)
            valid += 1
        order = np.argsort(distance_array[:count])
        return order[:valid], valid
