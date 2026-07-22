"""Franka Panda adapter for the vanilla kinodynamic RRT.

The state is ``[q(7), dq(7)]`` and the control is a constant joint
acceleration ``ddq(7)``.  Limits match CuRobo's
``franka_no_attachment.yml``.  Rollout validity checks every recorded
waypoint for joint limits, velocity limits, and robot self-collision.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


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
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


class EmptyFrankaEnvironment:
    """Compatibility environment for an empty-world single-arm RRT."""

    def __init__(self):
        self.size = np.ones(14, dtype=np.float64)
        self.env_start = -np.ones(14, dtype=np.float64)
        self.obstacle_buffer = 0.0
        self.boundary_buffer = 0.0
        self.static_circular_obstacles = np.empty((0, 3), dtype=np.float64)
        self.static_rectangular_obstacles = np.empty((0, 4), dtype=np.float64)


class FrankaSelfCollisionChecker:
    """PyBullet self-collision checker honoring the Panda SRDF exclusions."""

    def __init__(self, urdf_path: str | Path, *, visualize: bool = False):
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
        try:
            from diffusion_planner.pybullet.redirect_stream import RedirectStream
        except ImportError:
            RedirectStream = None

        flags = pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        if not visualize:
            flags |= pb.URDF_IGNORE_VISUAL_SHAPES
        if RedirectStream is None:
            self.client = BulletClient(connection_mode=mode)
            self.body_id = self.client.loadURDF(
                str(self.urdf_path), useFixedBase=True, flags=flags
            )
        else:
            with RedirectStream(sys.stdout), RedirectStream(sys.stderr):
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
        for joint_index, value in zip(self.arm_joint_indices, q):
            self.client.resetJointState(
                self.body_id, int(joint_index), float(value), targetVelocity=0.0
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
        self.self_collision_rejections = 0

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        q_norm = 2.0 * (state[..., :7] - Q_LOWER) / (Q_UPPER - Q_LOWER) - 1.0
        dq_norm = state[..., 7:] / DQ_MAX
        return np.concatenate((q_norm, dq_norm), axis=-1)

    def get_distance_metric_state(self, state: np.ndarray) -> np.ndarray:
        return self.normalize_state(state)

    def get_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        delta = self.normalize_state(state1) - self.normalize_state(state2)
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
        distance = agent.get_distance(state, goal)
        return distance <= goal_radius, distance
