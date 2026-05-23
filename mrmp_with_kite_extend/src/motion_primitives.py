from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import msgpack
import numpy as np
import yaml
from numba import njit

from kd_tree_unicycle import CircularAngleIndexNumba
from kd_tree_quadcopter6d import VxyzTree


@dataclass
class MotionPrimitiveBundle:
    start_states: np.ndarray
    final_states: np.ndarray
    trajectories: np.ndarray
    trajectory_lengths: np.ndarray
    actions: np.ndarray
    action_lengths: np.ndarray
    timesteps: np.ndarray
    num_edges: int
    dt: float
    representative_actions: Optional[np.ndarray] = None


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@njit
def transform_unicycle_trajectory_numba(base_state, edge_start, raw_traj):
    """
    Fast batch transform for SE(2)-style unicycle primitives.

    base_state: (x_c, y_c, theta_c)
    edge_start: canonical primitive start, typically (0, 0, theta_e)
    raw_traj:   primitive trajectory in canonical frame, shape (T, 3)
    """
    out = np.empty_like(raw_traj)
    x_c = base_state[0]
    y_c = base_state[1]
    theta_c = base_state[2]
    theta_e = edge_start[2]
    delta_theta = theta_c - theta_e
    cos_d = np.cos(delta_theta)
    sin_d = np.sin(delta_theta)

    for k in range(raw_traj.shape[0]):
        x_f = raw_traj[k, 0]
        y_f = raw_traj[k, 1]
        theta_f = raw_traj[k, 2]

        dx_world = x_f * cos_d - y_f * sin_d
        dy_world = x_f * sin_d + y_f * cos_d

        out[k, 0] = x_c + dx_world
        out[k, 1] = y_c + dy_world
        out[k, 2] = theta_c + (theta_f - theta_e)

    return out


@njit
def transform_quadcopter6d_trajectory_numba(base_state, edge_start, raw_traj):
    """
    Translate a canonical quadcopter primitive to a live base state.

    The primitive library is translationally invariant in position. We reuse the
    stored velocities from the primitive and shift only the position component.
    """
    out = np.empty_like(raw_traj)
    dx = base_state[0] - edge_start[0]
    dy = base_state[1] - edge_start[1]
    dz = base_state[2] - edge_start[2]

    for k in range(raw_traj.shape[0]):
        out[k, 0] = raw_traj[k, 0] + dx
        out[k, 1] = raw_traj[k, 1] + dy
        out[k, 2] = raw_traj[k, 2] + dz
        out[k, 3] = raw_traj[k, 3]
        out[k, 4] = raw_traj[k, 4]
        out[k, 5] = raw_traj[k, 5]

    return out


def load_unicycle_motion_primitives(*, num_edges=15000, dt=0.1, repo_root: Optional[str] = None):
    root = _repo_root() if repo_root is None else repo_root
    primitive_file = os.path.join(
        root,
        "motion_primitives",
        "unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack",
    )
    if not os.path.exists(primitive_file):
        raise FileNotFoundError(f"Missing primitive file: {primitive_file}")

    with open(primitive_file, "rb") as f:
        packed = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

    primitives = packed["data"] if isinstance(packed, dict) and "data" in packed else packed
    primitives = primitives[:num_edges]
    if len(primitives) == 0:
        raise ValueError("primitive file loaded, but contains zero primitives")

    lengths = np.array([len(p["states"]) for p in primitives], dtype=np.int64)
    action_lengths = np.array([len(p["actions"]) for p in primitives], dtype=np.int64)
    max_len = int(lengths.max())
    max_action_len = int(action_lengths.max())
    n = len(primitives)

    start_states = np.empty((n, 3), dtype=np.float64)
    final_states = np.empty((n, 3), dtype=np.float64)
    trajectories = np.full((n, max_len, 3), np.nan, dtype=np.float64)
    timesteps = np.empty(n, dtype=np.float64)
    actions = np.full((n, max_action_len, 2), np.nan, dtype=np.float64)
    representative_actions = np.empty((n, 2), dtype=np.float64)

    for i, primitive in enumerate(primitives):
        states = _as_float_array(primitive["states"])
        acts = _as_float_array(primitive["actions"])
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError(f"primitive {i} has states with shape {states.shape}, expected (*, 3)")
        if acts.ndim != 2 or acts.shape[1] != 2:
            raise ValueError(f"primitive {i} has actions with shape {acts.shape}, expected (*, 2)")

        path_len = states.shape[0]
        start_states[i] = states[0]
        final_states[i] = states[-1]
        trajectories[i, :path_len] = states
        lengths[i] = path_len
        timesteps[i] = acts.shape[0] * dt
        if acts.shape[0] > 0:
            actions[i, :acts.shape[0]] = acts
            representative_actions[i] = acts[0]
        else:
            representative_actions[i] = np.zeros(2, dtype=np.float64)

    motion_primitives = MotionPrimitiveBundle(
        start_states=start_states,
        final_states=final_states,
        trajectories=trajectories,
        trajectory_lengths=lengths,
        actions=actions,
        action_lengths=action_lengths,
        representative_actions=representative_actions,
        timesteps=timesteps,
        num_edges=n,
        dt=dt,
    )

    edge_ids = np.arange(motion_primitives.num_edges, dtype=np.int64)
    thetas = motion_primitives.start_states[:, 2]
    kd_tree = CircularAngleIndexNumba(thetas, ids=edge_ids)
    return motion_primitives, kd_tree


def _build_quadcopter6d_motion_primitive_bundle(
    start_states: np.ndarray,
    final_states: np.ndarray,
    trajectories: np.ndarray,
    trajectory_lengths: np.ndarray,
    actions: np.ndarray,
    action_lengths: np.ndarray,
    timesteps: np.ndarray,
    dt: float,
):
    n = int(start_states.shape[0])
    motion_primitives = MotionPrimitiveBundle(
        start_states=np.asarray(start_states, dtype=np.float64),
        final_states=np.asarray(final_states, dtype=np.float64),
        trajectories=np.asarray(trajectories, dtype=np.float64),
        trajectory_lengths=np.asarray(trajectory_lengths, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.float64),
        action_lengths=np.asarray(action_lengths, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.float64),
        num_edges=n,
        dt=float(dt),
    )

    edge_ids = np.arange(motion_primitives.num_edges, dtype=np.int64)
    vx = motion_primitives.start_states[:, 3]
    vy = motion_primitives.start_states[:, 4]
    vz = motion_primitives.start_states[:, 5]
    kd_tree = VxyzTree(vx, vy, vz, ids=edge_ids, scales=(1.0, 1.0, 1.0))
    return motion_primitives, kd_tree


def convert_quadcopter6d_yaml_motion_primitives_to_npz(
    yaml_file_location,
    npz_file_location,
    *,
    num_edges=None,
    dt=0.1,
):
    """
    Convert quadcopter6d YAML motion primitives to the npz schema used by
    load_quadcopter6d_motion_primitives.
    """
    if not os.path.exists(yaml_file_location):
        raise FileNotFoundError(f"Missing primitive file: {yaml_file_location}")

    with open(yaml_file_location, "r", encoding="utf-8") as f:
        primitives = yaml.safe_load(f)

    if num_edges is not None:
        primitives = primitives[: min(int(num_edges), len(primitives))]
    if len(primitives) == 0:
        raise ValueError("primitive file loaded, but contains zero primitives")

    lengths = np.array([len(p["states"]) for p in primitives], dtype=np.int64)
    action_lengths = np.array([len(p["actions"]) for p in primitives], dtype=np.int64)
    max_len = int(lengths.max())
    max_action_len = int(action_lengths.max())
    n = len(primitives)

    start_states = np.empty((n, 6), dtype=np.float64)
    final_states = np.empty((n, 6), dtype=np.float64)
    trajectories = np.full((n, max_len, 6), np.nan, dtype=np.float64)
    actions = np.full((n, max_action_len, 3), np.nan, dtype=np.float64)
    timesteps = np.empty(n, dtype=np.float64)

    for i, primitive in enumerate(primitives):
        states = _as_float_array(primitive["states"])
        acts = _as_float_array(primitive["actions"])
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError(
                f"primitive {i} has states with shape {states.shape}, expected (*, 6)")
        if acts.ndim != 2 or acts.shape[1] != 3:
            raise ValueError(
                f"primitive {i} has actions with shape {acts.shape}, expected (*, 3)")

        path_len = states.shape[0]
        act_len = acts.shape[0]
        start_states[i] = states[0]
        final_states[i] = states[-1]
        trajectories[i, :path_len] = states
        actions[i, :act_len] = acts
        timesteps[i] = float(primitive["cost"])

    np.savez_compressed(
        npz_file_location,
        start_states=start_states,
        final_states=final_states,
        trajectories=trajectories,
        trajectory_lengths=lengths,
        actions=actions,
        action_lengths=action_lengths,
        timesteps=timesteps,
        num_edges=n,
        dt=np.array(dt, dtype=np.float64),
    )


def load_quadcopter6d_motion_primitives(
    *,
    num_edges,
    dt=0.1,
    primitive_file_location="motion_primitives/quadcopter6d_long_50_1000_primitives.npz",
    repo_root: Optional[str] = None,
):
    root = _repo_root() if repo_root is None else repo_root
    primitive_path = (
        "motion_primitives/quadcopter6d_long_50_1000_primitives.npz"
        if primitive_file_location is None
        else primitive_file_location
    )
    if not os.path.isabs(primitive_path):
        primitive_path = os.path.join(root, primitive_path)
    if not os.path.exists(primitive_path):
        raise FileNotFoundError(f"Missing primitive file: {primitive_path}")

    primitives = np.load(primitive_path)
    total_edges = int(primitives["num_edges"])
    n = min(int(num_edges), total_edges)
    if n <= 0:
        raise ValueError("primitive file loaded, but contains zero primitives")
    return _build_quadcopter6d_motion_primitive_bundle(
        primitives["start_states"][:n],
        primitives["final_states"][:n],
        primitives["trajectories"][:n],
        primitives["trajectory_lengths"][:n],
        primitives["actions"][:n],
        primitives["action_lengths"][:n],
        primitives["timesteps"][:n],
        dt,
    )
