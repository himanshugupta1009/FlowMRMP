"""
Optional Crocoddyl-based repair stage for discontinuity-bounded QuadCopter6D paths.

This mirrors the local unicycle repair stage and Dynoplan's fixed-horizon
trajectory-optimization flow at a reduced scope:
    - fixed horizon N from the warm-start trajectory
    - warm-started BoxFDDP solve
    - 3D double-integrator dynamics
    - running obstacle / tracking / control costs
    - terminal goal and terminal velocity cost

The dependency on `crocoddyl` is optional and imported lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from numba import njit


def _flatten_action_sequences(action_sequences: Sequence[np.ndarray]) -> np.ndarray:
    if len(action_sequences) == 0:
        return np.empty((0, 3), dtype=np.float64)
    chunks = [np.asarray(chunk, dtype=np.float64) for chunk in action_sequences if len(chunk) > 0]
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _extract_obstacle_arrays_3d(env):
    spheres_src = getattr(env, "static_circular_obstacles", None)
    boxes_src = getattr(env, "static_rectangular_obstacles", None)

    if spheres_src is None:
        spheres = np.empty((0, 4), dtype=np.float64)
    elif isinstance(spheres_src, np.ndarray):
        spheres = np.asarray(spheres_src, dtype=np.float64).reshape(-1, 4)
    else:
        spheres_src = list(spheres_src)
        spheres = np.empty((len(spheres_src), 4), dtype=np.float64)
        for i, obstacle in enumerate(spheres_src):
            spheres[i, 0] = obstacle.x
            spheres[i, 1] = obstacle.y
            spheres[i, 2] = obstacle.z
            spheres[i, 3] = obstacle.r

    if boxes_src is None:
        boxes = np.empty((0, 6), dtype=np.float64)
    elif isinstance(boxes_src, np.ndarray):
        boxes = np.asarray(boxes_src, dtype=np.float64).reshape(-1, 6)
    else:
        boxes_src = list(boxes_src)
        boxes = np.empty((len(boxes_src), 6), dtype=np.float64)
        for i, obstacle in enumerate(boxes_src):
            half_l = 0.5 * obstacle.l
            half_w = 0.5 * obstacle.w
            half_h = 0.5 * obstacle.h
            boxes[i, 0] = obstacle.x - half_l
            boxes[i, 1] = obstacle.x + half_l
            boxes[i, 2] = obstacle.y - half_w
            boxes[i, 3] = obstacle.y + half_w
            boxes[i, 4] = obstacle.z - half_h
            boxes[i, 5] = obstacle.z + half_h

    return spheres, boxes


@njit(cache=True)
def _signed_distance_to_sphere_numba(x, y, z, cx, cy, cz, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz) - r


@njit(cache=True)
def _signed_distance_to_box_numba(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    hz = 0.5 * (zmax - zmin)

    dx = abs(x - cx) - hx
    dy = abs(y - cy) - hy
    dz = abs(z - cz) - hz

    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    outside_dz = max(dz, 0.0)
    outside_dist = math.sqrt(
        outside_dx * outside_dx + outside_dy * outside_dy + outside_dz * outside_dz
    )
    inside_dist = min(max(dx, max(dy, dz)), 0.0)
    return outside_dist + inside_dist


@njit(cache=True)
def _signed_distance_to_obstacles_3d_numba(x, y, z, spheres, boxes):
    best = 1e6

    for i in range(spheres.shape[0]):
        d = _signed_distance_to_sphere_numba(
            x, y, z, spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
        )
        if d < best:
            best = d

    for i in range(boxes.shape[0]):
        d = _signed_distance_to_box_numba(
            x,
            y,
            z,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            boxes[i, 4],
            boxes[i, 5],
        )
        if d < best:
            best = d

    return best


@njit(cache=True)
def _signed_distance_and_gradient_to_sphere_numba(x, y, z, cx, cy, cz, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= 1e-12:
        return -r, 1.0, 0.0, 0.0
    inv_dist = 1.0 / dist
    return dist - r, dx * inv_dist, dy * inv_dist, dz * inv_dist


@njit(cache=True)
def _signed_distance_and_gradient_to_box_numba(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    closest_x = min(max(x, xmin), xmax)
    closest_y = min(max(y, ymin), ymax)
    closest_z = min(max(z, zmin), zmax)
    dx = x - closest_x
    dy = y - closest_y
    dz = z - closest_z
    outside_dist_sq = dx * dx + dy * dy + dz * dz

    if outside_dist_sq > 1e-24:
        outside_dist = math.sqrt(outside_dist_sq)
        inv_dist = 1.0 / outside_dist
        return outside_dist, dx * inv_dist, dy * inv_dist, dz * inv_dist

    left = x - xmin
    right = xmax - x
    bottom = y - ymin
    top = ymax - y
    down = z - zmin
    up = zmax - z

    best = left
    gx = -1.0
    gy = 0.0
    gz = 0.0
    if right < best:
        best = right
        gx = 1.0
        gy = 0.0
        gz = 0.0
    if bottom < best:
        best = bottom
        gx = 0.0
        gy = -1.0
        gz = 0.0
    if top < best:
        best = top
        gx = 0.0
        gy = 1.0
        gz = 0.0
    if down < best:
        best = down
        gx = 0.0
        gy = 0.0
        gz = -1.0
    if up < best:
        best = up
        gx = 0.0
        gy = 0.0
        gz = 1.0

    return -best, gx, gy, gz


@njit(cache=True)
def _signed_distance_and_gradient_to_obstacles_3d_numba(x, y, z, spheres, boxes):
    best = 1e6
    best_gx = 1.0
    best_gy = 0.0
    best_gz = 0.0

    for i in range(spheres.shape[0]):
        d, gx, gy, gz = _signed_distance_and_gradient_to_sphere_numba(
            x, y, z, spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
        )
        if d < best:
            best = d
            best_gx = gx
            best_gy = gy
            best_gz = gz

    for i in range(boxes.shape[0]):
        d, gx, gy, gz = _signed_distance_and_gradient_to_box_numba(
            x,
            y,
            z,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            boxes[i, 4],
            boxes[i, 5],
        )
        if d < best:
            best = d
            best_gx = gx
            best_gy = gy
            best_gz = gz

    return best, best_gx, best_gy, best_gz


@njit(cache=True)
def _obstacle_penalty_and_derivatives_3d_inplace_finite_diff_numba(
    x,
    spheres,
    boxes,
    agent_radius,
    obstacle_buffer,
    clearance_margin,
    obstacle_weight,
    g,
    H,
):
    clearance = _signed_distance_to_obstacles_3d_numba(x[0], x[1], x[2], spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    deficit = clearance_margin - clearance
    for i in range(6):
        g[i] = 0.0
        for j in range(6):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    eps = 1e-4
    x0 = x[0]
    y0 = x[1]
    z0 = x[2]

    xp0 = _signed_distance_to_obstacles_3d_numba(x0 + eps, y0, z0, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    xm0 = _signed_distance_to_obstacles_3d_numba(x0 - eps, y0, z0, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    grad0 = (xp0 - xm0) / (2.0 * eps)
    hess00 = (xp0 - 2.0 * clearance + xm0) / (eps * eps)

    xp1 = _signed_distance_to_obstacles_3d_numba(x0, y0 + eps, z0, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    xm1 = _signed_distance_to_obstacles_3d_numba(x0, y0 - eps, z0, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    grad1 = (xp1 - xm1) / (2.0 * eps)
    hess11 = (xp1 - 2.0 * clearance + xm1) / (eps * eps)

    xp2 = _signed_distance_to_obstacles_3d_numba(x0, y0, z0 + eps, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    xm2 = _signed_distance_to_obstacles_3d_numba(x0, y0, z0 - eps, spheres, boxes) - (
        agent_radius + obstacle_buffer
    )
    grad2 = (xp2 - xm2) / (2.0 * eps)
    hess22 = (xp2 - 2.0 * clearance + xm2) / (eps * eps)

    cost = 0.5 * obstacle_weight * deficit * deficit
    g[0] = -obstacle_weight * deficit * grad0
    g[1] = -obstacle_weight * deficit * grad1
    g[2] = -obstacle_weight * deficit * grad2

    H[0, 0] = obstacle_weight * grad0 * grad0 - obstacle_weight * deficit * hess00
    H[0, 1] = obstacle_weight * grad0 * grad1
    H[0, 2] = obstacle_weight * grad0 * grad2
    H[1, 0] = H[0, 1]
    H[1, 1] = obstacle_weight * grad1 * grad1 - obstacle_weight * deficit * hess11
    H[1, 2] = obstacle_weight * grad1 * grad2
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    H[2, 2] = obstacle_weight * grad2 * grad2 - obstacle_weight * deficit * hess22
    return cost


@njit(cache=True)
def _obstacle_penalty_and_derivatives_3d_inplace_numba(
    x,
    spheres,
    boxes,
    agent_radius,
    obstacle_buffer,
    clearance_margin,
    obstacle_weight,
    g,
    H,
):
    signed_distance, grad0, grad1, grad2 = _signed_distance_and_gradient_to_obstacles_3d_numba(
        x[0], x[1], x[2], spheres, boxes
    )
    clearance = signed_distance - (agent_radius + obstacle_buffer)
    deficit = clearance_margin - clearance
    for i in range(6):
        g[i] = 0.0
        for j in range(6):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    cost = 0.5 * obstacle_weight * deficit * deficit
    g[0] = -obstacle_weight * deficit * grad0
    g[1] = -obstacle_weight * deficit * grad1
    g[2] = -obstacle_weight * deficit * grad2

    H[0, 0] = obstacle_weight * grad0 * grad0
    H[0, 1] = obstacle_weight * grad0 * grad1
    H[0, 2] = obstacle_weight * grad0 * grad2
    H[1, 0] = H[0, 1]
    H[1, 1] = obstacle_weight * grad1 * grad1
    H[1, 2] = obstacle_weight * grad1 * grad2
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    H[2, 2] = obstacle_weight * grad2 * grad2
    return cost


@njit(cache=True)
def _boundary_penalty_and_derivatives_3d_inplace_numba(
    x,
    env_size,
    agent_radius,
    boundary_buffer,
    clearance_margin,
    boundary_weight,
    g,
    H,
):
    lower = agent_radius + boundary_buffer

    clearance = x[0] - lower
    axis = 0
    sign = 1.0
    for i in range(3):
        low_clearance = x[i] - lower
        high_clearance = env_size[i] - lower - x[i]
        if low_clearance < clearance:
            clearance = low_clearance
            axis = i
            sign = 1.0
        if high_clearance < clearance:
            clearance = high_clearance
            axis = i
            sign = -1.0

    deficit = clearance_margin - clearance
    for i in range(6):
        g[i] = 0.0
        for j in range(6):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    cost = 0.5 * boundary_weight * deficit * deficit
    g[axis] = -boundary_weight * deficit * sign
    H[axis, axis] = boundary_weight
    return cost


@njit(cache=True)
def _velocity_limit_penalty_and_derivatives_3d_inplace_numba(
    x,
    max_speed,
    velocity_limit_weight,
    g,
    H,
):
    for i in range(6):
        g[i] = 0.0
        for j in range(6):
            H[i, j] = 0.0

    cost = 0.0
    for i in range(3):
        vel = x[i + 3]
        abs_vel = abs(vel)
        deficit = abs_vel - max_speed
        if deficit <= 0.0:
            continue
        sign = 1.0 if vel >= 0.0 else -1.0
        cost += 0.5 * velocity_limit_weight * deficit * deficit
        g[i + 3] = velocity_limit_weight * deficit * sign
        H[i + 3, i + 3] = velocity_limit_weight

    return cost


@njit(cache=True)
def _static_boundary_velocity_feasibility_3d_numba(
    xs,
    spheres,
    boxes,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    max_speed,
):
    required_clearance = agent_radius + obstacle_buffer
    lower_bound = agent_radius + boundary_buffer
    upper_x = env_size[0] - lower_bound
    upper_y = env_size[1] - lower_bound
    upper_z = env_size[2] - lower_bound

    for idx in range(xs.shape[0]):
        x = xs[idx]
        signed_dist = _signed_distance_to_obstacles_3d_numba(
            x[0], x[1], x[2], spheres, boxes
        )
        if signed_dist < required_clearance - 1e-6:
            return False, idx, 1, signed_dist
        if (
            x[0] < lower_bound
            or x[0] > upper_x
            or x[1] < lower_bound
            or x[1] > upper_y
            or x[2] < lower_bound
            or x[2] > upper_z
        ):
            return False, idx, 2, 0.0
        if (
            abs(x[3]) > max_speed + 1e-6
            or abs(x[4]) > max_speed + 1e-6
            or abs(x[5]) > max_speed + 1e-6
        ):
            return False, idx, 3, 0.0
    return True, -1, 0, 0.0


@njit(cache=True)
def _running_cost_terms_inplace_fused_numba(
    x,
    u,
    x_ref,
    u_ref,
    pos_w,
    vel_w,
    control_track_w,
    control_effort_w,
    spheres,
    boxes,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    clearance_margin,
    obstacle_weight,
    max_speed,
    velocity_limit_weight,
    Lx,
    Lu,
    Lxx,
    Luu,
    Lxu,
    obstacle_Lx,
    obstacle_Lxx,
    boundary_Lx,
    boundary_Lxx,
    velocity_Lx,
    velocity_Lxx,
):
    cost = 0.0
    control_w = control_track_w + control_effort_w

    for i in range(3):
        dpos = x[i] - x_ref[i]
        dvel = x[i + 3] - x_ref[i + 3]
        du = u[i] - u_ref[i]
        ui = u[i]

        cost += 0.5 * pos_w * dpos * dpos
        cost += 0.5 * vel_w * dvel * dvel
        cost += 0.5 * control_track_w * du * du
        cost += 0.5 * control_effort_w * ui * ui

        Lx[i] = pos_w * dpos
        Lx[i + 3] = vel_w * dvel
        Lu[i] = control_track_w * du + control_effort_w * ui

    Lxx[:, :] = 0.0
    Luu[:, :] = 0.0
    Lxu[:, :] = 0.0
    for i in range(3):
        Lxx[i, i] = pos_w
        Lxx[i + 3, i + 3] = vel_w
        Luu[i, i] = control_w

    obstacle_cost = _obstacle_penalty_and_derivatives_3d_inplace_numba(
        x,
        spheres,
        boxes,
        agent_radius,
        obstacle_buffer,
        clearance_margin,
        obstacle_weight,
        obstacle_Lx,
        obstacle_Lxx,
    )
    cost += obstacle_cost
    Lx += obstacle_Lx
    Lxx += obstacle_Lxx

    boundary_cost = _boundary_penalty_and_derivatives_3d_inplace_numba(
        x,
        env_size,
        agent_radius,
        boundary_buffer,
        clearance_margin,
        obstacle_weight,
        boundary_Lx,
        boundary_Lxx,
    )
    cost += boundary_cost
    Lx += boundary_Lx
    Lxx += boundary_Lxx

    velocity_cost = _velocity_limit_penalty_and_derivatives_3d_inplace_numba(
        x,
        max_speed,
        velocity_limit_weight,
        velocity_Lx,
        velocity_Lxx,
    )
    cost += velocity_cost
    Lx += velocity_Lx
    Lxx += velocity_Lxx
    return cost


@njit(cache=True)
def _terminal_cost_terms_inplace_fused_numba(
    x,
    goal_state,
    goal_weight,
    terminal_velocity_weight,
    spheres,
    boxes,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    clearance_margin,
    obstacle_weight,
    max_speed,
    velocity_limit_weight,
    Lx,
    Lxx,
    obstacle_Lx,
    obstacle_Lxx,
    boundary_Lx,
    boundary_Lxx,
    velocity_Lx,
    velocity_Lxx,
):
    cost = 0.0

    for i in range(3):
        dpos = x[i] - goal_state[i]
        dvel = x[i + 3] - goal_state[i + 3]
        cost += 0.5 * goal_weight * dpos * dpos
        cost += 0.5 * terminal_velocity_weight * dvel * dvel
        Lx[i] = goal_weight * dpos
        Lx[i + 3] = terminal_velocity_weight * dvel

    Lxx[:, :] = 0.0
    for i in range(3):
        Lxx[i, i] = goal_weight
        Lxx[i + 3, i + 3] = terminal_velocity_weight

    obstacle_cost = _obstacle_penalty_and_derivatives_3d_inplace_numba(
        x,
        spheres,
        boxes,
        agent_radius,
        obstacle_buffer,
        clearance_margin,
        obstacle_weight,
        obstacle_Lx,
        obstacle_Lxx,
    )
    cost += obstacle_cost
    Lx += obstacle_Lx
    Lxx += obstacle_Lxx

    boundary_cost = _boundary_penalty_and_derivatives_3d_inplace_numba(
        x,
        env_size,
        agent_radius,
        boundary_buffer,
        clearance_margin,
        obstacle_weight,
        boundary_Lx,
        boundary_Lxx,
    )
    cost += boundary_cost
    Lx += boundary_Lx
    Lxx += boundary_Lxx

    velocity_cost = _velocity_limit_penalty_and_derivatives_3d_inplace_numba(
        x,
        max_speed,
        velocity_limit_weight,
        velocity_Lx,
        velocity_Lxx,
    )
    cost += velocity_cost
    Lx += velocity_Lx
    Lxx += velocity_Lxx
    return cost


@dataclass
class Quadcopter6DTrajOptOptions:
    goal_weight: float = 200.0
    obstacle_weight: float = 100.0
    max_iter: int = 50
    th_stop: float = 1e-2
    init_reg: float = 1e2
    th_acceptnegstep: float = 0.3
    callbacks: bool = False

    state_tracking_weight: float = 0.05
    velocity_tracking_weight: float = 0.02
    control_tracking_weight: float = 0.02
    control_effort_weight: float = 0.005
    terminal_velocity_weight: float = 5.0
    clearance_margin: float = 0.0
    velocity_limit_weight: float = 100.0


@dataclass
class Quadcopter6DTrajOptResult:
    success: bool
    feasible: bool
    xs: np.ndarray
    us: np.ndarray
    cost: float
    solver_iters: int
    path_view: Optional["OptimizedTrajectoryView3D"] = None


class OptimizedTrajectoryView3D:
    def __init__(self, planner, xs: np.ndarray, us: np.ndarray, cost: float):
        self.env = planner.env
        self.agent = planner.agent
        self.start = np.asarray(planner.start, dtype=np.float64)
        self.goal = np.asarray(planner.goal, dtype=np.float64)
        self.goal_radius = float(planner.goal_radius)
        self.minimum_time_step = float(planner.minimum_time_step)
        self.roundoff_digits = int(getattr(planner, "roundoff_digits", 1))
        self.path_found = True
        self.path_cost = float(cost)
        self.path_time = float(us.shape[0]) * self.minimum_time_step
        self.goal_node_id = int(us.shape[0])

        self._xs = np.asarray(xs, dtype=np.float64)
        self._us = np.asarray(us, dtype=np.float64)

    def get_high_resolution_path_numpy_array(self):
        return self._xs.copy()

    def get_high_resolution_path(self):
        path_dict = {}
        for i, state in enumerate(self._xs):
            t = round(i * self.minimum_time_step, self.roundoff_digits)
            path_dict[t] = state.copy()
        return path_dict

    def get_path(self):
        node_ids = np.arange(self._xs.shape[0], dtype=np.int32)
        states = self._xs.copy()
        controls = self._us.copy()
        timesteps = np.full(self._us.shape[0], self.minimum_time_step, dtype=np.float64)
        return node_ids, states, controls, timesteps

    def get_high_resolution_path_and_actions(self):
        timesteps = np.full(self._us.shape[0], self.minimum_time_step, dtype=np.float64)
        return self._xs.copy(), self._us.copy(), timesteps


class _QuadcopterRunningModel:
    def __init__(self, crocoddyl, *, dt: float, env, agent,
                 x_ref: np.ndarray, u_ref: np.ndarray,
                 goal_xyz: np.ndarray, options: Quadcopter6DTrajOptOptions,
                 spheres: np.ndarray, boxes: np.ndarray):
        self.crocoddyl = crocoddyl
        self.state = crocoddyl.StateVector(6)
        self.model = crocoddyl.ActionModelAbstract(self.state, 3, 9)
        self.model.calc = self.calc
        self.model.calcDiff = self.calcDiff
        self.dt = float(dt)
        self.env = env
        self.agent = agent
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        self.u_ref = np.asarray(u_ref, dtype=np.float64)
        self.goal_xyz = np.asarray(goal_xyz[:3], dtype=np.float64)
        self.options = options
        self.spheres = spheres
        self.boxes = boxes
        self.env_size = np.asarray(env.size[:3], dtype=np.float64)
        self.obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
        self.boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
        self._zero_u = np.zeros(3, dtype=np.float64)
        self._obstacle_Lx = np.zeros(6, dtype=np.float64)
        self._obstacle_Lxx = np.zeros((6, 6), dtype=np.float64)
        self._boundary_Lx = np.zeros(6, dtype=np.float64)
        self._boundary_Lxx = np.zeros((6, 6), dtype=np.float64)
        self._velocity_limit_Lx = np.zeros(6, dtype=np.float64)
        self._velocity_limit_Lxx = np.zeros((6, 6), dtype=np.float64)
        self.use_fused_numba_cost = True

        accel = float(agent.max_acceleration)
        self.model.u_lb = np.array([-accel, -accel, -accel], dtype=np.float64)
        self.model.u_ub = np.array([accel, accel, accel], dtype=np.float64)

    def _running_cost_terms_inplace_legacy(self, x, u, Lx, Lu, Lxx, Luu, Lxu):
        pos_w = self.options.state_tracking_weight
        vel_w = self.options.velocity_tracking_weight
        control_track_w = self.options.control_tracking_weight
        control_effort_w = self.options.control_effort_weight
        control_w = control_track_w + control_effort_w

        cost = 0.0
        for i in range(3):
            dpos = float(x[i]) - float(self.x_ref[i])
            dvel = float(x[i + 3]) - float(self.x_ref[i + 3])
            du = float(u[i]) - float(self.u_ref[i])
            ui = float(u[i])

            cost += 0.5 * pos_w * dpos * dpos
            cost += 0.5 * vel_w * dvel * dvel
            cost += 0.5 * control_track_w * du * du
            cost += 0.5 * control_effort_w * ui * ui

            Lx[i] = pos_w * dpos
            Lx[i + 3] = vel_w * dvel
            Lu[i] = control_track_w * du + control_effort_w * ui

        Lxx[:] = 0.0
        Luu[:] = 0.0
        Lxu[:] = 0.0
        for i in range(3):
            Lxx[i, i] = pos_w
            Lxx[i + 3, i + 3] = vel_w
            Luu[i, i] = control_w

        obstacle_cost = _obstacle_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.spheres,
            self.boxes,
            self.agent.radius,
            self.obstacle_buffer,
            self.options.clearance_margin,
            self.options.obstacle_weight,
            self._obstacle_Lx,
            self._obstacle_Lxx,
        )
        cost += obstacle_cost
        Lx += self._obstacle_Lx
        Lxx += self._obstacle_Lxx

        boundary_cost = _boundary_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.env_size,
            self.agent.radius,
            self.boundary_buffer,
            self.options.clearance_margin,
            self.options.obstacle_weight,
            self._boundary_Lx,
            self._boundary_Lxx,
        )
        cost += boundary_cost
        Lx += self._boundary_Lx
        Lxx += self._boundary_Lxx

        velocity_limit_cost = _velocity_limit_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.agent.max_speed,
            self.options.velocity_limit_weight,
            self._velocity_limit_Lx,
            self._velocity_limit_Lxx,
        )
        cost += velocity_limit_cost
        Lx += self._velocity_limit_Lx
        Lxx += self._velocity_limit_Lxx

        return cost

    def _running_cost_terms_inplace(self, x, u, Lx, Lu, Lxx, Luu, Lxu):
        if self.use_fused_numba_cost:
            return _running_cost_terms_inplace_fused_numba(
                x,
                u,
                self.x_ref,
                self.u_ref,
                self.options.state_tracking_weight,
                self.options.velocity_tracking_weight,
                self.options.control_tracking_weight,
                self.options.control_effort_weight,
                self.spheres,
                self.boxes,
                self.env_size,
                self.agent.radius,
                self.obstacle_buffer,
                self.boundary_buffer,
                self.options.clearance_margin,
                self.options.obstacle_weight,
                self.agent.max_speed,
                self.options.velocity_limit_weight,
                Lx,
                Lu,
                Lxx,
                Luu,
                Lxu,
                self._obstacle_Lx,
                self._obstacle_Lxx,
                self._boundary_Lx,
                self._boundary_Lxx,
                self._velocity_limit_Lx,
                self._velocity_limit_Lxx,
            )
        return self._running_cost_terms_inplace_legacy(x, u, Lx, Lu, Lxx, Luu, Lxu)

    def calc(self, data, x, u=None):
        if u is None:
            u_vec = self._zero_u
        else:
            u_vec = u

        dt = self.dt
        half_dt2 = 0.5 * dt * dt
        data.xnext[0] = float(x[0]) + dt * float(x[3]) + half_dt2 * float(u_vec[0])
        data.xnext[1] = float(x[1]) + dt * float(x[4]) + half_dt2 * float(u_vec[1])
        data.xnext[2] = float(x[2]) + dt * float(x[5]) + half_dt2 * float(u_vec[2])
        data.xnext[3] = float(x[3]) + dt * float(u_vec[0])
        data.xnext[4] = float(x[4]) + dt * float(u_vec[1])
        data.xnext[5] = float(x[5]) + dt * float(u_vec[2])
        data.cost = self._running_cost_terms_inplace(
            x, u_vec, data.Lx, data.Lu, data.Lxx, data.Luu, data.Lxu
        )
        data.r[:] = 0.0

    def calcDiff(self, data, x, u=None):
        if u is None:
            u_vec = self._zero_u
        else:
            u_vec = u

        dt = self.dt
        half_dt2 = 0.5 * dt * dt
        data.Fx[:] = 0.0
        for i in range(6):
            data.Fx[i, i] = 1.0
        data.Fx[0, 3] = dt
        data.Fx[1, 4] = dt
        data.Fx[2, 5] = dt

        data.Fu[:] = 0.0
        data.Fu[0, 0] = half_dt2
        data.Fu[1, 1] = half_dt2
        data.Fu[2, 2] = half_dt2
        data.Fu[3, 0] = dt
        data.Fu[4, 1] = dt
        data.Fu[5, 2] = dt

        data.cost = self._running_cost_terms_inplace(
            x, u_vec, data.Lx, data.Lu, data.Lxx, data.Luu, data.Lxu
        )


class _QuadcopterTerminalModel:
    def __init__(self, crocoddyl, *, env, agent, goal_state: np.ndarray,
                 options: Quadcopter6DTrajOptOptions,
                 spheres: np.ndarray, boxes: np.ndarray):
        self.crocoddyl = crocoddyl
        self.state = crocoddyl.StateVector(6)
        self.model = crocoddyl.ActionModelAbstract(self.state, 0, 6)
        self.model.calc = self.calc
        self.model.calcDiff = self.calcDiff
        self.env = env
        self.agent = agent
        self.goal_state = np.asarray(goal_state, dtype=np.float64)
        self.options = options
        self.spheres = spheres
        self.boxes = boxes
        self.env_size = np.asarray(env.size[:3], dtype=np.float64)
        self.obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
        self.boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
        self.goal_boost = 5.0
        self._obstacle_Lx = np.zeros(6, dtype=np.float64)
        self._obstacle_Lxx = np.zeros((6, 6), dtype=np.float64)
        self._boundary_Lx = np.zeros(6, dtype=np.float64)
        self._boundary_Lxx = np.zeros((6, 6), dtype=np.float64)
        self._velocity_limit_Lx = np.zeros(6, dtype=np.float64)
        self._velocity_limit_Lxx = np.zeros((6, 6), dtype=np.float64)
        self.use_fused_numba_cost = True

    def _terminal_cost_terms_inplace_legacy(self, x, Lx, Lxx):
        goal_weight = self.goal_boost * self.options.goal_weight
        vel_weight = self.goal_boost * self.options.terminal_velocity_weight

        cost = 0.0
        for i in range(3):
            dpos = float(x[i]) - float(self.goal_state[i])
            dvel = float(x[i + 3]) - float(self.goal_state[i + 3])
            cost += 0.5 * goal_weight * dpos * dpos
            cost += 0.5 * vel_weight * dvel * dvel
            Lx[i] = goal_weight * dpos
            Lx[i + 3] = vel_weight * dvel

        Lxx[:] = 0.0
        for i in range(3):
            Lxx[i, i] = goal_weight
            Lxx[i + 3, i + 3] = vel_weight

        obstacle_cost = _obstacle_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.spheres,
            self.boxes,
            self.agent.radius,
            self.obstacle_buffer,
            self.options.clearance_margin,
            self.options.obstacle_weight,
            self._obstacle_Lx,
            self._obstacle_Lxx,
        )
        cost += obstacle_cost
        Lx += self._obstacle_Lx
        Lxx += self._obstacle_Lxx

        boundary_cost = _boundary_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.env_size,
            self.agent.radius,
            self.boundary_buffer,
            self.options.clearance_margin,
            self.options.obstacle_weight,
            self._boundary_Lx,
            self._boundary_Lxx,
        )
        cost += boundary_cost
        Lx += self._boundary_Lx
        Lxx += self._boundary_Lxx

        velocity_limit_cost = _velocity_limit_penalty_and_derivatives_3d_inplace_numba(
            x,
            self.agent.max_speed,
            self.options.velocity_limit_weight,
            self._velocity_limit_Lx,
            self._velocity_limit_Lxx,
        )
        cost += velocity_limit_cost
        Lx += self._velocity_limit_Lx
        Lxx += self._velocity_limit_Lxx
        return cost

    def _terminal_cost_terms_inplace(self, x, Lx, Lxx):
        if self.use_fused_numba_cost:
            return _terminal_cost_terms_inplace_fused_numba(
                x,
                self.goal_state,
                self.goal_boost * self.options.goal_weight,
                self.goal_boost * self.options.terminal_velocity_weight,
                self.spheres,
                self.boxes,
                self.env_size,
                self.agent.radius,
                self.obstacle_buffer,
                self.boundary_buffer,
                self.options.clearance_margin,
                self.options.obstacle_weight,
                self.agent.max_speed,
                self.options.velocity_limit_weight,
                Lx,
                Lxx,
                self._obstacle_Lx,
                self._obstacle_Lxx,
                self._boundary_Lx,
                self._boundary_Lxx,
                self._velocity_limit_Lx,
                self._velocity_limit_Lxx,
            )
        return self._terminal_cost_terms_inplace_legacy(x, Lx, Lxx)

    def calc(self, data, x, u=None):
        data.xnext[:] = x
        data.cost = self._terminal_cost_terms_inplace(x, data.Lx, data.Lxx)
        data.r[:] = 0.0

    def calcDiff(self, data, x, u=None):
        data.cost = self._terminal_cost_terms_inplace(x, data.Lx, data.Lxx)
        data.Fx[:] = 0.0
        data.Fu[:] = 0.0
        data.Lu[:] = 0.0
        data.Luu[:] = 0.0
        data.Lxu[:] = 0.0


def _import_crocoddyl():
    try:
        import crocoddyl
    except ImportError as exc:
        raise ImportError(
            "crocoddyl is required for QuadCopter6D trajectory repair. "
            "Install it first, or run this from an environment where it is available."
        ) from exc
    return crocoddyl


def optimize_quadcopter6d_warm_start(
    *,
    start: np.ndarray,
    goal_state: np.ndarray,
    env,
    agent,
    xs_init: np.ndarray,
    us_init: np.ndarray,
    dt: float,
    goal_radius: float = 0.25,
    options: Optional[Quadcopter6DTrajOptOptions] = None,
    spheres: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
) -> Quadcopter6DTrajOptResult:
    crocoddyl = _import_crocoddyl()

    if options is None:
        options = Quadcopter6DTrajOptOptions()

    start = np.asarray(start, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    xs_init = np.asarray(xs_init, dtype=np.float64)
    us_init = np.asarray(us_init, dtype=np.float64)
    if spheres is None or boxes is None:
        spheres, boxes = _extract_obstacle_arrays_3d(env)
    obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
    boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
    env_size = np.asarray(env.size[:3], dtype=np.float64)

    if xs_init.ndim != 2 or xs_init.shape[1] != 6:
        raise ValueError(f"xs_init must have shape (N+1, 6), got {xs_init.shape}")
    if us_init.ndim != 2 or us_init.shape[1] != 3:
        raise ValueError(f"us_init must have shape (N, 3), got {us_init.shape}")
    if xs_init.shape[0] != us_init.shape[0] + 1:
        raise ValueError("xs_init must contain exactly one more state than us_init controls")

    running_models: List[object] = []
    for x_ref, u_ref in zip(xs_init[:-1], us_init):
        raw_model = _QuadcopterRunningModel(
            crocoddyl,
            dt=dt,
            env=env,
            agent=agent,
            x_ref=x_ref,
            u_ref=u_ref,
            goal_xyz=goal_state[:3],
            options=options,
            spheres=spheres,
            boxes=boxes,
        ).model
        running_models.append(raw_model)

    terminal_model = _QuadcopterTerminalModel(
        crocoddyl,
        env=env,
        agent=agent,
        goal_state=goal_state,
        options=options,
        spheres=spheres,
        boxes=boxes,
    ).model

    accel = float(agent.max_acceleration)
    for model in running_models:
        model.u_lb = np.array([-accel, -accel, -accel], dtype=np.float64)
        model.u_ub = np.array([accel, accel, accel], dtype=np.float64)

    problem = crocoddyl.ShootingProblem(start, running_models, terminal_model)
    solver = crocoddyl.SolverBoxFDDP(problem)
    solver.th_stop = options.th_stop
    solver.th_acceptnegstep = options.th_acceptnegstep

    if options.callbacks:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    xs_guess = [x.copy() for x in xs_init]
    us_guess = [u.copy() for u in us_init]
    success = bool(solver.solve(xs_guess, us_guess, options.max_iter, False, options.init_reg))

    xs_out = np.asarray(solver.xs, dtype=np.float64)
    us_out = np.asarray(solver.us, dtype=np.float64)

    feasible, _, _, _ = _static_boundary_velocity_feasibility_3d_numba(
        xs_out,
        spheres,
        boxes,
        env_size,
        float(agent.radius),
        obstacle_buffer,
        boundary_buffer,
        float(agent.max_speed),
    )

    if np.linalg.norm(xs_out[-1][:3] - goal_state[:3]) > float(goal_radius):
        feasible = False

    return Quadcopter6DTrajOptResult(
        success=success,
        feasible=feasible,
        xs=xs_out,
        us=us_out,
        cost=float(solver.cost),
        solver_iters=int(solver.iter),
    )


def optimize_dbrrt_quadcopter6d_path(
    planner,
    *,
    options: Optional[Quadcopter6DTrajOptOptions] = None,
) -> Quadcopter6DTrajOptResult:
    highres_states, controls, _ = planner.get_high_resolution_path_and_actions()
    if highres_states.shape[0] == 0:
        raise ValueError("Planner has no path to optimize")

    us_init = np.asarray(controls, dtype=np.float64)
    if highres_states.shape[0] != us_init.shape[0] + 1:
        raise ValueError(
            f"Warm start length mismatch: states={highres_states.shape[0]}, controls={us_init.shape[0]}"
        )

    goal_state = np.zeros(6, dtype=np.float64)
    goal_state[:3] = np.asarray(planner.goal[:3], dtype=np.float64)

    result = optimize_quadcopter6d_warm_start(
        start=np.asarray(planner.start, dtype=np.float64),
        goal_state=goal_state,
        env=planner.env,
        agent=planner.agent,
        xs_init=highres_states,
        us_init=us_init,
        dt=float(planner.minimum_time_step),
        goal_radius=float(planner.goal_radius),
        options=options,
        spheres=getattr(planner, "static_circular_obstacles", None),
        boxes=getattr(planner, "static_rectangular_obstacles", None),
    )
    result.path_view = OptimizedTrajectoryView3D(planner, result.xs, result.us, result.cost)
    return result
