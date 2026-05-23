"""
Optional Crocoddyl-based repair stage for discontinuity-bounded unicycle paths.

This is a fixed-horizon feasibility-repair stage around the Db-RRT warm start:
    - fixed horizon N from the warm-start trajectory
    - warm-started BoxFDDP solve
    - unicycle dynamics
    - running obstacle / tracking / control costs
    - terminal goal cost

The dependency on `crocoddyl` is optional and imported lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence
import numpy as np
from numba import njit
import crocoddyl


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _angle_diff(theta: float, theta_ref: float) -> float:
    return _wrap_angle(theta - theta_ref)


@njit(cache=True)
def _wrap_angle_numba(theta):
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def _angle_diff_numba(theta, theta_ref):
    return _wrap_angle_numba(theta - theta_ref)


def _extract_obstacle_arrays(env):
    circles_src = getattr(env, "static_circular_obstacles", None)
    rects_src = getattr(env, "static_rectangular_obstacles", None)

    if circles_src is None:
        circles = np.empty((0, 3), dtype=np.float64)
    elif isinstance(circles_src, np.ndarray):
        circles = np.asarray(circles_src, dtype=np.float64).reshape(-1, 3)
    else:
        circles_src = list(circles_src)
        circles = np.empty((len(circles_src), 3), dtype=np.float64)
        for i, obstacle in enumerate(circles_src):
            circles[i, 0] = obstacle.x
            circles[i, 1] = obstacle.y
            circles[i, 2] = obstacle.r

    if rects_src is None:
        rects = np.empty((0, 4), dtype=np.float64)
    elif isinstance(rects_src, np.ndarray):
        rects = np.asarray(rects_src, dtype=np.float64).reshape(-1, 4)
    else:
        rects_src = list(rects_src)
        rects = np.empty((len(rects_src), 4), dtype=np.float64)
        for i, obstacle in enumerate(rects_src):
            half_w = 0.5 * obstacle.w
            half_h = 0.5 * obstacle.h
            rects[i, 0] = obstacle.x - half_w
            rects[i, 1] = obstacle.x + half_w
            rects[i, 2] = obstacle.y - half_h
            rects[i, 3] = obstacle.y + half_h

    return circles, rects


@njit(cache=True)
def _signed_distance_to_circle_numba(x, y, cx, cy, r):
    dx = x - cx
    dy = y - cy
    return math.sqrt(dx * dx + dy * dy) - r


@njit(cache=True)
def _signed_distance_to_rectangle_numba(x, y, xmin, xmax, ymin, ymax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    dx = abs(x - cx) - hx
    dy = abs(y - cy) - hy

    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    outside_dist = math.sqrt(outside_dx * outside_dx + outside_dy * outside_dy)
    inside_dist = min(max(dx, dy), 0.0)
    return outside_dist + inside_dist


@njit(cache=True)
def _signed_distance_to_obstacles_numba(x, y, circles, rects):
    best = 1e6

    for i in range(circles.shape[0]):
        d = _signed_distance_to_circle_numba(
            x, y, circles[i, 0], circles[i, 1], circles[i, 2]
        )
        if d < best:
            best = d

    for i in range(rects.shape[0]):
        d = _signed_distance_to_rectangle_numba(
            x, y, rects[i, 0], rects[i, 1], rects[i, 2], rects[i, 3]
        )
        if d < best:
            best = d

    return best


@njit(cache=True)
def _signed_distance_and_gradient_to_circle_numba(x, y, cx, cy, r):
    dx = x - cx
    dy = y - cy
    norm = math.sqrt(dx * dx + dy * dy)
    if norm <= 1e-12:
        return -r, 1.0, 0.0
    inv_norm = 1.0 / norm
    return norm - r, dx * inv_norm, dy * inv_norm


@njit(cache=True)
def _signed_distance_and_gradient_to_rectangle_numba(x, y, xmin, xmax, ymin, ymax):
    closest_x = min(max(x, xmin), xmax)
    closest_y = min(max(y, ymin), ymax)
    vx = x - closest_x
    vy = y - closest_y
    outside_dist_sq = vx * vx + vy * vy

    if outside_dist_sq > 0.0:
        outside_dist = math.sqrt(outside_dist_sq)
        inv_dist = 1.0 / outside_dist
        return outside_dist, vx * inv_dist, vy * inv_dist

    dist_left = x - xmin
    dist_right = xmax - x
    dist_bottom = y - ymin
    dist_top = ymax - y

    best = dist_left
    grad0 = -1.0
    grad1 = 0.0
    if dist_right < best:
        best = dist_right
        grad0 = 1.0
        grad1 = 0.0
    if dist_bottom < best:
        best = dist_bottom
        grad0 = 0.0
        grad1 = -1.0
    if dist_top < best:
        best = dist_top
        grad0 = 0.0
        grad1 = 1.0

    return -best, grad0, grad1


@njit(cache=True)
def _signed_distance_and_gradient_to_obstacles_numba(x, y, circles, rects):
    best = 1e6
    best_grad0 = 0.0
    best_grad1 = 0.0

    for i in range(circles.shape[0]):
        d, grad0, grad1 = _signed_distance_and_gradient_to_circle_numba(
            x, y, circles[i, 0], circles[i, 1], circles[i, 2]
        )
        if d < best:
            best = d
            best_grad0 = grad0
            best_grad1 = grad1

    for i in range(rects.shape[0]):
        d, grad0, grad1 = _signed_distance_and_gradient_to_rectangle_numba(
            x, y, rects[i, 0], rects[i, 1], rects[i, 2], rects[i, 3]
        )
        if d < best:
            best = d
            best_grad0 = grad0
            best_grad1 = grad1

    return best, best_grad0, best_grad1


@njit(cache=True)
def _static_boundary_feasibility_numba(
    xs,
    circles,
    rects,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    feasibility_tolerance,
):
    required_clearance = agent_radius + obstacle_buffer
    lower_x = agent_radius + boundary_buffer
    upper_x = env_size[0] - agent_radius - boundary_buffer
    lower_y = agent_radius + boundary_buffer
    upper_y = env_size[1] - agent_radius - boundary_buffer

    for i in range(xs.shape[0]):
        x = xs[i]
        signed_dist = _signed_distance_to_obstacles_numba(x[0], x[1], circles, rects)
        if signed_dist < required_clearance - feasibility_tolerance:
            return False, i, 1, signed_dist
        if (
            x[0] < lower_x - feasibility_tolerance
            or x[0] > upper_x + feasibility_tolerance
            or x[1] < lower_y - feasibility_tolerance
            or x[1] > upper_y + feasibility_tolerance
        ):
            return False, i, 2, 0.0

    return True, -1, 0, 0.0


@njit(cache=True)
def _obstacle_penalty_and_derivatives_inplace_numba(
    x,
    circles,
    rects,
    agent_radius,
    obstacle_buffer,
    clearance_margin,
    obstacle_weight,
    g,
    H,
):
    signed_dist, grad0, grad1 = _signed_distance_and_gradient_to_obstacles_numba(
        x[0], x[1], circles, rects
    )
    clearance = signed_dist - (agent_radius + obstacle_buffer)
    deficit = clearance_margin - clearance
    for i in range(3):
        g[i] = 0.0
        for j in range(3):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    cost = 0.5 * obstacle_weight * deficit * deficit
    g[0] = -obstacle_weight * deficit * grad0
    g[1] = -obstacle_weight * deficit * grad1
    g[2] = 0.0

    H[0, 0] = obstacle_weight * grad0 * grad0
    H[0, 1] = obstacle_weight * grad0 * grad1
    H[1, 0] = H[0, 1]
    H[1, 1] = obstacle_weight * grad1 * grad1
    H[0, 2] = 0.0
    H[1, 2] = 0.0
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 0.0

    return cost


@njit(cache=True)
def _obstacle_penalty_and_derivatives_inplace_finite_diff_numba(
    x,
    circles,
    rects,
    agent_radius,
    obstacle_buffer,
    clearance_margin,
    obstacle_weight,
    g,
    H,
):
    clearance = _signed_distance_to_obstacles_numba(x[0], x[1], circles, rects) - (
        agent_radius + obstacle_buffer
    )
    deficit = clearance_margin - clearance
    for i in range(3):
        g[i] = 0.0
        for j in range(3):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    eps = 1e-4
    x0 = x[0]
    y0 = x[1]
    # theta0 = x[2]

    xp0 = _signed_distance_to_obstacles_numba(x0 + eps, y0, circles, rects) - (
        agent_radius + obstacle_buffer
    )
    xm0 = _signed_distance_to_obstacles_numba(x0 - eps, y0, circles, rects) - (
        agent_radius + obstacle_buffer
    )
    grad0 = (xp0 - xm0) / (2.0 * eps)
    # hess00 = (xp0 - 2.0 * clearance + xm0) / (eps * eps)

    xp1 = _signed_distance_to_obstacles_numba(x0, y0 + eps, circles, rects) - (
        agent_radius + obstacle_buffer
    )
    xm1 = _signed_distance_to_obstacles_numba(x0, y0 - eps, circles, rects) - (
        agent_radius + obstacle_buffer
    )
    grad1 = (xp1 - xm1) / (2.0 * eps)
    # hess11 = (xp1 - 2.0 * clearance + xm1) / (eps * eps)

    cost = 0.5 * obstacle_weight * deficit * deficit
    g[0] = -obstacle_weight * deficit * grad0
    g[1] = -obstacle_weight * deficit * grad1
    g[2] = 0.0

    # H[0, 0] = obstacle_weight * grad0 * grad0 - obstacle_weight * deficit * hess00
    # H[0, 1] = obstacle_weight * grad0 * grad1
    # H[1, 0] = H[0, 1]
    # H[1, 1] = obstacle_weight * grad1 * grad1 - obstacle_weight * deficit * hess11
    # H[0, 2] = 0.0
    # H[1, 2] = 0.0
    # H[2, 0] = 0.0
    # H[2, 1] = 0.0
    # H[2, 2] = 0.0
    # H[:, :] = 0.0

    H[0, 0] = obstacle_weight * grad0 * grad0
    H[0, 1] = obstacle_weight * grad0 * grad1
    H[1, 0] = H[0, 1]
    H[1, 1] = obstacle_weight * grad1 * grad1

    H[0, 2] = 0.0
    H[1, 2] = 0.0
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 0.0

    return cost


@njit(cache=True)
def _boundary_penalty_and_derivatives_inplace_numba(
    x,
    env_size,
    agent_radius,
    boundary_buffer,
    clearance_margin,
    boundary_weight,
    g,
    H,
):
    lower_x = agent_radius + boundary_buffer
    upper_x = env_size[0] - agent_radius - boundary_buffer
    lower_y = agent_radius + boundary_buffer
    upper_y = env_size[1] - agent_radius - boundary_buffer

    clear_x_min = x[0] - lower_x
    clear_x_max = upper_x - x[0]
    clear_y_min = x[1] - lower_y
    clear_y_max = upper_y - x[1]

    clearance = clear_x_min
    axis = 0
    sign = 1.0
    if clear_x_max < clearance:
        clearance = clear_x_max
        axis = 0
        sign = -1.0
    if clear_y_min < clearance:
        clearance = clear_y_min
        axis = 1
        sign = 1.0
    if clear_y_max < clearance:
        clearance = clear_y_max
        axis = 1
        sign = -1.0

    deficit = clearance_margin - clearance
    for i in range(3):
        g[i] = 0.0
        for j in range(3):
            H[i, j] = 0.0

    if deficit <= 0.0:
        return 0.0

    cost = 0.5 * boundary_weight * deficit * deficit
    g[axis] = -boundary_weight * deficit * sign
    H[axis, axis] = boundary_weight
    return cost


@njit(cache=True)
def _running_cost_terms_inplace_fused_numba(
    x,
    u,
    x_ref,
    u_ref,
    state_w,
    theta_w,
    control_track_w,
    control_effort_w,
    circles,
    rects,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    clearance_margin,
    obstacle_weight,
    Lx,
    Lu,
    Lxx,
    Luu,
    Lxu,
    obstacle_Lx,
    obstacle_Lxx,
    boundary_Lx,
    boundary_Lxx,
):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    u0 = u[0]
    u1 = u[1]

    dx0 = x0 - x_ref[0]
    dx1 = x1 - x_ref[1]
    dtheta = _angle_diff_numba(x2, x_ref[2])
    du0 = u0 - u_ref[0]
    du1 = u1 - u_ref[1]
    control_w = control_track_w + control_effort_w

    cost = 0.5 * state_w * (dx0 * dx0 + dx1 * dx1)
    cost += 0.5 * theta_w * dtheta * dtheta
    cost += 0.5 * control_track_w * (du0 * du0 + du1 * du1)
    cost += 0.5 * control_effort_w * (u0 * u0 + u1 * u1)

    Lx[0] = state_w * dx0
    Lx[1] = state_w * dx1
    Lx[2] = theta_w * dtheta

    Lu[0] = control_track_w * du0 + control_effort_w * u0
    Lu[1] = control_track_w * du1 + control_effort_w * u1

    Lxx[:, :] = 0.0
    Lxx[0, 0] = state_w
    Lxx[1, 1] = state_w
    Lxx[2, 2] = theta_w

    Luu[:, :] = 0.0
    Luu[0, 0] = control_w
    Luu[1, 1] = control_w

    Lxu[:, :] = 0.0

    obstacle_cost = _obstacle_penalty_and_derivatives_inplace_numba(
        x,
        circles,
        rects,
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

    boundary_cost = _boundary_penalty_and_derivatives_inplace_numba(
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
    return cost


@njit(cache=True)
def _terminal_cost_terms_inplace_fused_numba(
    x,
    goal_xy,
    goal_weight,
    circles,
    rects,
    env_size,
    agent_radius,
    obstacle_buffer,
    boundary_buffer,
    clearance_margin,
    obstacle_weight,
    Lx,
    Lxx,
    obstacle_Lx,
    obstacle_Lxx,
    boundary_Lx,
    boundary_Lxx,
):
    dxy0 = x[0] - goal_xy[0]
    dxy1 = x[1] - goal_xy[1]

    cost = 0.5 * goal_weight * (dxy0 * dxy0 + dxy1 * dxy1)

    Lx[0] = goal_weight * dxy0
    Lx[1] = goal_weight * dxy1
    Lx[2] = 0.0

    Lxx[:, :] = 0.0
    Lxx[0, 0] = goal_weight
    Lxx[1, 1] = goal_weight
    Lxx[2, 2] = 0.0

    obstacle_cost = _obstacle_penalty_and_derivatives_inplace_numba(
        x,
        circles,
        rects,
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

    boundary_cost = _boundary_penalty_and_derivatives_inplace_numba(
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
    return cost


@dataclass
class UnicycleTrajOptOptions:
    # These top-level values mirror Dynoplan's visible Options_trajopt defaults
    # for the fixed-horizon trajectory-optimization branch.
    goal_weight: float = 200.0
    obstacle_weight: float = 50.0
    max_iter: int = 50
    th_stop: float = 1e-2
    init_reg: float = 1e2
    th_acceptnegstep: float = 0.3
    callbacks: bool = False

    # These are local approximations of Dynoplan's internal regularization terms.
    # They do not have a one-to-one mapping with the C++ implementation.
    state_tracking_weight: float = 1e-5 #0.05
    theta_tracking_weight: float = 0.0 #0.02
    control_tracking_weight: float = 0.0 #0.02
    control_effort_weight: float = 0.02 #0.005
    clearance_margin: float = 0.0
    feasibility_tolerance: float = 1e-3
    retry_on_infeasible: bool = False
    allow_raw_fallback: bool = True


@dataclass
class UnicycleTrajOptResult:
    success: bool
    feasible: bool
    optimizer_output_feasible: bool
    source: str
    xs: np.ndarray
    us: np.ndarray
    cost: float
    solver_iters: int
    path_view: Optional["OptimizedTrajectoryView"] = None


class OptimizedTrajectoryView:
    """
    Planner-like wrapper around an optimized trajectory.

    This exposes the same high-resolution path accessors used by the rest of
    the codebase (`get_high_resolution_path`, `get_high_resolution_path_numpy_array`,
    `get_path`) so downstream visualization / KCBS-style code can consume the
    optimized result without special-case trajectory handling.
    """

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
    

class _UnicycleRunningModel:
    def __init__(self, crocoddyl, *, dt: float, env, agent,
                 x_ref: np.ndarray, u_ref: np.ndarray,
                 goal_xy: np.ndarray, options: UnicycleTrajOptOptions,
                 circles: np.ndarray, rects: np.ndarray):
        self.crocoddyl = crocoddyl
        self.state = crocoddyl.StateVector(3)
        self.model = crocoddyl.ActionModelAbstract(self.state, 2, 6)
        self.model.calc = self.calc
        self.model.calcDiff = self.calcDiff
        self.dt = float(dt)
        self.env = env
        self.agent = agent
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        self.u_ref = np.asarray(u_ref, dtype=np.float64)
        self.goal_xy = np.asarray(goal_xy[:2], dtype=np.float64)
        self.options = options
        self.circles = circles
        self.rects = rects
        self.env_size = np.asarray(env.size, dtype=np.float64)
        self.obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
        self.boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
        self._zero_u = np.zeros(2, dtype=np.float64)
        self._obstacle_Lx = np.zeros(3, dtype=np.float64)
        self._obstacle_Lxx = np.zeros((3, 3), dtype=np.float64)
        self._boundary_Lx = np.zeros(3, dtype=np.float64)
        self._boundary_Lxx = np.zeros((3, 3), dtype=np.float64)
        self.use_fused_numba_cost = True

        self.model.u_lb = np.array([-agent.max_speed, -agent.max_omega], dtype=np.float64)
        self.model.u_ub = np.array([agent.max_speed, agent.max_omega], dtype=np.float64)

    def _running_cost_terms_inplace_legacy(self, x, u, Lx, Lu, Lxx, Luu, Lxu):
        x0 = float(x[0])
        x1 = float(x[1])
        x2 = float(x[2])
        u0 = float(u[0])
        u1 = float(u[1])

        dx0 = x0 - float(self.x_ref[0])
        dx1 = x1 - float(self.x_ref[1])
        dtheta = _angle_diff(x2, float(self.x_ref[2]))
        du0 = u0 - float(self.u_ref[0])
        du1 = u1 - float(self.u_ref[1])

        state_w = self.options.state_tracking_weight
        theta_w = self.options.theta_tracking_weight
        control_track_w = self.options.control_tracking_weight
        control_effort_w = self.options.control_effort_weight
        control_w = control_track_w + control_effort_w

        cost = 0.5 * state_w * (dx0 * dx0 + dx1 * dx1)
        cost += 0.5 * theta_w * dtheta * dtheta
        cost += 0.5 * control_track_w * (du0 * du0 + du1 * du1)
        cost += 0.5 * control_effort_w * (u0 * u0 + u1 * u1)

        Lx[0] = state_w * dx0
        Lx[1] = state_w * dx1
        Lx[2] = theta_w * dtheta

        Lu[0] = control_track_w * du0 + control_effort_w * u0
        Lu[1] = control_track_w * du1 + control_effort_w * u1

        Lxx[:] = 0.0
        Lxx[0, 0] = state_w
        Lxx[1, 1] = state_w
        Lxx[2, 2] = theta_w

        Luu[:] = 0.0
        Luu[0, 0] = control_w
        Luu[1, 1] = control_w

        Lxu[:] = 0.0

        obstacle_cost = _obstacle_penalty_and_derivatives_inplace_numba(
            x,
            self.circles,
            self.rects,
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

        boundary_cost = _boundary_penalty_and_derivatives_inplace_numba(
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

        return cost

    def _running_cost_terms_inplace(self, x, u, Lx, Lu, Lxx, Luu, Lxu):
        if self.use_fused_numba_cost:
            return _running_cost_terms_inplace_fused_numba(
                x,
                u,
                self.x_ref,
                self.u_ref,
                self.options.state_tracking_weight,
                self.options.theta_tracking_weight,
                self.options.control_tracking_weight,
                self.options.control_effort_weight,
                self.circles,
                self.rects,
                self.env_size,
                self.agent.radius,
                self.obstacle_buffer,
                self.boundary_buffer,
                self.options.clearance_margin,
                self.options.obstacle_weight,
                Lx,
                Lu,
                Lxx,
                Luu,
                Lxu,
                self._obstacle_Lx,
                self._obstacle_Lxx,
                self._boundary_Lx,
                self._boundary_Lxx,
            )
        return self._running_cost_terms_inplace_legacy(x, u, Lx, Lu, Lxx, Luu, Lxu)

    def calc(self, data, x, u=None):
        theta = float(x[2])
        if u is None:
            v = 0.0
            omega = 0.0
            u_vec = self._zero_u
        else:
            v = float(u[0])
            omega = float(u[1])
            u_vec = u

        data.xnext[0] = float(x[0]) + self.dt * v * math.cos(theta)
        data.xnext[1] = float(x[1]) + self.dt * v * math.sin(theta)
        data.xnext[2] = _wrap_angle(float(x[2]) + self.dt * omega)
        cost = self._running_cost_terms_inplace(
            x, u_vec, data.Lx, data.Lu, data.Lxx, data.Luu, data.Lxu
        )
        data.cost = cost
        data.r[:] = 0.0

    def calcDiff(self, data, x, u=None):
        if u is None:
            v = 0.0
            u_vec = self._zero_u
        else:
            v = float(u[0])
            u_vec = u

        theta = float(x[2])
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        data.Fx[:] = 0.0
        data.Fx[0, 0] = 1.0
        data.Fx[1, 1] = 1.0
        data.Fx[2, 2] = 1.0
        data.Fx[0, 2] = -self.dt * v * sin_theta
        data.Fx[1, 2] = self.dt * v * cos_theta

        data.Fu[:] = 0.0
        data.Fu[0, 0] = self.dt * cos_theta
        data.Fu[1, 0] = self.dt * sin_theta
        data.Fu[2, 1] = self.dt

        data.cost = self._running_cost_terms_inplace(
            x, u_vec, data.Lx, data.Lu, data.Lxx, data.Luu, data.Lxu
        )


class _UnicycleTerminalModel:
    def __init__(self, crocoddyl, *, env, agent, goal_xy: np.ndarray,
                 theta_ref: float, options: UnicycleTrajOptOptions,
                 circles: np.ndarray, rects: np.ndarray):
        self.crocoddyl = crocoddyl
        self.state = crocoddyl.StateVector(3)
        self.model = crocoddyl.ActionModelAbstract(self.state, 0, 4)
        self.model.calc = self.calc
        self.model.calcDiff = self.calcDiff
        self.env = env
        self.agent = agent
        self.goal_xy = np.asarray(goal_xy[:2], dtype=np.float64)
        self.theta_ref = float(theta_ref)
        self.options = options
        self.circles = circles
        self.rects = rects
        self.env_size = np.asarray(env.size, dtype=np.float64)
        self.obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
        self.boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
        self.goal_boost = 5.0
        self._obstacle_Lx = np.zeros(3, dtype=np.float64)
        self._obstacle_Lxx = np.zeros((3, 3), dtype=np.float64)
        self._boundary_Lx = np.zeros(3, dtype=np.float64)
        self._boundary_Lxx = np.zeros((3, 3), dtype=np.float64)
        self.use_fused_numba_cost = True

    def _terminal_cost_terms_inplace_legacy(self, x, Lx, Lxx):
        dxy0 = float(x[0]) - float(self.goal_xy[0])
        dxy1 = float(x[1]) - float(self.goal_xy[1])
        # dtheta = _angle_diff(float(x[2]), self.theta_ref)

        goal_weight = self.goal_boost * self.options.goal_weight
        # theta_weight = self.goal_boost * self.options.theta_tracking_weight

        cost = 0.5 * goal_weight * (dxy0 * dxy0 + dxy1 * dxy1)
        # cost += 0.5 * theta_weight * dtheta * dtheta

        Lx[0] = goal_weight * dxy0
        Lx[1] = goal_weight * dxy1
        Lx[2] = 0.0 #theta_weight * dtheta

        Lxx[:] = 0.0
        Lxx[0, 0] = goal_weight
        Lxx[1, 1] = goal_weight
        Lxx[2, 2] = 0.0 #theta_weight

        obstacle_cost = _obstacle_penalty_and_derivatives_inplace_numba(
            x,
            self.circles,
            self.rects,
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

        boundary_cost = _boundary_penalty_and_derivatives_inplace_numba(
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

        return cost

    def _terminal_cost_terms_inplace(self, x, Lx, Lxx):
        if self.use_fused_numba_cost:
            return _terminal_cost_terms_inplace_fused_numba(
                x,
                self.goal_xy,
                self.goal_boost * self.options.goal_weight,
                self.circles,
                self.rects,
                self.env_size,
                self.agent.radius,
                self.obstacle_buffer,
                self.boundary_buffer,
                self.options.clearance_margin,
                self.options.obstacle_weight,
                Lx,
                Lxx,
                self._obstacle_Lx,
                self._obstacle_Lxx,
                self._boundary_Lx,
                self._boundary_Lxx,
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


def optimize_unicycle_warm_start(*,
    start: np.ndarray,goal_xy: np.ndarray,
    env,agent,xs_init: np.ndarray,us_init: np.ndarray,
    dt: float, goal_radius: float = 0.25,
    options: Optional[UnicycleTrajOptOptions] = None,
    circles: Optional[np.ndarray] = None,
    rects: Optional[np.ndarray] = None,
    verbose: bool = False) -> UnicycleTrajOptResult:
    
    if options is None:
        options = UnicycleTrajOptOptions()
    if circles is None or rects is None:
        circles, rects = _extract_obstacle_arrays(env)

    # xs_init = np.asarray(xs_init, dtype=np.float64)
    # us_init = np.asarray(us_init, dtype=np.float64)

    if xs_init.ndim != 2 or xs_init.shape[1] != 3:
        raise ValueError(f"xs_init must have shape (N+1, 3), got {xs_init.shape}")
    if us_init.ndim != 2 or us_init.shape[1] != 2:
        raise ValueError(f"us_init must have shape (N, 2), got {us_init.shape}")
    if xs_init.shape[0] != us_init.shape[0] + 1:
        raise ValueError("xs_init must contain exactly one more state than us_init controls")

    running_models: List[object] = []
    for x_ref, u_ref in zip(xs_init[:-1], us_init):
        raw_model = _UnicycleRunningModel(
            crocoddyl,
            dt=dt,
            env=env,
            agent=agent,
            x_ref=x_ref,
            u_ref=u_ref,
            goal_xy=goal_xy,
            options=options,
            circles=circles,
            rects=rects,
        ).model
        running_models.append(raw_model)

    terminal_raw = _UnicycleTerminalModel(
        crocoddyl,
        env=env,
        agent=agent,
        goal_xy=goal_xy,
        theta_ref=float(xs_init[-1, 2]),
        options=options,
        circles=circles,
        rects=rects,
    ).model
    terminal_model = terminal_raw

    for model in running_models:
        model.u_lb = np.array([-agent.max_speed, -agent.max_omega], dtype=np.float64)
        model.u_ub = np.array([agent.max_speed, agent.max_omega], dtype=np.float64)

    problem = crocoddyl.ShootingProblem(np.asarray(start, dtype=np.float64), running_models, terminal_model)
    solver = crocoddyl.SolverBoxFDDP(problem)
    solver.th_stop = options.th_stop
    solver.th_acceptnegstep = options.th_acceptnegstep

    if options.callbacks:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # xs_guess = [np.asarray(x, dtype=np.float64).copy() for x in xs_init]
    # us_guess = [np.asarray(u, dtype=np.float64).copy() for u in us_init]
    xs_guess = [x.copy() for x in xs_init]
    us_guess = [u.copy() for u in us_init]
    success = bool(solver.solve(xs_guess, us_guess, options.max_iter, False, options.init_reg))

    xs_out = np.asarray(solver.xs, dtype=np.float64)
    us_out = np.asarray(solver.us, dtype=np.float64)

    obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
    boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
    feasible, violation_index, violation_kind, signed_dist = _static_boundary_feasibility_numba(
        xs_out,
        circles,
        rects,
        np.asarray(env.size, dtype=np.float64),
        float(agent.radius),
        obstacle_buffer,
        boundary_buffer,
        float(options.feasibility_tolerance),
    )
    if not feasible and verbose:
        x = xs_out[violation_index]
        if violation_kind == 1:
            required_clearance = agent.radius + obstacle_buffer
            print(
                "Optimization infeasible: obstacle collision at index "
                f"{violation_index}, state={x}, signed_distance={signed_dist}, "
                f"required>={required_clearance}"
            )
        elif violation_kind == 2:
            print(
                "Optimization infeasible: boundary violation at index "
                f"{violation_index}, state={x}, "
                f"x_range=[{agent.radius + boundary_buffer}, "
                f"{env.size[0] - agent.radius - boundary_buffer}], "
                f"y_range=[{agent.radius + boundary_buffer}, "
                f"{env.size[1] - agent.radius - boundary_buffer}]"
            )
    if np.linalg.norm(xs_out[-1][:2] - np.asarray(goal_xy[:2], dtype=np.float64)) > float(goal_radius):
        feasible = False
        if verbose:
            print(
                "Optimization infeasible: terminal state outside goal radius, "
                f"final_state={xs_out[-1]}, goal={np.asarray(goal_xy[:2], dtype=np.float64)}, "
                f"distance={np.linalg.norm(xs_out[-1][:2] - np.asarray(goal_xy[:2], dtype=np.float64))}, "
                f"goal_radius={float(goal_radius)}"
            )

    return UnicycleTrajOptResult(
        success=success,
        feasible=feasible,
        optimizer_output_feasible=feasible,
        source="optimized" if feasible else "failed",
        xs=xs_out,
        us=us_out,
        cost=float(solver.cost),
        solver_iters=int(solver.iter),
    )


def _retry_options(options: UnicycleTrajOptOptions):
    return (
        replace(
            options,
            state_tracking_weight=max(options.state_tracking_weight, 0.1),
            theta_tracking_weight=max(options.theta_tracking_weight, 0.01),
            control_tracking_weight=max(options.control_tracking_weight, 0.01),
            obstacle_weight=max(options.obstacle_weight, 500.0),
            max_iter=max(options.max_iter, 100),
            retry_on_infeasible=False,
        ),
        replace(
            options,
            state_tracking_weight=max(options.state_tracking_weight, 1.0),
            theta_tracking_weight=max(options.theta_tracking_weight, 0.02),
            control_tracking_weight=max(options.control_tracking_weight, 0.02),
            obstacle_weight=max(options.obstacle_weight, 2000.0),
            max_iter=max(options.max_iter, 100),
            retry_on_infeasible=False,
        ),
        replace(
            options,
            state_tracking_weight=max(options.state_tracking_weight, 1.0),
            theta_tracking_weight=max(options.theta_tracking_weight, 0.02),
            control_tracking_weight=max(options.control_tracking_weight, 0.02),
            obstacle_weight=max(options.obstacle_weight, 2000.0),
            clearance_margin=max(options.clearance_margin, 1e-3),
            max_iter=max(options.max_iter, 100),
            retry_on_infeasible=False,
        ),
    )


def _warm_start_fallback_result(planner, highres_states, us_init, options, optimizer_result):
    circles = getattr(planner, "static_circular_obstacles", None)
    rects = getattr(planner, "static_rectangular_obstacles", None)
    if circles is None or rects is None:
        circles, rects = _extract_obstacle_arrays(planner.env)

    obstacle_buffer = float(getattr(planner.env, "obstacle_buffer", 0.0))
    boundary_buffer = float(getattr(planner.env, "boundary_buffer", 0.0))
    feasible, _, _, _ = _static_boundary_feasibility_numba(
        highres_states,
        circles,
        rects,
        np.asarray(planner.env.size, dtype=np.float64),
        float(planner.agent.radius),
        obstacle_buffer,
        boundary_buffer,
        float(options.feasibility_tolerance),
    )
    if feasible:
        final_dist = np.linalg.norm(highres_states[-1][:2] - np.asarray(planner.goal[:2], dtype=np.float64))
        feasible = final_dist <= float(planner.goal_radius)

    if not feasible:
        return None

    return UnicycleTrajOptResult(
        success=bool(optimizer_result.success),
        feasible=True,
        optimizer_output_feasible=False,
        source="raw_fallback",
        xs=np.asarray(highres_states, dtype=np.float64).copy(),
        us=np.asarray(us_init, dtype=np.float64).copy(),
        cost=float(getattr(planner, "raw_path_cost", getattr(planner, "path_cost", float("inf")))),
        solver_iters=int(optimizer_result.solver_iters),
    )


def optimize_dbrrt_unicycle_path(planner,*,
    options: Optional[UnicycleTrajOptOptions] = None,) -> UnicycleTrajOptResult:
    
    if options is None:
        options = UnicycleTrajOptOptions()

    highres_states, us_init, _ = planner.get_high_resolution_path_and_actions()
    if highres_states.shape[0] == 0:
        raise ValueError("Planner has no path to optimize")

    # us_init = np.asarray(controls, dtype=np.float64)
    if highres_states.shape[0] != us_init.shape[0] + 1:
        raise ValueError(
            f"Warm start length mismatch: states={highres_states.shape[0]}, controls={us_init.shape[0]}"
        )

    result = optimize_unicycle_warm_start(
        start=planner.start,
        goal_xy=planner.goal,
        env=planner.env,
        agent=planner.agent,
        xs_init=highres_states,
        us_init=us_init,
        dt=float(planner.minimum_time_step),
        goal_radius=float(planner.goal_radius),
        options=options,
        circles=getattr(planner, "static_circular_obstacles", None),
        rects=getattr(planner, "static_rectangular_obstacles", None),
        verbose=bool(getattr(planner, "print_logs", False) or getattr(planner, "debug_flag", False)),
    )

    if not result.feasible and options.retry_on_infeasible:
        for retry_options in _retry_options(options):
            retry_result = optimize_unicycle_warm_start(
                start=planner.start,
                goal_xy=planner.goal,
                env=planner.env,
                agent=planner.agent,
                xs_init=highres_states,
                us_init=us_init,
                dt=float(planner.minimum_time_step),
                goal_radius=float(planner.goal_radius),
                options=retry_options,
                circles=getattr(planner, "static_circular_obstacles", None),
                rects=getattr(planner, "static_rectangular_obstacles", None),
                verbose=bool(getattr(planner, "print_logs", False) or getattr(planner, "debug_flag", False)),
            )
            if retry_result.feasible:
                result = retry_result
                break

    if not result.feasible and options.allow_raw_fallback:
        fallback_result = _warm_start_fallback_result(planner, highres_states, us_init, options, result)
        if fallback_result is not None:
            result = fallback_result

    result.path_view = OptimizedTrajectoryView(planner, result.xs, result.us, result.cost)
    return result
