import math
import numpy as np
from numba import njit

from utils import (
    euclidean_distance_numba_with_l,
    euclidean_distance_satisfaction_numba,
    validate_random_point_numba,
    is_state_valid_numba,
    is_new_node_valid_numba,
)


# ---------------------------------------------------------------------------
# Numba-accelerated bicycle dynamics
# ---------------------------------------------------------------------------

@njit
def bicycle_eom_numba(state, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2):
    """
    Bicycle model equations of motion.

    State  : [x, y, psi, v, D, delta]
    Control: [dD, dDelta]

    Returns state_dot as a length-6 array.
    """
    x, y, psi, v, D, delta = state[0], state[1], state[2], state[3], state[4], state[5]
    dD, dDelta = control[0], control[1]

    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * math.tanh(5.0 * v)

    out = np.empty(6, dtype=np.float64)
    out[0] = v * math.cos(psi + C1 * delta)          # dx/dt
    out[1] = v * math.sin(psi + C1 * delta)          # dy/dt
    out[2] = v * C2 * delta                           # dpsi/dt
    out[3] = (Fxd / m) * math.cos(C1 * delta)        # dv/dt
    out[4] = dD                                       # dD/dt
    out[5] = dDelta                                   # ddelta/dt
    return out


@njit
def bicycle_move_vehicle_numba(state, control, dt,
                                m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                max_speed, max_D, max_delta):
    """
    One RK4 integration step of the bicycle model.
    State bounds (v, D, delta) are enforced after integration.
    """
    k1 = bicycle_eom_numba(state, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k2 = bicycle_eom_numba(state + 0.5 * dt * k1, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k3 = bicycle_eom_numba(state + 0.5 * dt * k2, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k4 = bicycle_eom_numba(state + dt * k3, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)

    h = dt / 6.0
    next_state = np.empty(6, dtype=np.float64)
    for i in range(6):
        next_state[i] = state[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    # Enforce physical bounds
    if next_state[3] < 0.0:        # v >= 0
        next_state[3] = 0.0
    elif next_state[3] > max_speed:
        next_state[3] = max_speed
    if next_state[4] < -max_D:     # |D| <= max_D
        next_state[4] = -max_D
    elif next_state[4] > max_D:
        next_state[4] = max_D
    if next_state[5] < -max_delta: # |delta| <= max_delta
        next_state[5] = -max_delta
    elif next_state[5] > max_delta:
        next_state[5] = max_delta

    return next_state


@njit
def bicycle_get_next_state_numba(state, control, dt, num_steps,
                                  m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                  max_speed, max_D, max_delta):
    """
    Integrate the bicycle model for num_steps sub-steps of dt/num_steps each.

    Returns
    -------
    final_state : np.ndarray (6,)
    path        : np.ndarray (num_steps, 6)  — states at each sub-step
    """
    path = np.empty((num_steps, 6), dtype=np.float64)
    curr = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        curr = bicycle_move_vehicle_numba(curr, control, exec_dt,
                                          m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                          max_speed, max_D, max_delta)
        path[i] = curr

    return curr, path


# ---------------------------------------------------------------------------
# BicycleCar agent
# ---------------------------------------------------------------------------

class BicycleCar:
    """
    Pure-dynamics bicycle car agent.

    Mirrors the vehicle model in FlowMRMP/car_env.py but separated from any
    environment or gym interface — the same design as MAPF's SecondOrderCar.

    State  (6D): [x, y, psi, v, D, delta]
      x, y   — position [m]
      psi    — heading (yaw) [rad]
      v      — forward speed [m/s], clamped to [0, max_speed]
      D      — throttle/brake command, clamped to [-max_D, max_D]
      delta  — steering angle [rad], clamped to [-max_delta, max_delta]

    Action (2D): [dD, dDelta]
      dD     — rate of throttle change  ∈ [-max_dD,     max_dD]
      dDelta — rate of steering change  ∈ [-max_dDelta, max_dDelta]

    Default parameters match car_env.py exactly.
    Integration is upgraded to RK4 (car_env.py uses Euler).
    """

    def __init__(self, *,
                 agent_id=-1,
                 max_speed=10.0,
                 max_D=1.0,
                 max_delta=0.40,
                 max_dD=10.0,
                 max_dDelta=2.0,
                 radius=0.3,
                 m=0.043,
                 C1=0.5,
                 C2=15.5,
                 Cm1=0.28,
                 Cm2=0.05,
                 Cr0=0.011,
                 Cr2=0.006,
                 dt=0.02,
                 rng_seed=42):

        self.id = agent_id
        self.max_speed = max_speed
        self.max_D = max_D
        self.max_delta = max_delta
        self.max_dD = max_dD
        self.max_dDelta = max_dDelta
        self.radius = radius
        self.dt = dt
        self.rng = np.random.default_rng(rng_seed)

        # Physical parameters (identical to car_env.py)
        self.m = m
        self.C1 = C1
        self.C2 = C2
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cr0 = Cr0
        self.Cr2 = Cr2

        self.state_length = 6
        self.action_length = 2
        # Nearest-neighbour search in RRT uses only (x, y)
        self.distance_metric_state_size = 2

        # Dynamic limits enforced in is_new_node_valid: |D| <= max_D, |delta| <= max_delta
        # (v is clamped during integration so no separate limit check needed)
        self.dynamic_limit_indices = np.array([4, 5], dtype=np.int64)
        self.dynamic_limit_values = np.array([max_D, max_delta], dtype=np.float64)

        # Default zero initial state
        self.x0 = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def equation_of_motion(self, state, control):
        return bicycle_eom_numba(
            state, control,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2)

    def move_vehicle(self, state, control, dt):
        """One RK4 step."""
        return bicycle_move_vehicle_numba(
            state, control, dt,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2,
            self.max_speed, self.max_D, self.max_delta)

    def get_next_state(self, state, control, dt, num_steps=10):
        """
        Propagate state forward by dt, subdivided into num_steps sub-steps.

        Returns
        -------
        final_state : np.ndarray (6,)
        path        : np.ndarray (num_steps, 6)
        """
        return bicycle_get_next_state_numba(
            state, control, dt, num_steps,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2,
            self.max_speed, self.max_D, self.max_delta)

    # ------------------------------------------------------------------
    # Distance / sampling
    # ------------------------------------------------------------------

    def get_distance(self, state1, state2):
        """Euclidean distance in (x, y) only."""
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)

    def get_random_action(self, rng=None):
        r = rng if rng is not None else self.rng
        dD = r.uniform(-self.max_dD, self.max_dD)
        dDelta = r.uniform(-self.max_dDelta, self.max_dDelta)
        return np.array([dD, dDelta], dtype=np.float64)

    def get_random_point(self, env, rng=None):
        """
        Sample a random, collision-free full state from the environment.

        (x, y) is sampled until it avoids all obstacles.
        Remaining dimensions (psi, v, D, delta) are uniformly sampled.
        """
        r = rng if rng is not None else self.rng
        xy = r.uniform(env.env_start, env.env_start + env.size)
        while not validate_random_point_numba(xy, env.env_start, env.size,
                                              env.static_circular_obstacles,
                                              env.static_rectangular_obstacles,
                                              env.obstacle_buffer):
            xy = r.uniform(env.env_start, env.env_start + env.size)

        psi   = r.uniform(-np.pi, np.pi)
        v     = r.uniform(0.0, self.max_speed)
        D     = r.uniform(-self.max_D, self.max_D)
        delta = r.uniform(-self.max_delta, self.max_delta)
        return np.array([xy[0], xy[1], psi, v, D, delta], dtype=np.float64)

    # ------------------------------------------------------------------
    # Validity checks (static — delegate to numba functions in utils.py)
    # ------------------------------------------------------------------

    @staticmethod
    def is_state_valid(state, agent_radius, env_start, env_size,
                        circ_obs, rect_obs, obstacle_buffer, boundary_buffer):
        return is_state_valid_numba(state, agent_radius, env_start, env_size,
                                     circ_obs, rect_obs, obstacle_buffer, boundary_buffer)

    @staticmethod
    def is_new_node_valid(path, agent_radius, env_start, env_size,
                           circ_obs, rect_obs, limit_indices, limit_values,
                           obstacle_buffer, boundary_buffer):
        return is_new_node_valid_numba(path, agent_radius, env_start, env_size,
                                        circ_obs, rect_obs, limit_indices, limit_values,
                                        obstacle_buffer, boundary_buffer)

    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent=None):
        """True if the (x, y) position is within goal_radius of the goal."""
        reached, _ = euclidean_distance_satisfaction_numba(state[:2], goal[:2], goal_radius)
        return reached
