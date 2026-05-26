import math
import numpy as np
from numba import njit

from utils import (
    euclidean_distance_numba,
    euclidean_distance_numba_with_l,
    euclidean_distance_satisfaction_numba,
    validate_random_point_numba,
    is_state_valid_numba,
    is_new_node_valid_numba,
)


"""
Bicycle car agent matching the dynamics of FlowMRMP/car_env.py.

Vehicle State  S = (x, y, psi, v, D, delta)
  x, y   — position [m]
  psi    — heading (yaw) [rad]
  v      — forward speed [m/s],   clamped to [0, max_speed]
  D      — throttle/brake command, clamped to [-max_D, max_D]
  delta  — steering angle [rad],  clamped to [-max_delta, max_delta]

Vehicle Control U = (dD, dDelta)
  dD     — rate of change of throttle  ∈ [-max_dD,     max_dD]
  dDelta — rate of change of steering  ∈ [-max_dDelta, max_dDelta]

Physical parameters match car_env.py exactly.
Integration uses RK4 (car_env.py uses Euler).
"""


# ---------------------------------------------------------------------------
# Numba dynamics
# ---------------------------------------------------------------------------

@njit
def flow_car_equation_of_motion_numba(state, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2):
    x, y, psi, v, D, delta = state[0], state[1], state[2], state[3], state[4], state[5]
    dD, dDelta = control[0], control[1]

    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * math.tanh(5.0 * v)

    out = np.empty(6, dtype=np.float64)
    out[0] = v * math.cos(psi + C1 * delta)
    out[1] = v * math.sin(psi + C1 * delta)
    out[2] = v * C2 * delta
    out[3] = (Fxd / m) * math.cos(C1 * delta)
    out[4] = dD
    out[5] = dDelta
    return out


@njit
def flow_car_move_vehicle_numba(state, control, dt,
                                 m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                 max_speed, max_D, max_delta):
    k1 = flow_car_equation_of_motion_numba(state, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k2 = flow_car_equation_of_motion_numba(state + 0.5 * dt * k1, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k3 = flow_car_equation_of_motion_numba(state + 0.5 * dt * k2, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)
    k4 = flow_car_equation_of_motion_numba(state + dt * k3, control, m, C1, C2, Cm1, Cm2, Cr0, Cr2)

    h = dt / 6.0
    next_state = np.empty(6, dtype=np.float64)
    for i in range(6):
        next_state[i] = state[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    # Enforce physical bounds after integration
    if next_state[3] < 0.0:
        next_state[3] = 0.0
    elif next_state[3] > max_speed:
        next_state[3] = max_speed
    if next_state[4] < -max_D:
        next_state[4] = -max_D
    elif next_state[4] > max_D:
        next_state[4] = max_D
    if next_state[5] < -max_delta:
        next_state[5] = -max_delta
    elif next_state[5] > max_delta:
        next_state[5] = max_delta

    return next_state


@njit
def flow_car_get_next_state_numba(state, control, dt, num_steps,
                                   m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                   max_speed, max_D, max_delta, state_length):
    path = np.empty((num_steps, state_length), dtype=np.float64)
    curr = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        curr = flow_car_move_vehicle_numba(curr, control, exec_dt,
                                           m, C1, C2, Cm1, Cm2, Cr0, Cr2,
                                           max_speed, max_D, max_delta)
        path[i] = curr

    return curr, path


@njit
def flow_car_get_distance_covered_numba(start_state, path):
    if len(path) == 0:
        return 0.0
    dx = path[0, 0] - start_state[0]
    dy = path[0, 1] - start_state[1]
    total = math.sqrt(dx * dx + dy * dy)
    for i in range(1, path.shape[0]):
        dx = path[i, 0] - path[i - 1, 0]
        dy = path[i, 1] - path[i - 1, 1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


# ---------------------------------------------------------------------------
# Edge-bundle KD-tree helpers (same pattern as SecondOrderCar)
# ---------------------------------------------------------------------------

@njit
def flow_car_point_translate_function_kd_tree_numba(base_point,
                                                      edge_start_point,
                                                      edge_end_point):
    """
    base_point       : (x_c, y_c, psi_c, v_c, D_c, delta_c)
    edge_start_point : (0, 0, 0, v_s, D_s, delta_s)
    edge_end_point   : (x_e, y_e, psi_e, v_e, D_e, delta_e)

    Returns new final state applying the same edge from base_point.
    """
    psi_c = base_point[2]
    cos_p = math.cos(psi_c)
    sin_p = math.sin(psi_c)

    new_x = edge_end_point[0] * cos_p - edge_end_point[1] * sin_p
    new_y = edge_end_point[0] * sin_p + edge_end_point[1] * cos_p

    return (base_point[0] + new_x,
            base_point[1] + new_y,
            base_point[2] + edge_end_point[2],
            edge_end_point[3],
            edge_end_point[4],
            edge_end_point[5])


@njit
def flow_car_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                       start_states, final_states,
                                       curr_edge_indices, curr_edge_mask,
                                       distance_array):
    n = curr_edge_indices.shape[0]
    num_valid_edges = 0

    for i in range(n):
        if curr_edge_mask[i]:
            distance_array[i] = 1e10
        else:
            edge_idx = curr_edge_indices[i]
            potential = flow_car_point_translate_function_kd_tree_numba(
                closest_tree_point, start_states[edge_idx], final_states[edge_idx])
            distance_array[i] = euclidean_distance_numba(potential, random_point)
            num_valid_edges += 1

    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges


@njit
def flow_car_get_unmasked_kd_tree_edges_no_sorting_numba(closest_tree_point, random_point,
                                                           start_states, final_states,
                                                           curr_edge_indices, curr_edge_mask,
                                                           distance_array):
    n = curr_edge_indices.shape[0]
    good_edges = np.empty(n, dtype=np.int64)
    num_valid_edges = 0

    for i in range(n):
        if not curr_edge_mask[i]:
            good_edges[num_valid_edges] = i
            num_valid_edges += 1

    return good_edges[:num_valid_edges], num_valid_edges


# ---------------------------------------------------------------------------
# FlowCar agent class
# ---------------------------------------------------------------------------

class FlowCar:
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
        self.max_speed  = max_speed
        self.max_D      = max_D
        self.max_delta  = max_delta
        self.max_dD     = max_dD
        self.max_dDelta = max_dDelta
        self.radius     = radius
        self.dt         = dt

        # Physical parameters (match car_env.py)
        self.m   = m
        self.C1  = C1
        self.C2  = C2
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cr0 = Cr0
        self.Cr2 = Cr2

        self.rng = np.random.default_rng(rng_seed)

        self.state_datatype = np.dtype([
            ('x', 'f8'), ('y', 'f8'), ('psi', 'f8'),
            ('v', 'f8'), ('D', 'f8'), ('delta', 'f8')
        ])

        self.state_length             = 6
        self.action_length            = 2
        self.distance_metric_state_size = 2   # nearest-neighbour search on (x, y) only

        # v, D, delta limits used by is_new_node_valid_numba
        self.dynamic_limit_indices = np.array([3, 4, 5], dtype=np.int64)
        self.dynamic_limit_values  = np.array([max_speed, max_D, max_delta], dtype=np.float64)

        # Attributes for normalised state distance (used by dBA*)
        self.distance_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.distance_scales  = np.array([
            1.0,          # x
            1.0,          # y
            np.pi,        # psi
            max_speed,    # v
            max_D,        # D
            max_delta,    # delta
        ], dtype=np.float64)
        self.dbastar_distance_is_angle = np.array(
            [False, False, True, False, False, False], dtype=np.bool_)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def equation_of_motion(self, state, control):
        return flow_car_equation_of_motion_numba(
            state, control,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2)

    def move_vehicle(self, state, control, dt):
        return flow_car_move_vehicle_numba(
            state, control, dt,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2,
            self.max_speed, self.max_D, self.max_delta)

    def get_next_state(self, state, control, dt, num_steps=10):
        return flow_car_get_next_state_numba(
            state, control, dt, num_steps,
            self.m, self.C1, self.C2, self.Cm1, self.Cm2, self.Cr0, self.Cr2,
            self.max_speed, self.max_D, self.max_delta, self.state_length)

    # ------------------------------------------------------------------
    # Distance / sampling
    # ------------------------------------------------------------------

    def get_distance(self, state1, state2):
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)

    def get_random_action(self, rng):
        dD     = rng.uniform(-self.max_dD,     self.max_dD)
        dDelta = rng.uniform(-self.max_dDelta, self.max_dDelta)
        return (dD, dDelta)

    def check_collision(self, base_agent_state, point):
        return self.get_distance(base_agent_state, point) <= self.radius * 2

    # ------------------------------------------------------------------
    # Utility — static methods matching SecondOrderCar interface
    # ------------------------------------------------------------------

    @staticmethod
    def is_state_valid(state, agent_radius, env_size,
                       circ_obs, rect_obs, dyn_obs,
                       obstacle_buffer, dynamic_agent_clearance,
                       boundary_buffer, t):
        return is_state_valid_numba(state, agent_radius, env_size,
                                    circ_obs, rect_obs, dyn_obs,
                                    obstacle_buffer, dynamic_agent_clearance,
                                    boundary_buffer, t)

    @staticmethod
    def is_new_node_valid(path_to_new_state, agent_radius, env_size,
                          circ_obs, rect_obs, dyn_obs,
                          limit_indices, limit_values,
                          obstacle_buffer, dynamic_agent_clearance,
                          boundary_buffer,
                          start_time, time_duration,
                          dt_per_step=0.1):
        return is_new_node_valid_numba(path_to_new_state, agent_radius, env_size,
                                       circ_obs, rect_obs, dyn_obs,
                                       limit_indices, limit_values,
                                       obstacle_buffer, dynamic_agent_clearance,
                                       boundary_buffer,
                                       start_time, time_duration,
                                       dt_per_step)

    @staticmethod
    def get_cost(env, agent, parent_state, a, t, path):
        return t

    @staticmethod
    def get_random_point(env, circular_obstacles, rectangular_obstacles, rng):
        p = rng.uniform(env.env_start, env.size)
        while not validate_random_point_numba(p, circular_obstacles,
                                              rectangular_obstacles,
                                              env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)
        return np.array(p, dtype=np.float64)

    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        return euclidean_distance_satisfaction_numba(state, goal, goal_radius)

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point, edge_end_point):
        return flow_car_point_translate_function_kd_tree_numba(
            base_point, edge_start_point, edge_end_point)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
                           final_states, curr_edge_indices, curr_edge_mask,
                           distance_array):
        return flow_car_sort_kd_tree_edges_numba(
            closest_tree_point, random_point, start_states, final_states,
            curr_edge_indices, curr_edge_mask, distance_array)

    @staticmethod
    def no_sorting_kd_tree_edges(closest_tree_point, random_point, start_states,
                                  final_states, curr_edge_indices, curr_edge_mask,
                                  distance_array):
        return flow_car_get_unmasked_kd_tree_edges_no_sorting_numba(
            closest_tree_point, random_point, start_states, final_states,
            curr_edge_indices, curr_edge_mask, distance_array)

    def get_eb_kd_tree_query(self, state):
        return (state[4], state[5])  # D, delta
