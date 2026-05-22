import math
import numpy as np
from numba import njit


@njit
def euclidean_distance_numba_with_l(state1, state2, l):
    d = 0.0
    for i in range(l):
        diff = state1[i] - state2[i]
        d += diff * diff
    return math.sqrt(d)


@njit
def euclidean_distance_satisfaction_numba(state1, state2, threshold):
    d = 0.0
    for i in range(len(state1)):
        diff = state1[i] - state2[i]
        d += diff * diff
    dist = math.sqrt(d)
    return dist < threshold, dist


@njit
def validate_random_point_numba(xy, env_start, env_size, circ_obs, rect_obs, obstacle_buffer):
    x, y = xy[0], xy[1]
    if (x < env_start[0] or x > env_start[0] + env_size[0] or
            y < env_start[1] or y > env_start[1] + env_size[1]):
        return False
    for i in range(len(circ_obs)):
        cx, cy, r = circ_obs[i, 0], circ_obs[i, 1], circ_obs[i, 2]
        dx = x - cx
        dy = y - cy
        combined = r + obstacle_buffer
        if dx * dx + dy * dy < combined * combined:
            return False
    for i in range(len(rect_obs)):
        x_min = rect_obs[i, 0] - obstacle_buffer
        x_max = rect_obs[i, 1] + obstacle_buffer
        y_min = rect_obs[i, 2] - obstacle_buffer
        y_max = rect_obs[i, 3] + obstacle_buffer
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return False
    return True


@njit
def is_state_valid_numba(state, agent_radius, env_start, env_size,
                          circ_obs, rect_obs, obstacle_buffer, boundary_buffer):
    x, y = state[0], state[1]
    if (x < env_start[0] + boundary_buffer or
            x > env_start[0] + env_size[0] - boundary_buffer or
            y < env_start[1] + boundary_buffer or
            y > env_start[1] + env_size[1] - boundary_buffer):
        return False
    for i in range(len(circ_obs)):
        cx, cy, r = circ_obs[i, 0], circ_obs[i, 1], circ_obs[i, 2]
        dx = x - cx
        dy = y - cy
        combined = agent_radius + r + obstacle_buffer
        if dx * dx + dy * dy < combined * combined:
            return False
    for i in range(len(rect_obs)):
        x_min, x_max = rect_obs[i, 0], rect_obs[i, 1]
        y_min, y_max = rect_obs[i, 2], rect_obs[i, 3]
        closest_x = min(max(x, x_min), x_max)
        closest_y = min(max(y, y_min), y_max)
        dx = x - closest_x
        dy = y - closest_y
        combined = agent_radius + obstacle_buffer
        if dx * dx + dy * dy < combined * combined:
            return False
    return True


@njit
def is_new_node_valid_numba(path, agent_radius, env_start, env_size,
                             circ_obs, rect_obs, limit_indices, limit_values,
                             obstacle_buffer, boundary_buffer):
    for step_i in range(len(path)):
        if not is_state_valid_numba(path[step_i], agent_radius, env_start, env_size,
                                     circ_obs, rect_obs, obstacle_buffer, boundary_buffer):
            return False
        for j in range(len(limit_indices)):
            val = path[step_i, limit_indices[j]]
            if val < -limit_values[j] or val > limit_values[j]:
                return False
    return True


@njit
def agent_circle(ax, ay, agent_radius, cx, cy, obs_radius):
    """True if the circular agent overlaps a circular obstacle."""
    dx = ax - cx
    dy = ay - cy
    r = agent_radius + obs_radius
    return dx * dx + dy * dy < r * r


@njit
def agent_rectangle(ax, ay, agent_radius, x_min, x_max, y_min, y_max):
    """True if the circular agent overlaps an axis-aligned rectangular obstacle."""
    closest_x = min(max(ax, x_min), x_max)
    closest_y = min(max(ay, y_min), y_max)
    dx = ax - closest_x
    dy = ay - closest_y
    return dx * dx + dy * dy < agent_radius * agent_radius
