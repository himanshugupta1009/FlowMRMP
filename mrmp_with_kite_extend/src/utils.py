#utils.py

from cmath import rect
import numpy as np
import math
from numba import njit
from numba.typed import List


import numpy as np

class DynamicArray:
    def __init__(self, initial_capacity, datatype):
        self.array = np.empty(initial_capacity, dtype=datatype)
        self.count = 0
        self.dtype = datatype

    def append(self, value):
        if self.count >= self.array.shape[0]:
            self._grow()
        self.array[self.count] = value
        self.count += 1

    def set(self, index, value):
        """Store value at a given index. Auto-grows if index >= capacity."""
        if index < 0:
            raise IndexError("Negative index not allowed")
        while index >= self.array.shape[0]:
            self._grow()
        self.array[index] = value
        # Only increase count if we're filling past the previous count value.
        if index >= self.count:
            self.count = index + 1

    def _grow(self):
        old_cap = self.array.shape[0]
        new_cap = 2 * old_cap
        new_array = np.empty(new_cap, dtype=self.dtype)
        new_array[:old_cap] = self.array
        self.array = new_array

    def get_valid_array(self):
        return self.array[:self.count]

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.count:
            raise IndexError("Index out of bounds")
        return self.array[idx]



def preprocess_circular_obstacles(env):
    circ = []
    for obs in env.obstacles:
        if (obs.__class__.__name__ == "CircularObstacle2D" or 
            obs.__class__.__name__ == "CircularObstacle2DTimed"):
            circ.append((obs.x, obs.y, obs.r))
    if len(circ) == 0:
        return np.empty((0, 3), dtype=np.float64)  # Ensure 2D shape
    return np.array(circ, dtype=np.float64)

def preprocess_rectangular_obstacles(env):
    rect = []
    for obs in env.obstacles:
        if obs.__class__.__name__ == "RectangleObstacle2D":
            half_w, half_h = obs.w / 2, obs.h / 2
            rect.append((obs.x - half_w, obs.x + half_w,
                          obs.y - half_h, obs.y + half_h))
    if len(rect) == 0:
        return np.empty((0, 4), dtype=np.float64)  # Ensure 2D shape
    return np.array(rect, dtype=np.float64)

def preprocess_spherical_obstacles_3d(env):
    spheres = []
    for obs in env.obstacles:
        if obs.__class__.__name__ == "SphericalObstacle3D":
            spheres.append((obs.x, obs.y, obs.z, obs.r))
    if len(spheres) == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(spheres, dtype=np.float64)

def preprocess_cuboid_obstacles_3d(env):
    boxes = []
    for obs in env.obstacles:
        if obs.__class__.__name__ == "CuboidObstacle3D":
            half_l, half_w, half_h = obs.l / 2, obs.w / 2, obs.h / 2
            boxes.append((obs.x - half_l, obs.x + half_l,
                          obs.y - half_w, obs.y + half_w,
                          obs.z - half_h, obs.z + half_h))
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float64)
    return np.array(boxes, dtype=np.float64)

@njit(inline='always')
def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))

@njit(inline='always')
def wrap_between_0_and_2pi(x):
    twopi = 2.0 * np.pi
    y = x % twopi
    if y == twopi:
        return 0.0
    return y

# @njit
# def wrap_between_0_and_2pi(angle):
#     """
#     Wrap angle into [0, 2π) without using Python modulo.
#     Works under Numba.
#     """
#     twopi = 2.0 * np.pi
#     # use floor for stable wrapping
#     return angle - np.floor(angle / twopi) * twopi

def get_dtype_from_input(t):
    data_type = [(f'field_{i}', type(element)) for i, element in enumerate(t)]
    return np.dtype(data_type)
"""
t = (1, 2.5, 3)
c = get_dtype_from_input(t)

"""


@njit(inline='always') 
def get_distance_covered_numba(v, t):
    """
    Calculate the distance covered by an agent given its linear velocity v,
    and time step t.
    """
    return abs(v) * t

@njit
def find_roundoff_decimal_digits(x):
    if x == 0:
        return 0
    else:
        l = 0
        while x < 1:
            x *= 10
            l += 1
        return l


@njit
def squared_distance_numba(a, b):
    acc = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        acc += diff * diff
    return acc

@njit
def squared_distance_satisfaction_numba(a, b, threshold, length):
    acc = 0.0
    for i in range(length):
        diff = a[i] - b[i]
        acc += diff * diff
    return acc <= threshold

@njit(inline='always')
def euclidean_distance(p1, p2):
    return euclidean_distance_numba(p1,p2)

@njit(inline='always')
def euclidean_distance_numba(a, b):
    """
    Calculate the Euclidean distance between two vectors a and b where it
    iterates over the length of the b vector.
    """
    acc = 0.0
    for i in range(len(b)):
        diff = a[i] - b[i]
        acc += diff * diff
    return math.sqrt(acc)

@njit(inline='always')
def euclidean_distance_numba_with_l(a, b, c):
    acc = 0.0
    for i in range(c):
        diff = a[i] - b[i]
        acc += diff * diff
    return math.sqrt(acc)

@njit(inline='always')
def euclidean_distance_satisfaction_numba(a, b, threshold):
    """
    Calculate the Euclidean distance between two vectors a and b where it
    iterates over the length of the b vector.
    """
    acc = 0.0
    for i in range(len(b)):
        diff = a[i] - b[i]
        acc += diff * diff
    
    # return math.sqrt(acc) <= threshold
    d = math.sqrt(acc)
    return d <= threshold, d

@njit(inline='always')
def euclidean_distance_satisfaction_numba_with_l(a, b, c, threshold):
    """
    Calculate the Euclidean distance between two vectors a and b where it
    iterates over the value c.
    """
    acc = 0.0
    for i in range(c):
        diff = a[i] - b[i]
        acc += diff * diff
    
    # return math.sqrt(acc) <= threshold
    d = math.sqrt(acc)
    return d <= threshold, d

@njit
def get_nearest_index(states, count, query):
    best_idx = -1
    best_dist = 1e10
    for i in range(count):
        dist = squared_distance_numba(states[i], query)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

@njit
def get_active_nearest_index(states, active, count, query):
    best_idx = -1
    best_dist = 1e10
    for i in range(count):
        if active[i]:
            dist = squared_distance_numba(states[i], query)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx

@njit
def has_nearby_state(states, count, query, length, threshold):
    threshold_sq = threshold * threshold
    for i in range(count):
        acc = 0.0
        for j in range(length):
            diff = states[i, j] - query[j]
            acc += diff * diff
            if acc > threshold_sq:
                break
        if acc <= threshold_sq:
            return True
    return False


##########################################
#2d environment and collision functions
##########################################
@njit
def point_circle_collision(px, py, pr, cx, cy, cr):
    """Determines whether two circles are in a collision state

    Args:
        px (float): first circle x pos
        py (float): first circle y pos
        pr (float): first circle rad
        cx (float): second circle x pos
        cy (float): second circle y pos
        cr (float): second circle radius

    Returns:
        bool: True if no collision, false else
    """
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy < (cr + pr) * (cr + pr)

@njit
def point_rectangle_collision(px, py, x_min, x_max, y_min, y_max):
    closest_x = min(max(px, x_min), x_max)
    closest_y = min(max(py, y_min), y_max)
    dx = px - closest_x
    dy = py - closest_y
    return dx * dx + dy * dy < 1e-8  # epsilon collision radius

@njit
def circle_rectangle_collision(px, py, x_min, x_max, y_min, y_max, radius):
    closest_x = min(max(px, x_min), x_max)
    closest_y = min(max(py, y_min), y_max)
    dx = px - closest_x
    dy = py - closest_y
    return dx * dx + dy * dy < radius * radius

@njit
def check_dynamic_collisions_to_end(state, agent_radius, dyn_obs, 
                             dynamic_agent_clearance, start_t, timestep):
    """
    Call to make sure that an agent's final position does not collide with
    any dynamic obstacles after its planned time.
    
    """
    #Array index for the second dimension, the timestep
    index = round(start_t / timestep) + 1 
    for dyn_ob in dyn_obs: # ie. previously planned agent (paths)
        #For steps in the previous agent's path beyond the current 
        #agent's planned time
        for i in range(index, dyn_ob.shape[0]): 
            if point_circle_collision(state[0], state[1], agent_radius, 
                                      dyn_ob[i][0], # previous agent's x pos
                                      dyn_ob[i][1], # previous agent's y pos
                                      dyn_ob[i][2] + dynamic_agent_clearance): # previous agent's radius + clearance
                return True
    return False

@njit
def check_dynamic_collisions(state, agent_radius, dyn_obs, 
                             dynamic_agent_clearance, t, timestep):
    
    #Array index for the second dimension, the timestep
    #For t=0.3, this will be 3 if timestep is 0.1, meaning we will check the 
    #position of the previous agent at t=0.3 (the current time) for collision 
    #with the current agent's position at t=0.3. 
    #If the previous agent's path only goes up to t=0.2, then we 
    #will check its position at t=0.2 for collision with the current agent's
    #position at t=0.3, since we assume it will remain at that position 
    #after t=0.2.
    index = round(t / timestep) 

    for i in range(len(dyn_obs)): # loop over each dynamic obstacle
        current_index = index
        #Check if index is larger than the available time
        if index >= dyn_obs[i].shape[0]:  
            current_index = dyn_obs[i].shape[0] - 1
        if point_circle_collision(state[0], state[1], agent_radius, 
                                  dyn_obs[i][current_index][0], 
                                  dyn_obs[i][current_index][1], 
                                  dyn_obs[i][current_index][2] + dynamic_agent_clearance):
            return True
    return False
            
@njit
def state_respects_dynamic_limits_numba(state, limit_indices, limit_values):
    for k in range(limit_indices.shape[0]):
        idx = limit_indices[k]
        if abs(state[idx]) > limit_values[k]:
            return False
    return True

@njit
def is_state_valid_numba(state, agent_radius, env_size,
                         circ_obs, rect_obs, dyn_obs,
                         obstacle_buffer, dynamic_agent_clearance,
                         boundary_buffer, t, 
                         timestep = 0.1):
    x, y = state[0], state[1]
    if (x < (agent_radius + boundary_buffer) or
        x > (env_size[0] - agent_radius - boundary_buffer) or
        y < (agent_radius + boundary_buffer) or
        y > (env_size[1] - agent_radius - boundary_buffer)):
        return False

    for i in range(circ_obs.shape[0]):
        cx, cy, cr = circ_obs[i]
        if point_circle_collision(x, y, agent_radius, cx, cy, cr + obstacle_buffer):
            return False

    for i in range(rect_obs.shape[0]):
        if circle_rectangle_collision(x, y, rect_obs[i, 0], rect_obs[i, 1],
                                       rect_obs[i, 2], rect_obs[i, 3], 
                                       agent_radius + obstacle_buffer):
            return False
 
    return not check_dynamic_collisions(state, agent_radius, dyn_obs, dynamic_agent_clearance, t, timestep)

@njit
def validate_random_point_numba(point, circ_obs, rect_obs, obstacle_buffer):
    x, y = point[0], point[1]
    for i in range(circ_obs.shape[0]):
        cx, cy, cr = circ_obs[i]
        if point_circle_collision(x, y, 0.0, cx, cy, cr + obstacle_buffer):
            return False

    for i in range(rect_obs.shape[0]):
        if circle_rectangle_collision(x, y, rect_obs[i, 0], rect_obs[i, 1],
                                       rect_obs[i, 2], rect_obs[i, 3], 
                                       obstacle_buffer):
            return False
 
    return True

@njit
def is_new_node_valid_numba(path, agent_radius, env_size,
                            circ_obs, rect_obs, dyn_obs,
                            limit_indices, limit_values,
                            obstacle_buffer, dynamic_agent_clearance,
                            boundary_buffer,
                            start_time, time_duration,
                            dt_per_step=0.1):
    t = start_time + dt_per_step
    for i in range(path.shape[0]):
        state = path[i]
        if not state_respects_dynamic_limits_numba(state, limit_indices, limit_values):
            return False
        if not is_state_valid_numba(state, agent_radius, env_size,
                                    circ_obs, rect_obs, dyn_obs,
                                    obstacle_buffer, dynamic_agent_clearance,
                                    boundary_buffer, t, dt_per_step):
            return False
        t += dt_per_step
    return True


##########################################
#3d environment and collision functions
##########################################
@njit
def point_sphere_collision(px, py, pz, pr, cx, cy, cz, cr):
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return dx * dx + dy * dy + dz * dz < (cr + pr) * (cr + pr)

@njit
def sphere_box_collision(px, py, pz, x_min, x_max, y_min, y_max, z_min, z_max, radius):
    closest_x = min(max(px, x_min), x_max)
    closest_y = min(max(py, y_min), y_max)
    closest_z = min(max(pz, z_min), z_max)
    dx = px - closest_x
    dy = py - closest_y
    dz = pz - closest_z
    return dx * dx + dy * dy + dz * dz < radius * radius

@njit
def check_dynamic_collisions_to_end_3d(state, agent_radius, dyn_obs, 
                             dynamic_agent_clearance, start_t, timestep):
    """
    Call to make sure that an agent's final position does not collide with
    any dynamic obstacles after its planned time.
    """
    #Array index for the second dimension, the timestep
    index = round(start_t / timestep) + 1 
    for dyn_ob in dyn_obs: # ie. previously planned agent (paths)
        #For steps in the previous agent's path beyond the 
        #current agent's planned time
        for i in range(index, dyn_ob.shape[0]): 
            if point_sphere_collision(state[0], state[1], state[2],
                                  agent_radius, 
                                  dyn_ob[i][0], # previous agent x pos
                                  dyn_ob[i][1], # previous agent y pos
                                  dyn_ob[i][2], # previous agent z pos
                                  dyn_ob[i][3] + dynamic_agent_clearance): # previous agent radius + clearance
                return True
    return False

@njit
def check_dynamic_collisions_3d(state, agent_radius, dyn_obs, 
                                dynamic_agent_clearance, t, timestep):
    """
    Check if two spheres collide within a 
    
    :param state: Description
    :param agent_radius: Description
    :param dyn_obs: Description
    :param dynamic_agent_clearance: Description
    :param t: Description
    :param timestep: Description
    """
    index = round(t / timestep)
    for i in range(len(dyn_obs)):
        current_index = index
        if index >= dyn_obs[i].shape[0]:
            current_index = dyn_obs[i].shape[0] - 1
        if point_sphere_collision(state[0], state[1], state[2],
                                  agent_radius, 
                                  dyn_obs[i][current_index][0], # previous agent x pos
                                  dyn_obs[i][current_index][1], # previous agent y pos
                                  dyn_obs[i][current_index][2], # previous agent z pos
                                  dyn_obs[i][current_index][3] + dynamic_agent_clearance): # previous agent radius + clearance
            return True
    return False

@njit
def is_state_valid_3d_numba(state, agent_radius, env_size,
                            sph_obs, box_obs, dyn_obs,
                            obstacle_buffer, dynamic_agent_clearance,
                            boundary_buffer, t, 
                            timestep = 0.1):
    x, y, z = state[0], state[1], state[2]
    if (x < (agent_radius + boundary_buffer) or
        x > (env_size[0] - agent_radius - boundary_buffer) or
        y < (agent_radius + boundary_buffer) or
        y > (env_size[1] - agent_radius - boundary_buffer) or
        z < (agent_radius + boundary_buffer) or
        z > (env_size[2] - agent_radius - boundary_buffer)):
        # print("Boundary collision")
        return False

    for i in range(sph_obs.shape[0]):
        cx, cy, cz, cr = sph_obs[i]
        if point_sphere_collision(x, y, z, agent_radius, cx, cy, cz, cr + obstacle_buffer):
            # print("Sphere collision")
            return False

    for i in range(box_obs.shape[0]):
        if sphere_box_collision(x, y, z,
                                box_obs[i, 0], box_obs[i, 1],
                                box_obs[i, 2], box_obs[i, 3],
                                box_obs[i, 4], box_obs[i, 5],
                                agent_radius + obstacle_buffer):
            # print("Box collision")
            return False
 
    return not check_dynamic_collisions_3d(state, agent_radius, dyn_obs, dynamic_agent_clearance, t, timestep)

@njit
def validate_random_point_3d_numba(point, sph_obs, box_obs, obstacle_buffer):
    x, y, z = point[0], point[1], point[2]
    for i in range(sph_obs.shape[0]):
        cx, cy, cz, cr = sph_obs[i]
        if point_sphere_collision(x, y, z, 0.0, cx, cy, cz, cr + obstacle_buffer):
            return False

    for i in range(box_obs.shape[0]):
        if sphere_box_collision(x, y, z,
                                box_obs[i, 0], box_obs[i, 1],
                                box_obs[i, 2], box_obs[i, 3],
                                box_obs[i, 4], box_obs[i, 5],
                                obstacle_buffer):
            return False
 
    return True

@njit
def is_new_node_valid_3d_numba(path, agent_radius, env_size,
                                sph_obs, box_obs, dyn_obs,
                                limit_indices, limit_values,
                                obstacle_buffer, dynamic_agent_clearance,
                                boundary_buffer,
                                start_time, time_duration,
                                dt_per_step=0.1):
    t = start_time + dt_per_step
    for i in range(path.shape[0]):
        state = path[i]
        if not state_respects_dynamic_limits_numba(state, limit_indices, limit_values):
            return False
        if not is_state_valid_3d_numba(state, agent_radius, env_size,
                                    sph_obs, box_obs, dyn_obs,
                                    obstacle_buffer, dynamic_agent_clearance,
                                    boundary_buffer, 
                                    t, dt_per_step):
            return False
        t += dt_per_step
    return True


@njit
def compact_matrix(matrix, ids, valid_ids_set, count):
    write_index = 0
    for i in range(count):
        node_id = ids[i]
        if valid_ids_set[node_id] == 1:
            matrix[write_index] = matrix[i]
            ids[write_index] = node_id
            write_index += 1
    return write_index

@njit
def build_high_res_path(path_length, goal_node_id, start_state, state_length, 
                        node_parent_ids, node_paths):
    path_states = np.empty((path_length, state_length), dtype=np.float64)
    curr_index = path_length - 1
    node_id = goal_node_id

    while node_id != 0:
        path_to_node = node_paths[node_id]
        len_path = path_to_node.shape[0]
        start_index = curr_index - len_path + 1
        path_states[start_index:curr_index+1, :] = path_to_node
        curr_index -= len_path
        node_id = node_parent_ids[node_id]

    # Fill the remaining with start state
    for i in range(curr_index + 1):
        path_states[i, :] = start_state

    return path_states

@njit
def copy_numba_list(lst):
    new_lst = List()
    for i in range(len(lst)):
        # This appends a reference to the original array
        new_lst.append(lst[i])
    return new_lst




# Functions for computing general distance over the entoire state.

@njit
def wrapped_angle_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi

@njit
def normalized_state_distance_sq(a, b, indices, scales, is_angle):
    acc = 0.0
    for k in range(indices.shape[0]):
        idx = indices[k]
        if is_angle[k]:
            diff = wrapped_angle_diff(a[idx], b[idx])
        else:
            diff = a[idx] - b[idx]
        nd = diff / scales[k]
        acc += nd * nd
    return acc

@njit
def normalized_state_distance(a, b, indices, scales, is_angle):
    return np.sqrt(normalized_state_distance_sq(a, b, indices, scales, is_angle))



def verify_rollout_consistency(path_view, *, atol=1e-4, rtol=1e-4):
    states, controls, timesteps = path_view.get_high_resolution_path_and_actions()
    reported_highres = path_view.get_high_resolution_path_numpy_array()

    assert states.shape == reported_highres.shape, (
        f"High-res state shape mismatch: actions-path states={states.shape}, "
        f"highres={reported_highres.shape}"
    )
    assert np.allclose(states, reported_highres, atol=atol, rtol=rtol), (
        "Reported states from high-resolution accessors differ"
    )

    assert states.shape[0] == controls.shape[0] + 1, (
        f"High-res warm start mismatch: states={states.shape[0]}, controls={controls.shape[0]}"
    )

    rollout = np.empty_like(states)
    rollout[0] = np.asarray(path_view.start, dtype=np.float64)
    curr = rollout[0].copy()

    for i, (u, dt) in enumerate(zip(controls, timesteps)):
        next_state, _ = path_view.agent.get_next_state(curr, u, float(dt), num_steps=1)
        rollout[i + 1] = next_state
        curr = next_state

    diff = rollout - states
    if diff.shape[1] >= 3:
        diff[:, 2] = (diff[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

    max_abs_err = float(np.max(np.abs(diff)))
    print("\n===== Rollout consistency =====")
    print("Rollout shape:", rollout.shape)
    print("max_abs_err:", max_abs_err)
    print("Rolled final state:", rollout[-1])
    print("Reported final state:", states[-1])

    assert np.all(np.abs(diff) <= (atol + rtol * np.abs(states))), (
        f"Returned path is not rollout-consistent; max_abs_err={max_abs_err}"
    )
    print(f"[OK] returned states are rollout-consistent, max_abs_err={max_abs_err:.3e}")
