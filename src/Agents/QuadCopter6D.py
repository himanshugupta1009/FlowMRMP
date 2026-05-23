import math
import numpy as np
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    euclidean_distance_satisfaction_numba_with_l, validate_random_point_3d_numba,\
    is_new_node_valid_3d_numba, is_state_valid_3d_numba, get_distance_covered_numba


@njit
def quad6D_eom_numba(state, control, max_speed):
    # state: [x,y,z,vx,vy,vz]; control: [ax,ay,az]
    x, y, z, vx, vy, vz = state
    ax, ay, az = control

    # clamp velocities
    # vx = clamp(vx, -max_speed, max_speed)
    # vy = clamp(vy, -max_speed, max_speed)
    # vz = clamp(vz, -max_speed, max_speed)

    # dx = vx
    # dy = vy
    # dz = vz
    # dvx = ax
    # dvy = ay
    # dvz = az
    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


@njit
def quad6D_move_vehicle_numba(state, control, dt, max_speed):
    # RK4 step
    k1 = quad6D_eom_numba(state, control, max_speed)
    k2 = quad6D_eom_numba(state + 0.5 * dt * k1, control, max_speed)
    k3 = quad6D_eom_numba(state + 0.5 * dt * k2, control, max_speed)
    k4 = quad6D_eom_numba(state + dt * k3, control, max_speed)

    next_state = np.empty_like(state)
    tmp = (dt / 6.0)
    for i in range(state.shape[0]):
        next_state[i] = state[i] + tmp * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    # next_state[3] = clamp(next_state[3], -max_speed, max_speed)
    # next_state[4] = clamp(next_state[4], -max_speed, max_speed)
    # next_state[5] = clamp(next_state[5], -max_speed, max_speed)

    return next_state


@njit
def quad6D_get_next_state_numba(state, control, dt, num_steps, max_speed, state_length):
    # Pre-allocate path
    path = np.empty((num_steps, state_length), dtype=np.float64)
    curr_state = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        new_state = quad6D_move_vehicle_numba(curr_state, control, exec_dt, max_speed)
        path[i] = new_state
        curr_state = new_state

    return curr_state, path


@njit 
def quad6D_get_distance_covered_numba(start_state, path):
    if path.shape[0] == 0:
        return 0.0
    
    dx = path[0,0] - start_state[0]
    dy = path[0,1] - start_state[1]
    dz = path[0,2] - start_state[2]
    total_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    for i in range(1, path.shape[0]):
        dx = path[i,0] - path[i-1,0]
        dy = path[i,1] - path[i-1,1]
        dz = path[i,2] - path[i-1,2]
        total_distance += np.sqrt(dx*dx + dy*dy + dz*dz)
    return total_distance


@njit
def quad6D_point_translate_function_kd_tree_numba(base_point,
                                    edge_start_point, edge_end_point):
    """
    Apply a precomputed edge to a new base state.
    base_point: (x,y,z,vx,vy,vz)
    edge_start_point: (0,0,0,vx_s,vy_s,vz_s)
    edge_end_point: (dx,dy,dz,vx_e,vy_e,vz_e)
    Returns: new_point: (x+dx, y+dy, z+dz, vx_e, vy_e, vz_e)
    """
    return (
        base_point[0] + edge_end_point[0],
        base_point[1] + edge_end_point[1],
        base_point[2] + edge_end_point[2],
        edge_end_point[3],
        edge_end_point[4],
        edge_end_point[5],
    )
    #Returing this as a numpy array slows kinoTIEBRRT down, so don't - Himanshu

@njit
def quad_xyz_dist_after_edge(base_point, edge_end_point, random_point):
    dxp = (base_point[0] + edge_end_point[0]) - random_point[0]
    dyp = (base_point[1] + edge_end_point[1]) - random_point[1]
    dzp = (base_point[2] + edge_end_point[2]) - random_point[2]
    return math.sqrt(dxp*dxp + dyp*dyp + dzp*dzp)


@njit
def quad_sort_kd_tree_edges_numba(closest_tree_point,random_point,
                                start_states,final_states,curr_edge_indices,
                                curr_edge_mask,distance_array):
    
    n = curr_edge_indices.shape[0]
    num_valid_edges = 0

    for i in range(n):
        if curr_edge_mask[i]:
            #That edge has been explored before
            distance_array[i] = 1e10
        else:
            edge_idx = curr_edge_indices[i]
            # potential_new_point = quad6D_point_translate_function_kd_tree_numba(
            #     closest_tree_point,start_states[edge_idx],final_states[edge_idx])
            # # distance in (x,y,z)
            # dist = euclidean_distance_numba_with_l(potential_new_point, 
            #                                        random_point, 3)
            dist = quad_xyz_dist_after_edge(closest_tree_point,
                                    final_states[edge_idx], random_point)
            distance_array[i] = dist
            num_valid_edges += 1

    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges

@njit
def quad_no_sorting_kd_tree_edges_numba(closest_tree_point, random_point,
                                start_states, final_states, curr_edge_indices,
                                curr_edge_mask, distance_array):
    n = curr_edge_indices.shape[0]
    num_valid_edges = 0
    good_edges = np.empty(n, dtype=np.int64)

    for i in range(n):
        if curr_edge_mask[i] != True:
            good_edges[num_valid_edges] = i
            num_valid_edges += 1

    #Fisher-Yates shuffle. Use it if needed to randomize the order of edge
    #exploration when not sorting by distance.
    # for i in range(num_valid_edges - 1, 0, -1):
    #     j = np.random.randint(0, i + 1)
    #     tmp = good_edges[i]
    #     good_edges[i] = good_edges[j]
    #     good_edges[j] = tmp

    return good_edges[:num_valid_edges], num_valid_edges


class QuadCopter6D:
    '''
    6D Quadcopter planning model (double-integrator in Cartesian coords).

    State: [x, y, z, vx, vy, vz]
    Control: [ax, ay, az]
    '''
    def __init__(self, *, 
                agent_id=1,
                max_speed=0.5,
                max_acceleration=2.0,
                radius=0.25, 
                rng_seed=77
                ):
        
        self.id = agent_id
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.radius = float(radius)
        self.rng = np.random.default_rng(int(rng_seed))

        self.state_length = 6
        self.action_length = 3
        # use 3D for nearest-neighbor & random-point sampling
        self.distance_metric_state_size = 3
        # environment reference and cached geometry (populated lazily)
        self._env = None
        self._cache_valid = False
        self._cached_spheres = np.empty((0, 4), dtype=np.float64)
        self._cached_boxes = np.empty((0, 6), dtype=np.float64)
        self._env_start = np.zeros(2, dtype=np.float64)
        self._env_size = np.ones(2, dtype=np.float64)
        self._env_extent = np.ones(2, dtype=np.float64)
        self._low_xy = np.zeros(2, dtype=np.float64)
        self._high_xy = np.zeros(2, dtype=np.float64)
        self._boundary_buffer = 0.0
        self._obstacle_buffer = 0.0
        self.z_bounds = (0.0, 5.0)  # default z bounds

        self.dynamic_limit_indices = np.array([3, 4, 5], dtype=np.int64)
        self.dynamic_limit_values = np.array([self.max_speed, self.max_speed, self.max_speed],
                                    dtype=np.float64)
        
        # attributes for total state distance metric (x, y, z, vx, vy, vz)
        self.distance_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.distance_scales = np.array([
                                        1.0, # x scale in meters
                                        1.0, # y scale in meters
                                        1.0, # z scale in meters
                                        self.max_speed, # vx scale in m/s
                                        self.max_speed, # vy scale in m/s
                                        self.max_speed, # vz scale in m/s
                                    ], dtype=np.float64)
        self.distance_is_angle = np.array([False,False,False,False,False,False], dtype=np.bool_)

    def equation_of_motion(self, state, control):
        return quad6D_eom_numba(state, control, self.max_speed)

    def move_vehicle(self, state, control, dt):
        return quad6D_move_vehicle_numba(state, control, dt, self.max_speed)

    def get_next_state(self, state, control, dt, num_steps=10):
        return quad6D_get_next_state_numba(state, control, dt, num_steps, self.max_speed,
                                           self.state_length)
    
    def get_distance(self, state1, state2):
        # Euclidean distance over (x,y,z)
        # Uses first 3 dims by default
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)

    def get_random_action(self, rng):
        ax = rng.uniform(-self.max_acceleration, self.max_acceleration)
        ay = rng.uniform(-self.max_acceleration, self.max_acceleration)
        az = rng.uniform(-self.max_acceleration, self.max_acceleration)
        # return np.array((ax, ay, az), dtype=np.float64)
        return (ax, ay, az)

    def check_collision(self, base_agent_state, point):
        # Strict 3D distance between positions (x, y, z)
        return euclidean_distance_numba_with_l(base_agent_state, point, 3) <= (self.radius * 2.0)

    """
    UTILITY FUNCS
    """
    @staticmethod
    def is_state_valid(state, agent_radius, env_size,
                       sph_obs, box_obs, dyn_obs,
                       obstacle_buffer, dynamic_agent_clearance,
                       boundary_buffer, t):
        
        return is_state_valid_3d_numba(state, agent_radius,env_size,
                                sph_obs, box_obs, dyn_obs,
                                obstacle_buffer, dynamic_agent_clearance,
                                boundary_buffer, t)

    @staticmethod
    def is_new_node_valid(path_to_new_state, agent_radius, env_size,
                          sph_obs, box_obs, dyn_obs,
                          limit_indices, limit_values,
                          obstacle_buffer, dynamic_agent_clearance,
                          boundary_buffer,
                          start_time, time_duration, dt_per_step=0.1):
        
        return is_new_node_valid_3d_numba(path_to_new_state, agent_radius, env_size,
                                        sph_obs, box_obs, dyn_obs,
                                        limit_indices, limit_values,
                                        obstacle_buffer, dynamic_agent_clearance,
                                        boundary_buffer,
                                        start_time, time_duration, dt_per_step)

    @staticmethod
    def get_cost(env, agent, parent_state, control, t, path):
        # Simple cost: elapsed time + small acceleration penalty
        # if path is None or len(path) == 0:
        #     return t
        # acc_norm = np.linalg.norm(control)
        # return float(t + 0.05 * acc_norm * t)
        # return quad6D_get_distance_covered_numba(parent_state, path)
        return t #to match the dbCBS paper cost

    @staticmethod
    def get_random_point(env, spherical_obstacles, cuboid_obstacles, rng):
        """
        Gets a random 3D point for the quadcopter agent in the environment.
        """

        p = rng.uniform(env.env_start, env.size)

        while not validate_random_point_3d_numba(p, spherical_obstacles, 
                            cuboid_obstacles, env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)

        return np.array(p, dtype=np.float64)
            

    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """
        Return (reached_flag, distance) using strict 3D position distance.
        Expects `state` to have at least 3 elements [x,y,z,...] and
        `goal` to be [x,y,z].
        """
        return euclidean_distance_satisfaction_numba_with_l(state, goal, 3, goal_radius)

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point,
                                         edge_end_point):
        # Strict 3D translation of positions
        return quad6D_point_translate_function_kd_tree_numba(
            base_point, edge_start_point, edge_end_point)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
            final_states, curr_edge_indices, curr_edge_mask, distance_array):
        """
        Sorts edges based on their distance from a base point.
        """
        return quad_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                            start_states, final_states,
                                            curr_edge_indices, curr_edge_mask,
                                            distance_array)

    @staticmethod
    def no_sorting_kd_tree_edges(closest_tree_point, random_point, start_states,
            final_states, curr_edge_indices, curr_edge_mask, distance_array):
        """
        Returns unexplored edge candidate indices without distance sorting.
        """
        return quad_no_sorting_kd_tree_edges_numba(closest_tree_point,
                                            random_point, start_states,
                                            final_states, curr_edge_indices,
                                            curr_edge_mask, distance_array)
    
    def get_eb_kd_tree_query(self, state):
        """
        Gets the query point for the KD-Tree based on the agent's state.
        """
        return (state[3], state[4], state[5]) # vx, vy, vz
