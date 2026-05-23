import numpy as np
import random
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba, \
    get_distance_covered_numba


"""
Vehicle State S = (x,y,theta,v,omega)
Vehicle Control U = (acceleration, steering_angle)
"""

@njit
def second_order_car_equation_of_motion_numba(state, control, max_speed, max_phi, l):
    x, y, theta, v, phi = state
    acceleration, steering_rate = control

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v * np.tan(phi) / l
    v_dot = acceleration
    phi_dot = steering_rate

    return np.array([x_dot, y_dot, theta_dot, v_dot, phi_dot])
    # return (x_dot, y_dot, theta_dot, v_dot, phi_dot)

@njit
def second_order_car_move_vehicle_numba(state, control, dt, max_speed, max_phi, l):
    k1 = second_order_car_equation_of_motion_numba(state, control, max_speed, max_phi, l)
    k2 = second_order_car_equation_of_motion_numba(state + 0.5 * dt * k1, control, max_speed, max_phi, l)
    k3 = second_order_car_equation_of_motion_numba(state + 0.5 * dt * k2, control, max_speed, max_phi, l)
    k4 = second_order_car_equation_of_motion_numba(state + dt * k3, control, max_speed, max_phi, l)

    next_state = np.empty_like(state)
    tmp = (dt / 6.0)
    for i in range(state.shape[0]):
        next_state[i] = state[i] + tmp * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    # next_state[2] = wrap_between_0_and_2pi(next_state[2])

    return next_state

@njit
def second_order_car_get_next_state_numba(state, control, dt, num_steps, max_speed, max_phi, l, state_length):
    path = np.empty((num_steps, state_length), dtype=np.float64)
    curr_state = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        new_state = second_order_car_move_vehicle_numba(curr_state, control, exec_dt, max_speed, max_phi, l)
        path[i] = new_state
        curr_state = new_state

    return curr_state, path

@njit 
def second_order_car_get_distance_covered_numba(start_state,path):
    if len(path) == 0:
        return 0.0
    
    dx = path[0,0] - start_state[0]
    dy = path[0,1] - start_state[1]
    total_distance = np.sqrt(dx*dx + dy*dy)
    for i in range(1, path.shape[0]):
        # dx = path[i][0] - path[i-1][0]
        dx = path[i,0] - path[i-1,0]
        dy = path[i,1] - path[i-1,1]
        total_distance += np.sqrt(dx*dx + dy*dy)
    return total_distance

@njit 
def SOC_point_translate_function_kd_tree_numba(base_point, 
                            edge_start_point, edge_end_point):
    """
    base_point: (x_c, y_c, theta_c, v_c, phi_c)
    edge_start_point: (0, 0, 0, v_s, phi_s)
    edge_end_point: (x_e, y_e, theta_e, v_e, phi_e)

    Returns new final state (x_f', y_f', theta_f', v_f', phi_f') 
    applying same edge from base_point.
    """
    
    theta_c = base_point[2]
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)

    new_x = edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta

    return (base_point[0] + new_x, base_point[1] + new_y,
            base_point[2] + edge_end_point[2],
            edge_end_point[3],
            edge_end_point[4])

#Need to modify this function later for two things.
#1) You don't really need to add distance as 1e10 for already explored edges.
#2) Need to stop creating and returning a new array when you call argsort. 
@njit
def SOC_sort_kd_tree_edges_numba(closest_tree_point, random_point, 
                                start_states, final_states, curr_edge_indices, 
                                curr_edge_mask, distance_array):

    n = curr_edge_indices.shape[0]
    num_valid_edges = 0

    for i in range(n):
        if curr_edge_mask[i] == True:
            #That edge has been explored before
            distance_array[i] = 1e10
        else:
            edge_idx = curr_edge_indices[i]
            potential_new_point = SOC_point_translate_function_kd_tree_numba(
                closest_tree_point,start_states[edge_idx],final_states[edge_idx])
            dist = euclidean_distance_numba(potential_new_point, random_point)
            distance_array[i] = dist
            num_valid_edges += 1

    # sorted_indices = np.argsort(distance_array[0:num_valid_edges])
    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges

@njit
def SOC_get_unmasked_kd_tree_edges_no_sorting_numba(closest_tree_point, random_point, 
                                start_states, final_states, curr_edge_indices, 
                                curr_edge_mask, distance_array):

    n = curr_edge_indices.shape[0]
    num_valid_edges = 0
    good_edges = np.empty(n, dtype=np.int64)

    for i in range(n):
        if curr_edge_mask[i] != True:
            #That edge has not been explored before
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


class SecondOrderCar:
    def __init__(self, *, 
                agent_id=-1,
                max_speed=1.0,
                max_acceleration = 1.0,
                max_phi = np.pi/3,
                max_steering_rate = 0.5,
                radius = 0.3,
                wheelbase = 0.7,
                rng_seed=42
                ):
        
        self.max_speed=max_speed
        self.max_acceleration=max_acceleration
        self.max_phi=max_phi
        self.max_steering_rate=max_steering_rate
        self.id = agent_id
        self.radius = radius
        self.l = wheelbase
        self.rng = np.random.default_rng(rng_seed)
        self.state_datatype = np.dtype([('x', 'f8'), ('y', 'f8'), ('theta', 'f8'), ('v', 'f8'), ('phi', 'f8')])
        self.state_length = 5
        self.distance_metric_state_size = 2
        self.action_length = 2
        self.dynamic_limit_indices = np.array([3, 4], dtype=np.int64)
        self.dynamic_limit_values = np.array([self.max_speed, self.max_phi],dtype=np.float64)
        
        # attributes for total state distance metric (x, y, theta, v, phi)
        self.distance_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        self.distance_scales = np.array([
                                1.0,              # x scale in meters
                                1.0,              # y scale in meters
                                np.pi,            # theta scale in radians
                                self.max_speed,   # v scale
                                self.max_phi,     # phi scale
                            ], dtype=np.float64)
        self.dbastar_distance_is_angle = np.array([False,False,True,False,False,], dtype=np.bool_)

    def equation_of_motion(self, state, control):
        return second_order_car_equation_of_motion_numba(state, control, self.max_speed, self.max_phi, self.l)
    
    def move_vehicle(self, state, control, dt):
        return second_order_car_move_vehicle_numba(state, control, dt, self.max_speed, self.max_phi, self.l)
        
    def get_next_state(self, state, control, dt, num_steps=10):
        return second_order_car_get_next_state_numba(state, control, dt, num_steps, self.max_speed,
                                                      self.max_phi, self.l, self.state_length)

    def get_distance(self, state1, state2):
        # return euclidean_distance((state1[0], state1[1]), (state2[0], state2[1]))
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)

    def get_random_action(self, rng):
        a = rng.uniform(-self.max_acceleration, self.max_acceleration)
        steering_rate = rng.uniform(-self.max_steering_rate, self.max_steering_rate)
        # return np.array((a, steering_rate))
        return (a, steering_rate)
    
    def check_collision(self, base_agent_state, point):
        d = self.get_distance(base_agent_state,point)
        if d<=self.radius*2:
            return True
        else:
            return False

    """
    UTILITY FUNCS
    """
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
        """
        Checks if a new state and the path to that new state is valid for 
        an environment.

        :PRE: Environment contains circular obstacles ONLY!

        Args:
            env: Environment object 
            agent: 2nd order car agent object
            path_to_new_state (list[mecanum state tuple]): Path taken by the agent 
                from the parent state, non-inclusive of the parent state but inclusive
                of the final state 

        Returns:
            bool: True if new node is valid, false else 
        """
        # # check if each element of the path is valid
        # for state in path_to_new_state:
        #     if not SecondOrderCar.is_state_valid(env, agent, state):
        #         return False
        # return True
        return is_new_node_valid_numba(path_to_new_state, agent_radius, env_size,
                                        circ_obs, rect_obs, dyn_obs,
                                        limit_indices, limit_values,
                                        obstacle_buffer, dynamic_agent_clearance,
                                        boundary_buffer,
                                        start_time, time_duration,
                                        dt_per_step)
    
    @staticmethod
    def get_cost(env, agent, parent_state, a, t, path): 
        """
        Gets the cost a 2nd order car agent incurs to traverse 
        a path

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            env: Environment object 
            agent: 2nd order car agent object
            parent_state (2nd order car agent state tuple): Start point for path 
            a (2nd order car agent control type): control input that generated
                path 
            t (float): Time over which path was propagated NOT USED
            edge (list(mecanum_agent_state_type)): new path over which 
                to generate cost

        Returns:
            float: approximate path cost
        """        
        # return second_order_car_get_distance_covered_numba(parent_state, path)
        return t #to match the dbCBS paper cost
            
    @staticmethod
    def get_random_point(env, circular_obstacles, rectangular_obstacles, rng):
        """
        Gets a random point for a 2nd order car agent in an environment 

        Args:
            env: Environment object 
            agent: 2nd order car agent object
            rng (numpy.random): random number generator

        Returns:
            np.array([x, y]): new valid point for the agent in the environment  
        """        
        p = rng.uniform(env.env_start, env.size)

        while not validate_random_point_numba(p, circular_obstacles,
                                rectangular_obstacles, env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)

        return np.array(p, dtype=np.float64)
    
    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """
        Determines whether the agent has reached the goal region

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            state (agent_state_type): agent state
            goal (tuple(float, float)): goal center
            goal_radius (float): goal region radius
            agent: 2nd Order Car agent object NOT USED

        Returns:
            _type_: _description_
        """        
        return euclidean_distance_satisfaction_numba(state, goal, goal_radius)

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point, edge_end_point):
        """
        Translates the agent's state by a given 2D vector.
        """
        return SOC_point_translate_function_kd_tree_numba(base_point, 
                            edge_start_point, edge_end_point)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
            final_states, curr_edge_indices, curr_edge_mask, distance_array):
        """
        Sorts edges based on their distance from a base point.
        """
        return SOC_sort_kd_tree_edges_numba(closest_tree_point,
            random_point, start_states, final_states, curr_edge_indices,
            curr_edge_mask, distance_array)
    
    @staticmethod
    def no_sorting_kd_tree_edges(closest_tree_point, random_point, start_states,
            final_states, curr_edge_indices, curr_edge_mask, distance_array):
        """
        Returns unexplored edge candidate indices without distance sorting.
        """
        return SOC_get_unmasked_kd_tree_edges_no_sorting_numba(
            closest_tree_point,random_point, start_states, final_states,
            curr_edge_indices, curr_edge_mask, distance_array)

    def get_eb_kd_tree_query(self, state):
        """
        Gets the query point for the KD-Tree based on the agent's state.
        """
        # return np.array([state[3],state[4]], dtype=np.float64)
        return (state[3], state[4]) # v, phi
