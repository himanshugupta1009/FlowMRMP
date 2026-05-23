"""
I attempted to make RRT even faster by replacing the networkX tree for storing nodes
with a custom data structure which is numba compatible.

However, even after these optimizations, there was no performance gain.
I suspect this is primarily due to the overhead of using this custom TreeNodeStruct class.
"""


from numba import njit, float64, int32
from numba.experimental import jitclass
from rrt import DynamicMatrix
from numba.typed import List
from numba import types

import numpy as np
import time
from utils import get_dtype_from_input,find_roundoff_decimal_digits,get_nearest_index, \
    preprocess_circular_obstacles, preprocess_rectangular_obstacles, compact_matrix


rrt_node_spec = [
    ('id', int32),
    ('state', float64[:]),
    ('parent_id', int32),
    ('parent_action', float64[:]),
    ('parent_action_duration', float64),
    ('path_from_parent', float64[:, :]),  # 2D array of path points
    ('time_elapsed', float64),
    ('cost_so_far', float64),
]

@jitclass(rrt_node_spec)
class TreeNodeStruct:
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                 path_from_parent, time_so_far, cost):
        self.id = sid
        self.state = state
        self.parent_id = parent_id
        self.parent_action = parent_action
        self.parent_action_duration = parent_action_duration
        self.path_from_parent = path_from_parent
        self.time_elapsed = time_so_far
        self.cost_so_far = cost


class NumbaRRT():
    def __init__(self, * , start, goal, goal_radius, env, agent,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1,
                    max_iter=1000,
                    planning_time=10.0, 
                    isvalid_function,
                    cost_function,
                    reached_goal_function, 
                    random_point_function,
                    udf_seed = 77, 
                    debug_flag = False,
                    print_logs = False,
                    dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))
                    ):
        
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time #Bool to use fixed sampling time or not
        self.max_sample_T = sampling_time_step #Max time to apply the control for in propogating each edge
        self.minimum_time_step = minimum_time_step #Smallest time duration between consecutive agent positions
        self.max_iter = max_iter #Max number of iterations for RRT
        self.planning_time = planning_time #Time to plan the path in seconds
        self.isvalid = isvalid_function #Function to check if the path is valid
        self.cost = cost_function #Function to calculate the cost of a edge
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.debug_flag = debug_flag
        self.print_logs = print_logs


        self.tree = List()
        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.last_added_node_id = -1
        self.threshold = 3.0

        # Preprocess the static obstacles for fast collision checking
        # This is done only once at the start of the RRT planning
        self.static_circular_obstacles = preprocess_circular_obstacles(self.env)
        self.static_rectangular_obstacles = preprocess_rectangular_obstacles(self.env)
        # Dynamic obstacles are not preprocessed, they are updated in real-time
        self.dynamic_agent_obstacles = dynamic_obstacles

        # Use DynamicMatrix for preallocated fast storage
        self.state_distance_dim = 2  # adjust if your state is larger
        self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self.state_distance_dim)

        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

    def get_random_time(self):
        return round(self.rng.uniform(self.minimum_time_step, self.max_sample_T), self.roundoff_digits)

    def get_fixed_time(self):
        return self.max_sample_T
    
    def get_time(self):
        if self.use_fixed_sampling_time:
            return self.get_fixed_time()
        else:
            return self.get_random_time()

    def num_rrt_nodes(self):
        return len(self.tree)
    
    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self.tree = List()
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
            self._node_matrix.count = 0
        else:
            self.tree = some_existing_tree
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            last_added_node_id = next(reversed(self.tree._node))
            self.last_added_node_id = last_added_node_id
            self._node_matrix.count = last_added_node_id + 1

    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost):
        # new_node_id = len(self.tree.nodes)
        new_node_id = self.last_added_node_id + 1
        new_node = TreeNodeStruct(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost)
        self.tree.append(new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant position dimensions (e.g., x, y)
        self._node_matrix.append(state[:self.state_distance_dim], new_node_id)

        return new_node_id

    def sample_random_point(self):
        """
        TO-DO: Implement a function to sample a random point for every environment and agent 
        that we generate using multiple dispatch from multipledispatch package for this. 
        """
        r = self.rng.uniform(0, 1)
        if r < 0.1:
            random_point = self.goal
        else:
            # random_point = self.get_random_point(self.env, self.agent, self.rng)
            random_point = self.get_random_point(self.env, self.static_circular_obstacles,
                                                 self.static_rectangular_obstacles, self.rng)
        return random_point
    
    def get_nearest_node(self, random_point):
        states = self._node_matrix.get_valid_matrix()
        nearest_index = get_nearest_index(states, random_point)
        nearest_node_id = self._node_matrix.get_valid_ids()[nearest_index]
        nearest_node = self.tree[nearest_node_id]

        return nearest_node_id, nearest_node

    def find_descendants(self, node_id):
        descendants = np.zeros(self.last_added_node_id + 1, dtype=np.bool_)
        descendants[node_id] = True

        for i in range(node_id + 1, self.last_added_node_id + 1):
            if i < len(self.tree):
                parent = self.tree[i].parent_id
                if descendants[parent]:
                    descendants[i] = True

        return np.nonzero(descendants)[0]
    
    def remove_descendants(self, descendants):
        # Remove from tree by rebuilding the list without descendants
        remaining_tree = List()
        valid_ids_set = np.zeros(self.last_added_node_id + 1, dtype=np.int8)

        for node in self.tree:
            if node.id not in descendants:
                remaining_tree.append(node)
                valid_ids_set[node.id] = 1

        self.tree = remaining_tree

        # Re-compact the node matrix
        self._node_matrix.count = compact_matrix(self._node_matrix.matrix,
                                                self._node_matrix.ids,
                                                valid_ids_set,
                                                self._node_matrix.count)
    
    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action,
                                                                  random_time, num_steps=num_record_steps)
        # accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                                       self.static_circular_obstacles,
                                       self.static_rectangular_obstacles,
                                       self.dynamic_agent_obstacles,
                                       self.env.obstacle_buffer,
                                       self.env.boundary_buffer,
                                       parent_node.time_elapsed,
                                       random_time,
                                       self.minimum_time_step)

        if not accept_new_node:
            if self.debug_flag:
                print("Sampled New RRT Node is invalid. Trying again!")
            return 
        else:
            reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
            if reached_goal_flag:
                edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                            random_time, path_to_new_state)
                total_elapsed_time = parent_node.time_elapsed + random_time
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                self.path_found = True
                if self.debug_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_id
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                return
            else:
                if goal_distance < self.threshold:
                    total_elapsed_time = parent_node.time_elapsed
                    for (index, intermediate_state) in enumerate(path_to_new_state):
                        total_elapsed_time += self.minimum_time_step
                        goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                                         self.goal_radius, self.agent)
                        if goal_flag:
                            modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                            new_path_to_new_state = path_to_new_state[:index+1]
                            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action,
                                                modified_edge_time, new_path_to_new_state)
                            total_cost = parent_node.cost_so_far + edge_cost
                            new_node_id = self.add_rrt_node(intermediate_state, parent_node_id, random_action, 
                                                            modified_edge_time, new_path_to_new_state, 
                                                            total_elapsed_time, total_cost)
                            self.path_found = True
                            if self.debug_flag:
                                print("Goal Reached! Path found for ",self.agent.id)
                            self.goal_node_id = new_node_id
                            self.path_cost = total_cost
                            self.path_time = total_elapsed_time
                            return
                
                edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                                        random_time, path_to_new_state)
                total_elapsed_time = parent_node.time_elapsed + random_time
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                if self.debug_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
            return

    def get_path_to_node_id(self, goal_node_id):

        # Return RRT node ids, agent state and the executed control from start to goal 

        # get state types
        # state_data_type = get_dtype_from_input(self.start)
        state_length = self.agent.state_length
        curr_rng = np.random.default_rng(77)
        control_data_type = get_dtype_from_input(self.agent.get_random_action(curr_rng))

        max_nodes = len(self.tree)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        # path_states = np.empty(max_nodes, dtype=state_data_type)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty(max_nodes, dtype=control_data_type)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_id = goal_node_id
        path_length = 0            

        while node_id != -1:
            # print("Node ID: ", node_id)
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_states[path_length] = rrt_node.state
            path_controls[path_length] = rrt_node.parent_action
            path_timesteps[path_length] = rrt_node.parent_action_duration
            path_length+=1
            node_id = rrt_node.parent_id

        """
        Note: 
        1) Slicing a numpy array doesn't cause new array allocation. It just returns a view
        of the original array.
        2) Reversing the numpy array using [::-1] doesn't cause new array allocation. 
        It also just returns a view of the original array.
        3) Slicing and reversing the numpy array doesn't cause any allocation!!
        """    

        ids = path_rrt_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length-1][::-1]
        timesteps = path_timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps


    def get_path(self):
        # get state types
        state_data_type = get_dtype_from_input(self.start)
        curr_rng = np.random.default_rng(77)
        control_data_type = get_dtype_from_input(self.agent.get_random_action(curr_rng))

        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")
            return np.empty(0,dtype=np.int16), np.empty(0, dtype=state_data_type), \
                    np.empty(0, dtype=control_data_type), np.empty(0, dtype=np.float64)
        else:           
            return self.get_path_to_node_id(self.goal_node_id)
        
    def plan_path(self):
        """
        Plan a path using RRT algorithm from scratch.
        """

        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.reset_tree()
        first_node_state = self.start 
        self.add_rrt_node(first_node_state, -1, self.agent.get_random_action(self.rng), 
                          -1.0, np.zeros((self.agent.state_length,1)), 0.0, 0.0)

        curr_num_steps = 0
        start_time = time.time()
        
        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1


        end_time = time.time()
        total_time = end_time - start_time

        if self.print_logs or self.debug_flag:
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations" 
            print(planning_time_msg + ": ", total_time)


"""
import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar
from numba_rrt import *
from printer import *

obstacles= []
obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 20, 4),
            CircularObstacle2D(20, 6, 3),
            # CircularObstacle2D(35, 15, 2),
            # CircularObstacle2D(10, 30, 5),
            # CircularObstacle2D(25, 15, 5),
            # CircularObstacle2D(7, 17, 5),
            # CircularObstacle2D(16, 13, 2),
            ] 
obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 2),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 4),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]   
obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 4),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 5),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]   
# obstacles = [
#             CircularObstacle2D(10, 10, 2),
#             CircularObstacle2D(16, 25, 3),
#             CircularObstacle2D(20, 5, 2),
#             CircularObstacle2D(35, 15, 2),
#             CircularObstacle2D(30, 34, 4),
#             CircularObstacle2D(25, 15, 4),
#             # RectangleObstacle2D(7, 19, 4, 4),
#             ]
                    

env = SquareEnvironment(40, 40, obstacles)
agent = SecondOrderCar(agent_id = 1, 
                       max_speed = 2.0,
                       max_acceleration = 1.0,
                       max_phi = np.pi/3,
                       max_steering_rate = 0.5,
                       radius = 0.3,
                       wheelbase = 0.7,
                       rng_seed=42
                       )
edge_bundle = None
start = np.array([7.0, 5.0, 0, 0.0, 0.0], dtype=np.float64)
goal = np.array([24.0, 37.0], dtype=np.float64)
goal_radius = 0.5

s = np.random.randint(0, 1000)
s=829
rrt  = NumbaRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_second_order_car.png')



"""