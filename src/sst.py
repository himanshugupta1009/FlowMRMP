import numpy as np
import time
from utils import get_dtype_from_input,find_roundoff_decimal_digits, get_nearest_index, \
    get_nearest_index, compact_matrix, check_dynamic_collisions_to_end
from numba.typed import List
from numba import types, njit
from rrt import TreeNode, RRT  # Assuming TreeNode and RRT are defined in rrt.py
from typing import Optional, Tuple

# -----------------------------
# Metrics & utilities (Numba)
# -----------------------------
@njit
def l2(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return np.sqrt(s)

@njit
def l2_pos(a: np.ndarray, b: np.ndarray, pos_dims: int = 2) -> float:
    s = 0.0
    for i in range(pos_dims):
        d = a[i] - b[i]
        s += d * d
    return np.sqrt(s)

@njit
def get_best_active_near_index(states: np.ndarray, costs: np.ndarray, active: np.ndarray,
                  n_nodes: int, sample: np.ndarray, best_near_radius: float) -> int:
    """
    Return index of best-cost active node within best_near_radius of sample;
    Return -1 if none.
    """
    best = -1
    best_cost = 1e30
    for i in range(n_nodes):
        if active[i] == 1:
            # print("Active Node Index: ", i)
            if l2_pos(states[i], sample) < best_near_radius:
                # print("Dist: ", l2_pos(states[i], sample))
                c = costs[i]
                if c < best_cost:
                    best = i
                    best_cost = c
                # print("Best Cost: ", best_cost)
    return best

@njit
def get_nearest_active_index(states: np.ndarray, active: np.ndarray, n_nodes: int,
                       sample: np.ndarray) -> int:
    best = -1
    best_d = 1e30
    for i in range(n_nodes):
        if active[i] == 1:
            d = l2_pos(states[i], sample)
            if d < best_d:
                best = i
                best_d = d
    return best

@njit
def get_best_or_nearest_active_index(states: np.ndarray, costs: np.ndarray, active: np.ndarray,
                                     n_nodes: int, sample: np.ndarray,
                                     best_near_radius: float) -> int:
    """
    Return index of best-cost active node within best_near_radius of sample.
    If none, return nearest active node.
    """
    best = -1
    best_cost = 1e30
    nearest = -1
    nearest_d = 1e30

    for i in range(n_nodes):
        if active[i] == 1:
            d = l2_pos(states[i], sample)
            # Track nearest regardless
            if d < nearest_d:
                nearest = i
                nearest_d = d
            # Track best-cost within radius
            if d < best_near_radius:
                c = costs[i]
                if c < best_cost:
                    best = i
                    best_cost = c

    if best == -1:
        return nearest
    return best

@njit
def get_nearest_witness_index(w_states: np.ndarray, n_w: int, x: np.ndarray) -> Tuple[int, float]:
    """
    Nearest witness by witness state; returns (index, distance).
    """
    if n_w == 0:
        return -1, 1e30
    best = 0
    best_d = l2_pos(w_states[0], x)
    for i in range(1, n_w):
        d = l2_pos(w_states[i], x)
        if d < best_d:
            best = i
            best_d = d
    return best, best_d


class TreeNodeMatrix:
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length):
        self.id = np.full(initial_capacity, -1, dtype=np.int32)
        self.state = np.zeros((initial_capacity, state_dim), dtype=np.float64)
        self.parent = np.full(initial_capacity, -1, dtype=np.int32)
        self.parent_action = np.zeros((initial_capacity, action_dim), dtype=np.float64)
        self.action_duration = np.full(initial_capacity, -1, dtype=np.float32)
        self.path_from_parent = np.zeros((initial_capacity, max_sub_path_length, state_dim), dtype=np.float64)
        self.sub_path_length = np.full(initial_capacity, -1, dtype=np.int32)
        self.time_elapsed = np.full(initial_capacity, -1, dtype=np.float32)
        self.cost = np.full(initial_capacity, -1, dtype=np.float32)  # time cost
        self.count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_sub_path_length = max_sub_path_length

    def append(self,node_id,state,parent,action,action_duration,path_from_parent,time_elapsed,cost):
        if self.count >= self.state.shape[0]:
            self._grow()
        curr_index = self.count
        self.id[curr_index] = node_id
        self.state[curr_index] = state
        self.parent[curr_index] = parent
        self.parent_action[curr_index] = action
        self.action_duration[curr_index] = action_duration
        len_path_from_parent = path_from_parent.shape[0]
        self.path_from_parent[curr_index][:len_path_from_parent] = path_from_parent
        self.sub_path_length[curr_index] = len_path_from_parent
        self.time_elapsed[curr_index] = time_elapsed
        self.cost[curr_index] = cost
        self.count += 1
        return curr_index

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action = np.zeros((new_cap, self.action_dim), dtype=np.float64)
        new_action_duration = np.full(new_cap, -1, dtype=np.float32)
        new_path_from_parent = np.zeros((new_cap, self.max_sub_path_length, self.state_dim),
                                             dtype=np.float64)
        new_sub_path_length = np.full(new_cap, -1, dtype=np.int32)
        new_time_elapsed = np.full(new_cap, -1, dtype=np.float32)
        new_cost = np.full(new_cap, -1, dtype=np.float32)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_parent[:old_cap] = self.parent
        new_parent_action[:old_cap] = self.parent_action
        new_action_duration[:old_cap] = self.action_duration
        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length
        new_time_elapsed[:old_cap] = self.time_elapsed
        new_cost[:old_cap] = self.cost

        self.id = new_id
        self.state = new_state
        self.parent = new_parent
        self.parent_action = new_parent_action
        self.action_duration = new_action_duration
        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length
        self.time_elapsed = new_time_elapsed
        self.cost = new_cost

    def get_valid_states(self):
        return self.state[:self.count]

    def get_valid_ids(self):
        return self.id[:self.count]


class SSTNodeMatrix(TreeNodeMatrix):
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length):
        super().__init__(initial_capacity, state_dim, action_dim, max_sub_path_length)
        self.active = np.full(initial_capacity, -1, dtype=np.uint8)

    def append(self,node_id,state,parent,action,action_duration,path_from_parent,
               time_elapsed,cost):
        if self.count >= self.state.shape[0]:
            self._grow()
        curr_index = self.count
        self.id[curr_index] = node_id
        self.state[curr_index] = state
        self.parent[curr_index] = parent
        self.parent_action[curr_index] = action
        self.action_duration[curr_index] = action_duration
        len_path_from_parent = path_from_parent.shape[0]
        self.path_from_parent[curr_index][:len_path_from_parent] = path_from_parent
        self.sub_path_length[curr_index] = len_path_from_parent
        self.time_elapsed[curr_index] = time_elapsed
        self.cost[curr_index] = cost
        self.active[curr_index] = 0  # Mark as inactive
        self.count += 1
        return curr_index

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action = np.zeros((new_cap, self.action_dim), dtype=np.float64)
        new_action_duration = np.full(new_cap, -1, dtype=np.float32)
        new_path_from_parent = np.zeros((new_cap, self.max_sub_path_length, self.state_dim),
                                             dtype=np.float64)
        new_sub_path_length = np.full(new_cap, -1, dtype=np.int32)
        new_time_elapsed = np.full(new_cap, -1, dtype=np.float32)
        new_cost = np.full(new_cap, -1, dtype=np.float32)
        new_active = np.full(new_cap, -1, dtype=np.uint8)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_parent[:old_cap] = self.parent
        new_parent_action[:old_cap] = self.parent_action
        new_action_duration[:old_cap] = self.action_duration
        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length
        new_time_elapsed[:old_cap] = self.time_elapsed
        new_cost[:old_cap] = self.cost
        new_active[:old_cap] = self.active

        self.id = new_id
        self.state = new_state
        self.parent = new_parent
        self.parent_action = new_parent_action
        self.action_duration = new_action_duration
        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length
        self.time_elapsed = new_time_elapsed
        self.cost = new_cost
        self.active = new_active

    def get_valid_states(self):
        return self.state[:self.count]

    def get_valid_ids(self):
        return self.id[:self.count]


class SSTWitnessMatrix:
    def __init__(self, initial_capacity, state_dim):
        self.id = np.full(initial_capacity, -1, dtype=np.int32)
        self.state = np.zeros((initial_capacity, state_dim), dtype=np.float64)
        self.rep_index = np.full(initial_capacity, -1, dtype=np.int32)
        self.count = 0
        self.state_dim = state_dim

    def append(self,node_id,state,rep_index):
        if self.count >= self.state.shape[0]:
            self._grow()
        self.id[self.count] = node_id
        self.state[self.count] = state
        self.rep_index[self.count] = rep_index
        self.count += 1

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_rep_idx = np.full(new_cap, -1, dtype=np.int32)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_rep_idx[:old_cap] = self.rep_index
        
        self.id = new_id
        self.state = new_state
        self.rep_index = new_rep_idx

    def get_valid_ids(self):
        return self.id[:self.count]

    def get_valid_states(self):
        return self.state[:self.count]
    
    def get_valid_representatives(self):
        return self.rep_index[:self.count]


class SST(RRT):
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
                    best_near_radius=2.0,
                    prune_radius = 0.5,
                    debug_flag = False,
                    print_logs = False, 
                    dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))
                    ):


        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
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
        self.best_near_radius = best_near_radius
        self.prune_radius = prune_radius
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.debug_flag = debug_flag
        self.print_logs = print_logs

        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.last_added_node_id = -1
        self.last_added_witness_id = -1
        self.threshold = 3.0

        # Preprocess the static obstacles for fast collision checking
        # This is done only once at the start of the planning
        self.static_circular_obstacles = self.env.static_circular_obstacles
        self.static_rectangular_obstacles = self.env.static_rectangular_obstacles
        # Dynamic obstacles are not preprocessed, they are updated in real-time
        self.dynamic_agent_obstacles = dynamic_obstacles

        #Set the variable for distance metric state size that will be used for distance calculations
        if hasattr(self.agent, 'distance_metric_state_size'):
            self.distance_metric_state_size = self.agent.distance_metric_state_size
        else:
            self.distance_metric_state_size = 2

        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

        # Use Matrices for preallocated fast storage and for Numba optimization
        state_length = len(self.start)
        agent_action = agent.get_random_action(agent.rng)
        self.dummy_root_action = np.full(np.array(agent_action).shape, -1, dtype=np.float64)
        action_length = len(agent_action)
        max_sub_path_length = int(round(self.max_sample_T/self.minimum_time_step, 
                                        self.roundoff_digits))
        self.state_length = state_length
        self.action_length = action_length
        self.max_sub_path_length = max_sub_path_length
        self._node_matrix = SSTNodeMatrix(initial_capacity=1024,state_dim=state_length,
                            action_dim=action_length,max_sub_path_length=max_sub_path_length)
        self.type_node_matrix = type(self._node_matrix)
        self._witness_matrix = SSTWitnessMatrix(initial_capacity=1024, state_dim=state_length)
        self.type_witness_matrix = type(self._witness_matrix)


    def add_sst_node(self, state, parent_index, parent_action, parent_action_duration, 
                    path_from_parent, time_elapsed, cost):
        # Add a new node to the SST
        new_node_id = self.last_added_node_id + 1
        added_matrix_index = self._node_matrix.append(new_node_id, state, parent_index, 
                                    parent_action, parent_action_duration, path_from_parent,
                                    round(time_elapsed, self.roundoff_digits), cost)
        self.last_added_node_id = new_node_id

        return added_matrix_index


    def add_sst_witness(self,state: np.ndarray,rep_index: int) -> int:
        # Add a new witness to the SST
        new_witness_id = self.last_added_witness_id + 1
        # added_matrix_index = self._witness_matrix.append(new_witness_id, state[:self.distance_metric_state_size], 
        #                             rep_index)
        added_matrix_index = self._witness_matrix.append(new_witness_id, state[:], 
                                    rep_index)
        self.last_added_witness_id = new_witness_id

        return added_matrix_index


    def extend_tree(self, parent_node_index, random_point):
        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        # print("Parent Node Index: ", parent_node_index)
        # print("Parent Node State: ", self._node_matrix.state[parent_node_index])
        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        # print("Random Action: ", random_action)
        # print("Random Time: ", random_time)
        #Get values for the parent_node
        sst_nodes = self._node_matrix
        parent_state = sst_nodes.state[parent_node_index]
        parent_time_elapsed = sst_nodes.time_elapsed[parent_node_index]
        parent_cost = sst_nodes.cost[parent_node_index]

        new_state, path_to_new_state = self.agent.get_next_state(parent_state, random_action,
                                                                  random_time, num_steps=num_record_steps)
        # print("New State: ", new_state)
        # print("Path to New State: ", path_to_new_state)
        # accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                                       self.static_circular_obstacles,
                                       self.static_rectangular_obstacles,
                                       self.dynamic_agent_obstacles,
                                       self.env.obstacle_buffer,
                                       self.env.boundary_buffer,
                                       parent_time_elapsed,
                                       random_time,
                                       self.minimum_time_step)

        if not accept_new_node:
            if self.debug_flag:
                print("~~~~~~~~~~Sampled New SST Node is invalid. Trying again!~~~~~~~~~~")
            return -1
        else:
            reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
            if reached_goal_flag:
                if check_dynamic_collisions_to_end(new_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.env.obstacle_buffer,
                                                   parent_time_elapsed + random_time,
                                                   self.minimum_time_step):
                    if self.debug_flag:
                        print("Goal state will collide with high-priority agent. Trying again!")
                    return -1
                edge_cost = self.cost(self.env, self.agent, parent_state, random_action, 
                            random_time, path_to_new_state)
                total_elapsed_time = parent_time_elapsed + random_time
                total_cost = parent_cost + edge_cost

                new_node_index = self.add_sst_node(new_state, parent_node_index, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                self.path_found = True
                if self.debug_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_index
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                return new_node_index
            else:
                if goal_distance < self.threshold:
                    total_elapsed_time = parent_time_elapsed
                    for (index, intermediate_state) in enumerate(path_to_new_state):
                        total_elapsed_time += self.minimum_time_step
                        goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                                         self.goal_radius, self.agent)
                        if goal_flag:
                            if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius, 
                                                    self.dynamic_agent_obstacles, 
                                                    self.env.obstacle_buffer,
                                                    total_elapsed_time,
                                                    self.minimum_time_step):
                                    if self.print_logs:
                                        print("Goal state will collide with high-priority agent. Trying again!")
                                    return -1
                            modified_edge_time = total_elapsed_time - parent_time_elapsed
                            new_path_to_new_state = path_to_new_state[:index+1]
                            edge_cost = self.cost(self.env, self.agent, parent_state, random_action,
                                                modified_edge_time, new_path_to_new_state)
                            total_cost = parent_cost + edge_cost
                            new_node_index = self.add_sst_node(intermediate_state, parent_node_index, random_action,
                                                            modified_edge_time, new_path_to_new_state,
                                                            total_elapsed_time, total_cost)
                            self.path_found = True
                            if self.debug_flag:
                                print("Goal Reached! Path found for ",self.agent.id)
                            self.goal_node_id = new_node_index
                            self.path_cost = total_cost
                            self.path_time = total_elapsed_time
                            return new_node_index
                        
                edge_cost = self.cost(self.env, self.agent, parent_state, random_action,
                                        random_time, path_to_new_state)
                total_elapsed_time = parent_time_elapsed + random_time
                total_cost = parent_cost + edge_cost

                new_node_index = self.add_sst_node(new_state, parent_node_index, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                if self.debug_flag:
                    print("New Node Added to the SST Tree: ", new_node_index)
                return new_node_index


    def get_path_to_node_id(self, goal_node_id):

        # Return node ids, agent state and the executed control from start to goal 

        node_matrix = self._node_matrix
        state_length = node_matrix.state_dim
        action_length = node_matrix.action_dim
        max_nodes = node_matrix.count

        path_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty((max_nodes, action_length), dtype=np.float64)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_index = goal_node_id
        path_length = 0

        while node_index != -1:
            # print("Node ID: ", node_index)
            path_node_ids[path_length] = node_matrix.id[node_index]
            path_states[path_length] = node_matrix.state[node_index]
            path_controls[path_length] = node_matrix.parent_action[node_index]
            path_timesteps[path_length] = node_matrix.action_duration[node_index]
            path_length += 1
            node_index = node_matrix.parent[node_index]

        """
        Note: 
        1) Slicing a numpy array doesn't cause new array allocation. It just returns a view
        of the original array.
        2) Reversing the numpy array using [::-1] doesn't cause new array allocation. 
        It also just returns a view of the original array.
        3) Slicing and reversing the numpy array doesn't cause any allocation!!
        """    

        ids = path_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length-1][::-1]
        timesteps = path_timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps


    def get_high_resolution_path_numpy_array(self):
        """
        Return a high-resolution path from start to goal using SST matrices.
        """
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return np.empty((0, self.agent.state_length), dtype=np.float64)

        total_path_time = self.path_time
        min_time_step = self.minimum_time_step
        path_length = int(round(total_path_time / min_time_step, self.roundoff_digits)) + 1
        path_states = np.empty((path_length, self.agent.state_length), dtype=np.float64)

        node_index = self.goal_node_id
        curr_index = path_length - 1

        node_matrix = self._node_matrix

        while node_index != -1:  # Repeat until you reach the root node
            path_to_node_len = node_matrix.sub_path_length[node_index]
            start_index = curr_index - path_to_node_len + 1
            path_states[start_index:curr_index + 1] = node_matrix.path_from_parent[node_index, :path_to_node_len]
            curr_index -= path_to_node_len
            node_index = node_matrix.parent[node_index]

        # Fill remaining states with start state
        path_states[:curr_index + 1] = self.start

        return path_states


    def get_tree_structure(self):
        return (self._node_matrix, self._witness_matrix)


    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self._node_matrix = self.type_node_matrix(initial_capacity=1024,
                            state_dim=self.state_length,action_dim=self.action_length,
                            max_sub_path_length=self.max_sub_path_length)
            self._witness_matrix = self.type_witness_matrix(initial_capacity=1024,
                            state_dim=self.state_length)
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
            self.last_added_witness_id = -1
            self._node_matrix.count = 0
            self._witness_matrix.count = 0
        else:
            self._node_matrix = some_existing_tree[0]
            self._witness_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = self._node_matrix.count - 1
            self.last_added_witness_id = self._witness_matrix.count - 1

    def plan_path(self):
        """
        Plan a path using SST algorithm from scratch.
        """

        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.last_added_witness_id = -1
        self.reset_tree()
        first_node_state = self.start
        state_length = len(first_node_state)
        empty_path_from_parent = np.empty((0, state_length), dtype=np.float64)
        new_sst_node_index = self.add_sst_node(first_node_state, -1, self.dummy_root_action, 
                                               0.0, empty_path_from_parent, 0.0, 0.0)
        self._node_matrix.active[new_sst_node_index] = 1
        self.add_sst_witness(first_node_state, new_sst_node_index)

        curr_num_steps = 0
        sst_nodes = self._node_matrix
        sst_witnesses = self._witness_matrix

        start_time = time.time()
        
        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration in SST algorithm: ", curr_num_steps)

            # Sample a random point in the environment
            random_point = self.sample_random_point()

            #Find the nearest active node in the tree within best_near_radius
            #distance from the random point.
            # print("Count: ", sst_nodes.count)
            nearest_node_index = get_best_or_nearest_active_index(sst_nodes.state, sst_nodes.cost,
                                sst_nodes.active, sst_nodes.count, random_point, self.best_near_radius)

            if nearest_node_index==-1:
                # No active node close to x_rand; skip this iteration (SST behavior)
                nearest_node_index = get_nearest_active_index(sst_nodes.state, sst_nodes.active,
                                                               sst_nodes.count, random_point)

            #Propogate the tree from the nearest node with random control and duration
            new_node_index = self.extend_tree(nearest_node_index, random_point)
            new_node_state = sst_nodes.state[new_node_index]
            if self.debug_flag:
                print("New Node Index: ", new_node_index)
                print("New Node State: ", new_node_state)

            #Find the index of the nearest witness state from the new_node and the distance.
            nearest_witness_index, nearest_witness_distance = get_nearest_witness_index(
                            sst_witnesses.state, sst_witnesses.count, new_node_state)

            if self.debug_flag:
                print("Nearest Witness Index: ", nearest_witness_index)
                print("Nearest Witness Distance: ", nearest_witness_distance)

            if nearest_witness_distance > self.prune_radius:
                #Time to add a new witness.
                new_witness_index = self.add_sst_witness(new_node_state, new_node_index)
                sst_nodes.active[new_node_index] = 1
            else:
                #Existing witness is within the prune radius of some existing witness
                #No need to add a new one.
                curr_rep_index = sst_witnesses.rep_index[nearest_witness_index]
                if sst_nodes.cost[new_node_index] < sst_nodes.cost[curr_rep_index]:
                    # If the new node is better than the current representative, update it
                   sst_nodes.active[curr_rep_index] = 0
                   sst_witnesses.rep_index[nearest_witness_index] = new_node_index
                   sst_nodes.active[new_node_index] = 1

            if self.path_found or (time.time() - start_time >= self.planning_time):
                # If the goal is reached or planning time exceeded, break the loop
                break
            curr_num_steps += 1

        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations" 
            print(planning_time_msg + ": ", total_time)

