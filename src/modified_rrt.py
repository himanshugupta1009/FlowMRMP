from rrt import *
from numba import njit

class GoalTruncatedRRTType1(RRT):
    """
    RRT where its checked if an edge went through the goal region or not
    only when the final new_state is within some threshold radius to the goal :
    """
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
                    print_flag = False
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
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed) 
        self.print_flag = print_flag 
        self.last_added_node_id = -1
        self.threshold = 3.0 #Distance threshold to check if the new node is close to the goal


    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
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
                if self.print_flag:
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
                            if self.print_flag:
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
                if self.print_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
                return
    

class GoalTruncatedRRTType2(RRT):
    """
    RRT where its checked if an edge went through the goal region or not 
    for every edge.
    """
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
                    print_flag = False
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
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed) 
        self.print_flag = print_flag 
        self.last_added_node_id = -1
        self.threshold = 3.0 #Distance threshold to check if the new node is close to the goal

    
    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
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
                if self.print_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_id
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                return
            else:
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
                        if self.print_flag:
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
                if self.print_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
                return


class MatrixNNGoalTruncatedRRTType1(RRT):
    """
    RRT where its checked if an edge went through the goal region or not
    only when the new_state is within some threshold radius to the goal.

    Also, this RRT uses a list to store the node states and ids and uses
    a fast nearest neighbor search using the squared distance optimized with 
    numba (get_nearby_index).
    """

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
                    print_flag = False
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
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed) 
        self.print_flag = print_flag 
        self.last_added_node_id = -1
        self.threshold = 3.0 #Distance threshold to check if the new node is close to the goal
        self._node_state_matrix = []
        self._node_id_list = []

    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration,
                    path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id + 1
        new_node = TreeNode(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                            path_from_parent, time_elapsed, cost)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Update matrix + id list
        self._node_state_matrix.append(state[:2])  # make sure this is numeric and flat
        self._node_id_list.append(new_node_id)

        return new_node_id

    # def get_nearest_node(self, random_point):
    #     # query = np.asarray(random_point, dtype=np.float64).ravel()
    #     # states = np.asarray(self._node_state_matrix, dtype=np.float64)  # (N x D)
    #     states = self._node_state_matrix  # (N x D)
    #     query = random_point

    #     # Efficient squared distance computation to avoid unnecessary sqrt
    #     deltas = states - query
    #     dists_squared = np.einsum('ij,ij->i', deltas, deltas)  # faster than norm(states - query, axis=1)
    #     nearest_index = np.argmin(dists_squared)

    #     nearest_node_id = self._node_id_list[nearest_index]
    #     nearest_node = self.tree.nodes[nearest_node_id]['value']
    #     return nearest_node_id, nearest_node
    
    def get_nearest_node(self, random_point):
        # query = np.asarray(random_point, dtype=np.float64).ravel()
        states = np.asarray(self._node_state_matrix, dtype=np.float64)  # (N x D)
        # states = self._node_state_matrix  # (N x D)
        query = random_point

        # Efficient squared distance computation to avoid unnecessary sqrt
        # deltas = states - query
        # dists_squared = np.einsum('ij,ij->i', deltas, deltas)  # faster than norm(states - query, axis=1)
        # nearest_index = np.argmin(dists_squared)

        nearest_index = get_nearest_index(states, query)

        nearest_node_id = self._node_id_list[nearest_index]
        nearest_node = self.tree.nodes[nearest_node_id]['value']
        return nearest_node_id, nearest_node


    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
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
                if self.print_flag:
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
                            if self.print_flag:
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
                if self.print_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
                return
    

class DynamicMatrix:
    def __init__(self, initial_capacity, dim):
        self.matrix = np.zeros((initial_capacity, dim), dtype=np.float64)
        self.ids = [None] * initial_capacity
        self.count = 0
        self.dim = dim

    def append(self, row, node_id):
        if self.count >= self.matrix.shape[0]:
            self._grow()
        self.matrix[self.count] = row
        self.ids[self.count] = node_id
        self.count += 1

    def _grow(self):
        old_cap = self.matrix.shape[0]
        new_cap = 2 * old_cap
        new_matrix = np.zeros((new_cap, self.dim), dtype=np.float64)
        new_matrix[:old_cap] = self.matrix
        self.matrix = new_matrix

        self.ids.extend([None] * old_cap)

    def get_valid_matrix(self):
        return self.matrix[:self.count]

    def get_valid_ids(self):
        return self.ids[:self.count]


class MatrixNNGoalTruncatedRRTType2(RRT):
    """
    RRT where its checked if an edge went through the goal region or not
    only when the new_state is within some threshold radius to the goal.

    Also, this RRT uses a preallocated matrix to store the node states and ids
    and uses a fast nearest neighbor search using the squared distance optimized with 
    numba (get_nearby_index).
    """
    def __init__(self, *, start, goal, goal_radius, env, agent,
                 use_fixed_sampling_time=True, 
                 sampling_time_step=1.0,
                 minimum_time_step=0.1, 
                 max_iter=1000, 
                 planning_time=10.0,
                 isvalid_function, 
                 cost_function, 
                 reached_goal_function,
                 random_point_function, 
                 udf_seed=77, 
                 print_flag=False
                 ):

        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time
        self.max_sample_T = sampling_time_step
        self.minimum_time_step = minimum_time_step
        self.max_iter = max_iter
        self.planning_time = planning_time
        self.isvalid = isvalid_function
        self.cost = cost_function
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.print_flag = print_flag
        self.last_added_node_id = -1
        self.threshold = 3.0

        # NEW: Use DynamicMatrix for preallocated fast storage
        # self._state_dim = len(self.start)  # adjust if your state is larger
        self._state_dim = 2  # adjust if your state is larger
        self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self._state_dim)


    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration,
                     path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id + 1
        new_node = TreeNode(new_node_id, state, parent_node_id, parent_action,
                           parent_action_duration, path_from_parent, time_elapsed, cost)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant position dimensions (e.g., x, y)
        self._node_matrix.append(state[:self._state_dim], new_node_id)

        return new_node_id

    def get_nearest_node(self, random_point):
        # query = np.asarray(random_point, dtype=np.float64).ravel()
        query = random_point
        states = self._node_matrix.get_valid_matrix()

        nearest_index = get_nearest_index(states, query)
        nearest_node_id = self._node_matrix.get_valid_ids()[nearest_index]
        nearest_node = self.tree.nodes[nearest_node_id]['value']

        return nearest_node_id, nearest_node

    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
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
                if self.print_flag:
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
                            if self.print_flag:
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
                if self.print_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
                return
            

class MatrixNNGoalTruncatedRRTType3(RRT):
    """
    RRT where its checked if an edge went through the goal region or not
    only when the new_state is within some threshold radius to the goal.

    Also, this RRT uses a preallocated matrix to store the node states and ids
    and uses numpy vector dot product to compute distances.
    """
    def __init__(self, *, start, goal, goal_radius, env, agent,
                 use_fixed_sampling_time=True, sampling_time_step=1.0,
                 minimum_time_step=0.1, max_iter=1000, planning_time=10.0,
                 isvalid_function, cost_function, reached_goal_function,
                 random_point_function, udf_seed=77, print_flag=False):

        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time
        self.max_sample_T = sampling_time_step
        self.minimum_time_step = minimum_time_step
        self.max_iter = max_iter
        self.planning_time = planning_time
        self.isvalid = isvalid_function
        self.cost = cost_function
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.print_flag = print_flag
        self.last_added_node_id = -1
        self.threshold = 3.0

        # NEW: Use DynamicMatrix for preallocated fast storage
        # self._state_dim = len(self.start)  # adjust if your state is larger
        self._state_dim = 2  # adjust if your state is larger
        self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self._state_dim)


    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration,
                     path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id + 1
        new_node = TreeNode(new_node_id, state, parent_node_id, parent_action,
                           parent_action_duration, path_from_parent, time_elapsed, cost)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant position dimensions (e.g., x, y)
        self._node_matrix.append(state[:self._state_dim], new_node_id)

        return new_node_id

    def get_nearest_node(self, random_point):
        # query = np.asarray(random_point, dtype=np.float64).ravel()
        query = random_point
        states = self._node_matrix.get_valid_matrix()

        # Efficient squared distance computation to avoid unnecessary sqrt
        deltas = states - query
        dists_squared = np.einsum('ij,ij->i', deltas, deltas)  # faster than norm(states - query, axis=1)
        nearest_index = np.argmin(dists_squared)

        # nearest_index = get_nearest_index(states, query)
        nearest_node_id = self._node_matrix.get_valid_ids()[nearest_index]
        nearest_node = self.tree.nodes[nearest_node_id]['value']

        return nearest_node_id, nearest_node

    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
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
                if self.print_flag:
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
                            if self.print_flag:
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
                if self.print_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
                return