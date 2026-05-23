import networkx as nx
import numpy as np
from operator import add
import time
from numba.typed import List
from numba import njit, types

from rrt import TreeNode, RRT, DynamicMatrix
from utils import get_dtype_from_input,find_roundoff_decimal_digits, \
     point_circle_collision, point_sphere_collision

@njit(cache=True)
def _superstate_to_matrix_state_numba(superstate, num_agents,
                                      agent_position_state_dim,
                                      distance_metric_state_size):
    matrix_state = np.empty(distance_metric_state_size)
    for agent_counter in range(num_agents):
        matrix_state_start = agent_position_state_dim * agent_counter
        for state_dim in range(agent_position_state_dim):
            matrix_state[matrix_state_start + state_dim] = superstate[agent_counter][state_dim]
    return matrix_state

class CrrtTreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost, reached_goals = []):
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,\
                         path_from_parent, time_so_far, cost)
        self.reached_goals = reached_goals
        

def check_collisions(crrt, list_of_paths, start_time = None, start_index=0, agent_index=None):
    """
    Collision check func for 'mathematical' sims, i.e. those without high-level 
    sim environments (i.e. PyBullet)

    :MAINT: Assumes circular agents (i.e. agents with a 'radius' field)!

    Args:
        crrt: crrt obj
        list_of_paths (list(list(agent_state_type))): list of paths for each state, 
            should be equal length and identically spaced
        start_index (int, optional): path index to start checking from (i.e. ignore 
            indices up to this point). Defaults to Zero
        start_time (float, optional): NOT USED. Defaults to None
        agent_index (int, optional). If set, only check collisions against this agent
            instead of between all agents. Defaults to None

    Returns:
        bool: True if a collision between any two agents detected, false else 
    """

    # if the agent_index is defined, only check against that agent
    first_agent_range = range(len(crrt.agents))
    if agent_index is not None:
        first_agent_range = range(agent_index, agent_index+1)

    second_agent_range = lambda x : range(x+1, len(crrt.agents))
    if agent_index is not None:
        second_agent_range = lambda ai : (x for x in range(len(crrt.agents)) if x != ai)

    # for each agent...
    for ind_first_agent in first_agent_range:
        first_agent = crrt.agents[ind_first_agent]
        first_agent_path = list_of_paths[ind_first_agent]
        first_agent_path_len = len(first_agent_path)

        # ...go through all other agents 
        for ind_second_agent in second_agent_range(ind_first_agent):
            second_agent = crrt.agents[ind_second_agent]
            second_agent_path = list_of_paths[ind_second_agent]
            second_agent_path_len = len(second_agent_path)

            path_len = max(first_agent_path_len, second_agent_path_len)

            # check each state in the paths against each other
            for state_index in range(start_index, path_len):
                state_first_agent = first_agent_path[-1]
                if (state_index < first_agent_path_len):
                    state_first_agent = first_agent_path[state_index]
                state_second_agent = second_agent_path[-1]
                if (state_index < second_agent_path_len):
                    state_second_agent = second_agent_path[state_index]

                # :MAINT: assuming circular agents!!!!
                if point_circle_collision(state_first_agent[0], state_first_agent[1], first_agent.radius, 
                                          state_second_agent[0], state_second_agent[1], second_agent.radius+crrt.env.obstacle_buffer):
                    # shortcut the process: if one collision is found, all 
                    # paths are invalid 
                    return True # there is a collision
    
    return False # there is no collision

def check_collisions_3d(crrt, list_of_paths, start_time = None, start_index=0, agent_index=None):
    """
    Collision check func for 'mathematical' sims, i.e. those without high-level 
    sim environments (i.e. PyBullet)

    :MAINT: Assumes circular agents (i.e. agents with a 'radius' field)!

    Args:
        crrt: crrt obj
        list_of_paths (list(list(agent_state_type))): list of paths for each state, 
            should be equal length and identically spaced
        start_index (int, optional): path index to start checking from (i.e. ignore 
            indices up to this point). Defaults to Zero
        start_time (float, optional): NOT USED. Defaults to None
        agent_index (int, optional). If set, only check collisions against this agent
            instead of between all agents. Defaults to None

    Returns:
        bool: True if a collision between any two agents detected, false else 
    """

    # if the agent_index is defined, only check against that agent
    first_agent_range = range(len(crrt.agents))
    if agent_index is not None:
        first_agent_range = range(agent_index, agent_index+1)

    second_agent_range = lambda x : range(x+1, len(crrt.agents))
    if agent_index is not None:
        second_agent_range = lambda ai : (x for x in range(len(crrt.agents)) if x != ai)

    # for each agent...
    for ind_first_agent in first_agent_range:
        first_agent = crrt.agents[ind_first_agent]
        first_agent_path = list_of_paths[ind_first_agent]
        first_agent_path_len = len(first_agent_path)

        # ...go through all other agents 
        for ind_second_agent in second_agent_range(ind_first_agent):
            second_agent = crrt.agents[ind_second_agent]
            second_agent_path = list_of_paths[ind_second_agent]
            second_agent_path_len = len(second_agent_path)

            path_len = max(first_agent_path_len, second_agent_path_len)

            # check each state in the paths against each other
            for state_index in range(start_index, path_len):
                state_first_agent = first_agent_path[-1]
                if (state_index < first_agent_path_len):
                    state_first_agent = first_agent_path[state_index]
                state_second_agent = second_agent_path[-1]
                if (state_index < second_agent_path_len):
                    state_second_agent = second_agent_path[state_index]

                # :MAINT: assuming circular agents!!!!
                if point_sphere_collision(state_first_agent[0], state_first_agent[1], state_first_agent[2], first_agent.radius, 
                                          state_second_agent[0], state_second_agent[1], state_second_agent[2], 
                                          second_agent.radius+crrt.env.obstacle_buffer):
                    # shortcut the process: if one collision is found, all 
                    # paths are invalid 
                    return True # there is a collision

class CRRT(RRT):
    def __init__(self, *, agents, starts, goals, goal_radii, env,
                    use_fixed_sampling_time=True, sampling_time_step=1.0, minimum_time_step=0.1, 
                    max_iter=1000, planning_time=10.0,
                    isvalid_function, 
                    cost_function, 
                    reached_goal_function, 
                    random_point_function, 
                    udf_seed = 77, 
                    dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C')), 
                    collision_check_func = None,
                    truncate_paths = False,
                    print_logs = False,
                    debug_flag = False):
        """
        Instantiate a new Centralized RRT (cRRT) planner. 

        CRRT plans for each agent concurrently. At each new step, a random point is 
        generated for each agent. The tree is searched for the closest existing joint
        state between all the agents to the new joint state created from the random 
        points and each agent propagates a new path from that existing state. Those 
        paths are checked for mutual collisions. If no collision is found, that new
        joint state is added to the tree. If a mutual collision is found the new 
        joint state is discarded. Once a joint state is found where all the agents
        are in their goal regions, a valid path has been found and the planning 
        process stops. 

        Args:
            agents (list(agent objs)): List of agent objects
            starts (list(agent_state_type)): Agent start positions
            goals (list(tuple(x, y))): list of agent goal centers
            goal_radii (list(float)): list of agent goal radii
            env: environment object
            isvalid_function (bool): List of functions for each agent that return true if a path is valid
                for an agent 
                (agent_state_type path_to_new_state, float agent_radius, tuple env_size,
                    list circ_obs, list rect_obs, list dyn_obs,
                    np.array limit_indices, np.array limit_values,
                    float obstacle_buffer, float boundary_buffer,
                    float start_time, float time_duration,
                    float dt_per_step=0.1) -> bool  
            cost_function (float): List of functions for each agent that return the cost an agent 
                incurs over a path
                (env object, agent object, agent_state_type start state, agent_action_type random_action,
                    float time_delta, list(agent_state_type) path_to_new_state) -> float
            reached_goal_function (bool): List of functions for each agent that returns true if the agent 
                has reached its goal, false else. 
                (agent_state_type state, tuple(x float, y float) goal center, float goal radius, 
                    agent object) -> bool  
            random_point_function (tuple(x float, y float)): List of functions for each agent that 
                generate a new point in the environment 
                (env object, list circular obstacles, list rectangular obstacles, rng object) -> tuple(x float, y float)
            use_fixed_sampling_time (bool, optional): Don't use random timesteps. Defaults to True.
            sampling_time_step (float, optional): Max sampling time to use. Defaults to 1.0.
            minimum_time_step (float, optional): Minimum timestep in agent paths. Defaults to 0.1
            max_iter (int, optional): Maximum number of planning iterations before failure. Defaults to 1000.
            planning_time (float, optional): maximum time to plan before failure. Defaults to 10.0.
            udf_seed (int, optional): local rng seed. Defaults to 77.
            collision_check_func (bool, optional): Function to check collisions between paths. 
                Defaults to check_collisions.
                (crrt obj, list(list(agent_state_type)) -> bool
            dynamic_obstacles (numba list, optional): List of dynamic obstacles to avoid. Defaults to empty
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false
        """
        # init the superclass, most of which will not be used
        RRT.__init__(self, start=starts[0], goal=goals[0], goal_radius=goal_radii[0], env=env, agent=agents[0],
                         use_fixed_sampling_time=use_fixed_sampling_time, 
                         sampling_time_step=sampling_time_step, 
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter, 
                         planning_time=planning_time,
                         isvalid_function=isvalid_function[0],
                         cost_function=cost_function[0], 
                         reached_goal_function=reached_goal_function[0],
                         random_point_function=random_point_function[0],
                         udf_seed=udf_seed,
                         print_logs=print_logs,
                         debug_flag=debug_flag,
                         dynamic_obstacles=dynamic_obstacles)

        self.agents = agents
        self.starts = [np.array(start) for start in starts]
        self.goals = [np.array(goal) for goal in goals]
        self.goal_radii = goal_radii

        self.planning_time = planning_time #Time to plan the path in seconds
        self.isvalid = isvalid_function #Functions to check if the path is valid
        self.cost_funcs = cost_function #Functions to calculate the cost of a edge
        self.reached_goal_funcs = reached_goal_function

        # dict of agent ids to whether they've reached their goal,
        # false for each at start 
        self.goal_seen_by_agent = {}
        for agent in self.agents:
            self.goal_seen_by_agent[agent.id] = False
        self.collision_check_func = collision_check_func
        self.get_random_point_funcs = random_point_function

        # reset some things from RRT to account for the superstate. 
        agent_position_state_dim = len(env.size)
        if self.collision_check_func == None:
            if agent_position_state_dim == 2:
                self.collision_check_func = check_collisions
            elif agent_position_state_dim == 3:
                self.collision_check_func = check_collisions_3d
            else:
                raise ValueError("CRRT only supports 2D and 3D state positions!")
        state_distance_dim = agent_position_state_dim * len(agents)  
        self.distance_metric_state_size = state_distance_dim # overwrite from RRT
        self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size) # Overwrite from RRT
        self.agent_position_state_dim = agent_position_state_dim
        if truncate_paths:
            self.threshold = 0.
    
    def get_agent_id_order(self):
        """
        Get the order of agent ids in the superstate

        Returns:
            list(int): list of agent ids in the order they appear 
                in the superstate 
        """
        agent_id_order = np.empty(len(self.agents), dtype=np.int32)
        for index, agent in enumerate(self.agents):
            agent_id_order[index] = agent.id
        return agent_id_order
    
    def get_costs(self):
        """
        Get the costs for each agent's path if a path has been found

        Returns:
            list(float): list of costs for each agent's path,   
        """
        if(not self.path_found):
            print("Costs can't be found because goal hasn't been reached!")
            return [float('inf') for _ in range(len(self.agents))] 
        else:
            return self.path_cost

    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self.tree = nx.DiGraph()
            self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
            self._node_matrix.count = 0
            self.goal_seen_by_agent = {}
            for agent in self.agents:
                self.goal_seen_by_agent[agent.id] = False
        else:
            self.tree = some_existing_tree[0]
            self._node_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            last_added_node_id = next(reversed(self.tree._node))
            self.last_added_node_id = last_added_node_id
            self.goal_seen_by_agent = {}
            for agent in self.agents:
                self.goal_seen_by_agent[agent.id] = False
            
    def superstate_to_matrix_state(self, superstate):
        """Converts a superstate to a vector of just each agent's position
        values (i.e. not the attitude values)

        Args:
            superstate (np.array): vector of each agent's entire state (pos, attitude, etc.)

        Returns:
            np.array: vector of just each agent's position (i.e. x, y, z) 
        """
        # return _superstate_to_matrix_state_numba(
        #     superstate,
        #     len(self.agents),
        #     self.agent_position_state_dim,
        #     self.distance_metric_state_size)

        #Note: Doing the numba call here slowed things down due to python-Numba data conversion overhead.
        #The for loop is actually faster in this case.

        matrix_state = np.empty(self.distance_metric_state_size)
        dim = self.agent_position_state_dim
        for agent_counter, substate in enumerate(superstate):
            start = dim * agent_counter
            matrix_state[start:(start + dim)] = substate[:dim]
        return matrix_state
     
    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost, reached_goals = []):
        # new_node_id = len(self.tree.nodes)
        new_node_id = self.last_added_node_id + 1
        new_node = CrrtTreeNode(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost, reached_goals)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant position dimensions (e.g., x, y)
        self._node_matrix.append(self.superstate_to_matrix_state(state), new_node_id)

        return new_node_id

    def sample_random_point(self):
        """
        Samples a new joint state between all agents.

        If an agent has reached its goal, its section of the new joint 
        state is its goal point. 

        Returns:
            list(agent_state_type): New joint state 
        """
        random_state = np.empty(len(self.agents) * self.agent_position_state_dim)

        for agent_index in range(len(self.agents)):
            new_superstate_slice = slice((agent_index * self.agent_position_state_dim), 
                                         ((agent_index+1) * self.agent_position_state_dim))

            agent = self.agents[agent_index]
            r = self.rng.uniform(0, 1)
            if r < 0.1 or (r < 0.7 and self.goal_seen_by_agent[agent.id]):
                random_state[new_superstate_slice] = self.goals[agent_index][:self.agent_position_state_dim]
            else:
                random_substate = self.get_random_point_funcs[agent_index](self.env, self.static_circular_obstacles,
                                                 self.static_rectangular_obstacles, self.rng)
                random_state[new_superstate_slice] = random_substate
        
        return random_state
    
    def check_for_goals(self, parent_node, random_time, random_action, list_of_paths, new_state):
        """
        :MAINT: All the paths sent here are COLLISION FREE. This function
            a)  Checks if any or all of the agents have reached the goal state
            b)  Truncates any paths that cross their goal state
                b.i) Then re-checks for collisions    
        """
        # list of paths after update 
        updated_paths = list_of_paths
        # list of agent indices that reached the goal
        reached_goal = []
        # list of costs for each agent
        costs = [0 for _ in range(len(self.agents))] 

        for agent_index in range(len(self.agents)):
            agent = self.agents[agent_index]
            reached_goal_flag, goal_distance = self.reached_goal_funcs[agent_index](
                new_state[agent_index], 
                self.goals[agent_index], 
                self.goal_radii[agent_index], 
                agent)
            if reached_goal_flag:
                # calculate costs         
                edge_cost = self.cost_funcs[agent_index](
                        self.env, 
                        agent, 
                        parent_node.state[agent_index], 
                        random_action[agent_index], 
                        random_time, 
                        updated_paths[agent_index])
                costs[agent_index] = edge_cost

                reached_goal.append(agent.id)
            # if the path is kinda close to the goal, see if any intermediate states
            # hit the goal
            elif goal_distance < self.threshold:
                full_agent_path = np.copy(list_of_paths[agent_index])
                total_elapsed_time = parent_node.time_elapsed
                found_truncated_path_to_goal = False
                for (index, intermediate_state) in enumerate(list_of_paths[agent_index]):
                    total_elapsed_time += self.minimum_time_step
                    goal_flag, d = self.reached_goal_funcs[agent_index](
                                        intermediate_state, 
                                        self.goals[agent_index], 
                                        self.goal_radii[agent_index], 
                                        agent)
                    if goal_flag:
                        modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                        new_path_to_new_state = np.copy(full_agent_path)
                        # set the rest of the path to the last state. 
                        new_path_to_new_state[index+1:] = intermediate_state
                        updated_paths[agent_index] = new_path_to_new_state
                        # make sure this path doesn't collide with any of the others
                        if not self.collision_check_func(self, updated_paths, start_time=parent_node.time_elapsed,
                                                         start_index=index, agent_index=agent_index):
                            edge_cost = self.cost_funcs[agent_index](
                                    self.env, 
                                    agent, 
                                    parent_node.state[agent_index], 
                                    random_action[agent_index], 
                                    modified_edge_time,
                                    new_path_to_new_state)
                            new_state[agent_index] = intermediate_state
                            costs[agent_index] = edge_cost
                            reached_goal.append(agent.id)
                            found_truncated_path_to_goal = True
                            break 
                # end loop over intermediate states
                # if at this point, none of the intermediate states got to the goal
                # calculate the cost anyway 
                if not found_truncated_path_to_goal:
                    # :MAINT: this could have been modified, must set it 
                    # back to make sure the original is still there 
                    updated_paths[agent_index] = full_agent_path
                    edge_cost = self.cost_funcs[agent_index](
                                        self.env, 
                                        agent, 
                                        parent_node.state[agent_index], 
                                        random_action[agent_index], 
                                        random_time,
                                        full_agent_path)
                    costs[agent_index] = edge_cost
            else: # i.e. nowhere near goal
                # calculate costs         
                edge_cost = self.cost_funcs[agent_index](
                        self.env, 
                        agent, 
                        parent_node.state[agent_index], 
                        random_action[agent_index], 
                        random_time, 
                        updated_paths[agent_index])
                costs[agent_index] = edge_cost
        # end loop over agent ids

        # at this point, every updated path (if there were any) has 
        # been verified to be collision-free. Return the updated paths, 
        # costs, and which agents hit the goal
        return updated_paths, new_state, costs, reached_goal

    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
        Sample a time duration and a random action for each agent.
        Extend the tree towards the random point with a new joint state if possible.

        Args:
            parent_node_id: Tree node id for the parent node, or the
            state to propagate from 
            parent_node: the RRT node representation of the parent node
            random_point: point to extend to. NOT USED. 

        Returns:
            int: new node id, or very negative number if a new node couldn't be found
        """
        accept_new_state = True
        # list of paths to new state for each agent
        list_of_paths = []
        # list of actions for each agent
        random_actions = []
        # new joint state after each agent has followed 
        # its path 
        new_state = []
        # list of costs for each agent 
        costs = []

        # get a random time. Each agent uses the same random time 
        # to propagate its own action
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)       

        # get next state for each agent after propagating 
        # a random action for a (maybe) random time
        for agent_index in range(len(self.agents)):
            # if agent_index in parent_node.reached_goals:
            #     # this agent has already reached its goal, 
            #     # so just keep its state the same 
            #     random_actions.append(np.zeros(self.agents[agent_index].action_length))
            #     new_state.append(parent_node.state[agent_index])
            #     list_of_paths.append( np.array([parent_node.state[agent_index] for _ in range(num_record_steps)]) )
            #     if self.debug_flag: 
            #         print(f"Agent {agent_index} has already reached its goal. Keeping state the same.")
            #     continue
            # else:
                agent = self.agents[agent_index]
                
                random_actions.append(self.agents[agent_index].get_random_action(self.rng))
                new_substate, path_to_new_state = agent.get_next_state(parent_node.state[agent_index], 
                        random_actions[agent_index], random_time, num_record_steps)
                new_state.append(new_substate)

                # for PyBullet envs, collision checks between agents happen here
                accept_new_state = accept_new_state and self.isvalid[agent_index](
                    path_to_new_state, self.agents[agent_index].radius, self.env.size,
                    self.static_circular_obstacles,
                    self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles,
                    self.agents[agent_index].dynamic_limit_indices,
                    self.agents[agent_index].dynamic_limit_values,
                    self.env.obstacle_buffer,
                    self.env.boundary_buffer,
                    parent_node.time_elapsed,
                    random_time,
                    self.minimum_time_step)
                if not accept_new_state:
                    # this agent's action resulted in an invalid state. 
                    # exit attempting to propagate from this parent node
                    break
                
                list_of_paths.append(path_to_new_state)

        # check if the new state should be accepted based on 
        # each agent's individual path fitness as well as if 
        # any of the paths resulted in collisions between agents
        accept_new_state = accept_new_state and not self.collision_check_func(self, list_of_paths, 
                                                                              start_time=parent_node.time_elapsed)
        # print(accept_new_state,"FINAL") 
        if not accept_new_state:
            # if self.debug_flag: print("Sampled New CRRT Node is invalid. Trying again!")
            # new node is invalid, exit this round. No new nodes are added to the tree
            # and the caller will attempt to add a whole new node with a new random state, 
            # etc. 
            return -10000 # clearly not a valid node ID!
        else:
            # This new joint state is valid! 
            total_elapsed_time = parent_node.time_elapsed + random_time
            
            updated_paths, new_state, costs, reached_goal = self.check_for_goals(parent_node, 
                                                                      random_time,
                                                                      random_actions,
                                                                      list_of_paths,
                                                                      new_state)
            
            cum_cost = list( map(add, parent_node.cost_so_far, costs) )
            new_node_id = self.add_rrt_node(new_state, parent_node_id, random_actions, random_time,
                                                    updated_paths, total_elapsed_time, cum_cost, reached_goal)

            if len(reached_goal) > 0 and self.debug_flag: print("Agent found goal: " + str(reached_goal))
            # set individual agent goal flags
            for agent_id in reached_goal: self.goal_seen_by_agent[agent_id] = True 
            
            # check if total solution found!
            self.path_found = len(reached_goal) == len(self.agents)
            if(self.path_found):
                self.path_cost = cum_cost
                self.path_time = total_elapsed_time
 
            return new_node_id 
        
    def get_agent_path_times(self, high_resolution_path_numpy_array = None):
        """
        Get the total time taken for each agent's path

        Args:
            high_resolution_path_numpy_array (list(np.array), optional): 
                list of numpy arrays for each agent, each array
                is shape (num_timesteps, state_dim). If provided, the 
                time is calculated based on the number of timesteps
                in each array. If not provided, this parameter will
                be calculated. Defaults to None.
        Returns:
            list(float): list of total times for each agent's path
        """
        if high_resolution_path_numpy_array is None:
            high_resolution_path_numpy_array = self.get_high_resolution_path_numpy_array()
        
        path_times = []
        for agent_index in range(len(self.agents)):
            agent_path = high_resolution_path_numpy_array[agent_index]
            num_timesteps = agent_path.shape[0]
            last_same = num_timesteps - 1
            while last_same > 0 and np.allclose(agent_path[last_same], agent_path[last_same-1]):
                last_same -= 1
            total_time = (last_same) * self.minimum_time_step
            path_times.append(round(total_time, self.roundoff_digits))
        
        return path_times
        
    def get_high_resolution_path_numpy_array(self): 
        """
        Return a path with a smaller resolution than the one returned by get_path. 
        Each substate between each node along the path is added to the aggregate for
        each agent. 

        Returns:
            list(np.array): list of numpy arrays for each agent, each array
                is shape (num_timesteps, state_dim)
        """

        if(self.path_found == False):
            print(f"Path can't be found for agents {self.get_agent_id_order} because goal hasn't been reached!")
            return np.empty((0,self.distance_metric_state_size),dtype=np.float64)
        else:
            # init dict for each agent
            total_path_time = self.path_time
            min_time_step = self.minimum_time_step
            path_length = int( round(total_path_time/min_time_step,self.roundoff_digits)) + 1
            path_states = []
            for agent_index in range(len(self.agents)):    
                path_array = np.empty((path_length, self.agents[agent_index].state_length), dtype=np.float16)
                path_states.append(path_array)
            
            node_id = self.goal_node_id
            curr_index = path_length - 1

            while node_id != 0: # Repeat until you have reached the start node
                # print("Node ID: ", node_id)
                rrt_node = self.tree.nodes[node_id]['value']
                for agent_index in range(len(self.agents)):
                    path_to_node = rrt_node.path_from_parent[agent_index]
                    len_path_to_node = len(path_to_node) # will be same for each agent
                    start_index = curr_index - len_path_to_node + 1 # will be same for each agent  
                    path_states[agent_index][start_index:curr_index+1] = path_to_node
                curr_index -= len_path_to_node
                node_id = rrt_node.parent_id

            # Fill the remaining states with the start state
            for agent_index in range(len(self.agents)):
                path_states[agent_index][:curr_index+1] = self.starts[agent_index]
            return path_states
        
    def get_high_resolution_paths(self):
        """
        Return a path with a smaller resolution than the one returned by get_path. 
        Each substate between each node along the path is added to the aggregate for
        each agent. 

        Returns:
            list(dict(float, agent_state_type)): list of dicts from timestep to state
                for each agent 
        """
        # init dict for each agent
        max_nodes = len(self.tree.nodes)
        high_res_dicts = []
        for i in range(len(self.agents)):    
            state_data_type = get_dtype_from_input(self.starts[i])
            path_dict: Dict[int,state_data_type] = {}
            high_res_dicts.append(path_dict)
        
        # get the node ids for a valid path 
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)

        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")
            return high_res_dicts
        else:
            node_id = self.goal_node_id
            path_length = 0
            path_time = [0.0 for _ in range(len(self.agents))]          

            # move down the path from goal to start
            while node_id != -1:
                # print("Node ID: ", node_id)
                rrt_node = self.tree.nodes[node_id]['value']
                path_rrt_node_ids[path_length] = rrt_node.id
                path_length+=1
                node_id = rrt_node.parent_id
            
            # place starts
            ids = path_rrt_node_ids[:path_length][::-1]
            start_id = ids[0]
            for i in range(len(self.agents)):
                high_res_dicts[i][path_time[i]] = self.tree.nodes[start_id]['value'].state[i]
            roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

            # place the rest of the paths 
            for node_id in ids[1:]:
                rrt_node = self.tree.nodes[node_id]['value']
                for i in range(len(self.agents)):
                    path_from_parent = rrt_node.path_from_parent[i]
                    for point in path_from_parent:
                        path_time[i] = round(path_time[i] + self.minimum_time_step,roundoff_digits)
                        high_res_dicts[i][path_time[i]] = point

            return high_res_dicts

    def get_path(self):
        """
        Return RRT node ids, agent states, the executed controls, and 
        timesteps from start to goal for each agent along with a list of costs

        Returns:
            tuple(list, list, list, list, list): lists of:
                Successful path node ids for each agent
                Successful path state tuples for each agent
                Successful path control inputs for each agent
                Successful path timesteps, i.e. the time delta between each node in the path, for each agent
                    :MAINT: The timesteps are NOT cumulative!
                The cost for each agent's path. 
                ~OR~
                Empty lists if no path has been found
        """

        if(not self.path_found):
            return [],[],[],[],[]
        

        max_nodes = len(self.tree.nodes)
        num_agents = len(self.agents)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_states = []
        path_controls = []
        path_costs = []
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
            
        # init states, controls, costs 
        for i in range(num_agents):
            state_data_type = get_dtype_from_input(self.starts[i])
            curr_rng = np.random.default_rng(77)
            control_data_type = get_dtype_from_input(self.agents[i].get_random_action(curr_rng))

            path_states.append(np.empty((max_nodes, len(state_data_type)), dtype=np.float16))
            path_controls.append(np.empty((max_nodes), dtype=control_data_type))
            path_costs.append(np.empty(max_nodes, dtype=np.float64))

        node_id = self.goal_node_id
        path_length = 0            

        # populate lists
        while node_id != -1:
            # print("Node ID: ", node_id)
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_timesteps[path_length] = rrt_node.parent_action_duration

            for i in range(num_agents):
                path_states[i][path_length] = rrt_node.state[i]
                path_controls[i][path_length] = rrt_node.parent_action[i]
                path_costs[i][path_length] = rrt_node.cost_so_far[i]
            
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
        timesteps = path_timesteps[:path_length-1][::-1]

        states = []
        controls = [] 
        costs = []
        for i in range(num_agents):
            states.append(path_states[i][:path_length][::-1])
            controls.append(path_controls[i][:path_length-1][::-1])
            costs.append(path_costs[i][:path_length-1][::-1])
        return ids, states, controls, timesteps, costs

    def plan_path(self):
        """
        Plan paths through the environment for each agent 

        Returns:
            float: total planning time, fractional seconds. 
        """
        self.reset_tree()
        curr_num_steps = 0
        first_node_state = self.starts 
        self.add_rrt_node(first_node_state, 
                          -1, # parent node id 
                          [None for _ in range(len(self.agents))], # parent action 
                          None, # parent action duration 
                          [None for _ in range(len(self.agents))], # path from parent 
                          0.0, # time elapsed
                          np.zeros(len(self.agents)), # cost (per agent)
                          []) # nothing has reached the goal yet
        start_time = time.time()
        last_added_node_id = -1

        while curr_num_steps<=self.max_iter:
            # print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            last_added_node_id = self.extend_tree(nearest_node_id, nearest_node, random_point)

            curr_num_steps+=1            
            if self.path_found:   
                break

            if time.time() - start_time >= self.planning_time:
                break

        end_time = time.time()
        total_time = end_time - start_time
        if self.print_logs: print("Total Planning Time after", curr_num_steps, "steps:", total_time)
        if(self.path_found):
            self.goal_node_id = last_added_node_id
        return total_time
    
    def replan_path(self):
        """
        Replan paths through the environment for each agent by continuing
        to expand the existing tree.

        Returns:
            float: total planning time, fractional seconds. 
        """
        self.path_found = False
        self.goal_node_id = None
        self.path_time = 0.0
        self.path_cost = float('inf')

        self.goal_seen_by_agent = {}
        for agent in self.agents:
            self.goal_seen_by_agent[agent.id] = False

        if self.debug_flag or self.print_logs:
            print("Replanning the path...")

        curr_num_steps = 0
        start_time = time.time()
        last_added_node_id = -1

        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration: ", curr_num_steps)

            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            last_added_node_id = self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1

        end_time = time.time()
        total_time = end_time - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            print("Total Planning Time after", curr_num_steps, "steps:", total_time)

        if self.path_found:
            self.goal_node_id = last_added_node_id
        return total_time
