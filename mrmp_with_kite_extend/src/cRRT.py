import networkx as nx
import numpy as np
from operator import add
import time
from numba.typed import List
from numba import njit, types

from rrt import TreeNode, RRT, DynamicMatrix
from utils import get_dtype_from_input,find_roundoff_decimal_digits, \
     point_circle_collision, point_sphere_collision, euclidean_distance_numba_with_l

@njit(cache=True)
def _joint_state_to_matrix_state_numba(joint_state, agent_state_starts,
                                       num_agents,
                                       agent_position_state_dim,
                                       distance_metric_state_size):
    matrix_state = np.empty(distance_metric_state_size)
    for agent_counter in range(num_agents):
        joint_state_start = agent_state_starts[agent_counter]
        matrix_state_start = agent_position_state_dim * agent_counter
        for state_dim in range(agent_position_state_dim):
            matrix_state[matrix_state_start + state_dim] = joint_state[joint_state_start + state_dim]
    return matrix_state

@njit(cache=True)
def _joint_path_collides_2d_numba(joint_path, agent_state_starts, agent_radii,
                                  dynamic_agent_clearance, start_index):
    num_steps = joint_path.shape[0]
    num_agents = agent_state_starts.shape[0] - 1
    for state_index in range(start_index, num_steps):
        for first_agent in range(num_agents):
            first_start = agent_state_starts[first_agent]
            first_x = joint_path[state_index, first_start]
            first_y = joint_path[state_index, first_start + 1]
            first_radius = agent_radii[first_agent]
            for second_agent in range(first_agent + 1, num_agents):
                second_start = agent_state_starts[second_agent]
                radius_sum = first_radius + agent_radii[second_agent] + dynamic_agent_clearance
                dx = first_x - joint_path[state_index, second_start]
                dy = first_y - joint_path[state_index, second_start + 1]
                if dx * dx + dy * dy <= radius_sum * radius_sum:
                    return True
    return False

@njit(cache=True)
def _first_joint_path_collision_2d_numba(joint_path, agent_state_starts,
                                         agent_radii, dynamic_agent_clearance,
                                         start_index):
    num_steps = joint_path.shape[0]
    num_agents = agent_state_starts.shape[0] - 1
    for state_index in range(start_index, num_steps):
        for first_agent in range(num_agents):
            first_start = agent_state_starts[first_agent]
            first_x = joint_path[state_index, first_start]
            first_y = joint_path[state_index, first_start + 1]
            first_radius = agent_radii[first_agent]
            for second_agent in range(first_agent + 1, num_agents):
                second_start = agent_state_starts[second_agent]
                radius_sum = first_radius + agent_radii[second_agent] + dynamic_agent_clearance
                dx = first_x - joint_path[state_index, second_start]
                dy = first_y - joint_path[state_index, second_start + 1]
                if dx * dx + dy * dy <= radius_sum * radius_sum:
                    return True, first_agent, second_agent, state_index
    return False, -1, -1, -1

@njit(cache=True)
def _joint_path_collides_3d_numba(joint_path, agent_state_starts, agent_radii,
                                  dynamic_agent_clearance, start_index):
    num_steps = joint_path.shape[0]
    num_agents = agent_state_starts.shape[0] - 1
    for state_index in range(start_index, num_steps):
        for first_agent in range(num_agents):
            first_start = agent_state_starts[first_agent]
            first_x = joint_path[state_index, first_start]
            first_y = joint_path[state_index, first_start + 1]
            first_z = joint_path[state_index, first_start + 2]
            first_radius = agent_radii[first_agent]
            for second_agent in range(first_agent + 1, num_agents):
                second_start = agent_state_starts[second_agent]
                radius_sum = first_radius + agent_radii[second_agent] + dynamic_agent_clearance
                dx = first_x - joint_path[state_index, second_start]
                dy = first_y - joint_path[state_index, second_start + 1]
                dz = first_z - joint_path[state_index, second_start + 2]
                if dx * dx + dy * dy + dz * dz <= radius_sum * radius_sum:
                    return True
    return False

@njit(cache=True)
def _first_joint_path_collision_3d_numba(joint_path, agent_state_starts,
                                         agent_radii, dynamic_agent_clearance,
                                         start_index):
    num_steps = joint_path.shape[0]
    num_agents = agent_state_starts.shape[0] - 1
    for state_index in range(start_index, num_steps):
        for first_agent in range(num_agents):
            first_start = agent_state_starts[first_agent]
            first_x = joint_path[state_index, first_start]
            first_y = joint_path[state_index, first_start + 1]
            first_z = joint_path[state_index, first_start + 2]
            first_radius = agent_radii[first_agent]
            for second_agent in range(first_agent + 1, num_agents):
                second_start = agent_state_starts[second_agent]
                radius_sum = first_radius + agent_radii[second_agent] + dynamic_agent_clearance
                dx = first_x - joint_path[state_index, second_start]
                dy = first_y - joint_path[state_index, second_start + 1]
                dz = first_z - joint_path[state_index, second_start + 2]
                if dx * dx + dy * dy + dz * dz <= radius_sum * radius_sum:
                    return True, first_agent, second_agent, state_index
    return False, -1, -1, -1

class CrrtTreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost, reached_goals):
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,\
                         path_from_parent, time_so_far, cost)
        self.reached_goals = reached_goals
        
class CRRT(RRT):
    def __init__(self, *, agents, starts, goals, goal_radii, env,
                    use_fixed_sampling_time=True, sampling_time_step=1.0, minimum_time_step=0.1, 
                    max_iter=1000, planning_time=10.0,
                    num_extension_trials=1,
                    truncation_check_threshold = 1.0,
                    isvalid_function, 
                    cost_function, 
                    reached_goal_function, 
                    random_point_function, 
                    udf_seed = 77, 
                    dynamic_agent_clearance=0.0,
                    dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C')), 
                    truncate_paths = False,
                    branch_goal_parking = True,
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
            dynamic_obstacles (numba list, optional): List of dynamic obstacles to avoid. Defaults to empty
            truncate_paths (bool, optional): If True, keep the original
                terminal-edge truncation behavior. Defaults to False.
            branch_goal_parking (bool, optional): If True, goal completion is
                branch-local: agents that reach their goal at an accepted edge
                endpoint stay parked for descendant expansions on that branch.
                Defaults to True. This mode is endpoint-only and requires
                truncate_paths=False.
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false
        """
        # Goal handling modes:
        # 1. branch_goal_parking=True, truncate_paths=False:
        #    New MRMP default. Goal hits are endpoint-only and branch-local;
        #    reached agents stay parked for descendant expansions.
        # 2. branch_goal_parking=False, truncate_paths=True:
        #    Original cRRT behavior. No branch-local parking; terminal edges
        #    may be truncated if all agents reach goals inside the same edge.
        # 3. branch_goal_parking=False, truncate_paths=False:
        #    Strict endpoint-only behavior. A solution is accepted only when
        #    all agents end the same accepted edge in their goal regions.
        if branch_goal_parking and truncate_paths:
            raise ValueError(
                "branch_goal_parking requires truncate_paths=False")

        # init the superclass, most of which will not be used
        RRT.__init__(self, start=starts[0], goal=goals[0], goal_radius=goal_radii[0], env=env, agent=agents[0],
                         use_fixed_sampling_time=use_fixed_sampling_time, 
                         sampling_time_step=sampling_time_step, 
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter, 
                         planning_time=planning_time,
                         num_extension_trials=num_extension_trials,
                         isvalid_function=isvalid_function[0],
                         cost_function=cost_function[0], 
                         reached_goal_function=reached_goal_function[0],
                         random_point_function=random_point_function[0],
                         udf_seed=udf_seed,
                         dynamic_agent_clearance=dynamic_agent_clearance,
                         print_logs=print_logs,
                         debug_flag=debug_flag,
                         dynamic_obstacles=dynamic_obstacles)

        self.agents = agents
        self.starts = [np.array(start) for start in starts]
        self.goals = [np.array(goal) for goal in goals]
        self.goal_radii = goal_radii
        self.num_agents = len(self.agents)
        self.agent_id_to_index = {agent.id: index for index, agent in enumerate(self.agents)}

        self.agent_state_lengths = np.array([len(start) for start in self.starts], dtype=np.int32)
        self.agent_state_starts = np.empty(len(self.agents) + 1, dtype=np.int32)
        self.agent_state_starts[0] = 0
        for agent_index in range(len(self.agents)):
            self.agent_state_starts[agent_index + 1] = (
                self.agent_state_starts[agent_index] + self.agent_state_lengths[agent_index])
        self.joint_state_size = self.agent_state_starts[-1]
        self.start_joint_state = self.agent_states_to_joint_state(self.starts)

        self.agent_radii = np.array([agent.radius for agent in self.agents], dtype=np.float64)

        self.agent_action_lengths = np.array([agent.action_length for agent in self.agents], dtype=np.int32)
        self.agent_action_starts = np.empty(len(self.agents) + 1, dtype=np.int32)
        self.agent_action_starts[0] = 0
        for agent_index in range(len(self.agents)):
            self.agent_action_starts[agent_index + 1] = (
                self.agent_action_starts[agent_index] + self.agent_action_lengths[agent_index])
        self.joint_action_size = self.agent_action_starts[-1]

        self.planning_time = planning_time #Time to plan the path in seconds
        self.isvalid = isvalid_function #Functions to check if the path is valid
        self.cost_funcs = cost_function #Functions to calculate the cost of a edge
        self.reached_goal_funcs = reached_goal_function
        self.branch_goal_parking = bool(branch_goal_parking)

        # Whether each agent has reached its goal, indexed by superstate order.
        self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)
        self.get_random_point_funcs = random_point_function

        # reset some things from RRT to account for the superstate. 
        agent_position_state_dim = len(env.size)
        if agent_position_state_dim not in (2, 3):
            raise ValueError("CRRT only supports 2D and 3D state positions!")
        state_distance_dim = agent_position_state_dim * len(agents)  
        self.distance_metric_state_size = state_distance_dim # overwrite from RRT
        self._node_matrix = DynamicMatrix(initial_capacity=1024, 
                            dim=self.distance_metric_state_size) # Overwrite from RRT
        self.agent_position_state_dim = agent_position_state_dim
        self.goal_matrix_state = np.empty(self.distance_metric_state_size)
        for agent_index in range(self.num_agents):
            position_start = agent_index * self.agent_position_state_dim
            position_end = position_start + self.agent_position_state_dim
            self.goal_matrix_state[position_start:position_end] = (
                self.goals[agent_index][:self.agent_position_state_dim])
        if truncate_paths:
            self.threshold = truncation_check_threshold * np.sqrt(self.num_agents)
        else:
            self.threshold = 0.0

    def _agent_is_parked(self, parent_node, agent_index):
        return (self.branch_goal_parking and
                parent_node.reached_goals[agent_index])

    def _fill_parked_agent(self, joint_action, joint_path, new_joint_state,
                           parent_node, agent_index):
        agent_state = self.get_agent_state(parent_node.state, agent_index)
        self.set_agent_action(
            joint_action,
            agent_index,
            np.zeros(self.agent_action_lengths[agent_index]))
        self.set_agent_state(new_joint_state, agent_index, agent_state)
        agent_slice = self.get_agent_state_slice(agent_index)
        joint_path[:, agent_slice] = agent_state
    
    def get_agent_state_slice(self, agent_index):
        return slice(self.agent_state_starts[agent_index],
                     self.agent_state_starts[agent_index + 1])

    def get_agent_state(self, joint_state, agent_index):
        # return joint_state[self.get_agent_state_slice(agent_index)]
        start = self.agent_state_starts[agent_index]
        end = self.agent_state_starts[agent_index + 1]
        return joint_state[start:end]

    def set_agent_state(self, joint_state, agent_index, agent_state):
        # joint_state[self.get_agent_state_slice(agent_index)] = agent_state
        start = self.agent_state_starts[agent_index]
        end = self.agent_state_starts[agent_index + 1]
        joint_state[start:end] = agent_state

    def agent_states_to_joint_state(self, agent_states):
        joint_state = np.empty(self.joint_state_size)
        for agent_index in range(len(self.agents)):
            self.set_agent_state(joint_state, agent_index, agent_states[agent_index])
        return joint_state

    def joint_state_to_agent_states(self, joint_state):
        agent_states = []
        for agent_index in range(len(self.agents)):
            agent_states.append(np.copy(self.get_agent_state(joint_state, agent_index)))
        return agent_states

    def get_agent_path(self, joint_path, agent_index):
        return joint_path[:, self.get_agent_state_slice(agent_index)]

    def set_agent_path(self, joint_path, agent_index, agent_path):
        joint_path[:, self.get_agent_state_slice(agent_index)] = agent_path

    def joint_path_collides(self, joint_path, start_index=0):
        if self.agent_position_state_dim == 2:
            return _joint_path_collides_2d_numba(
                joint_path,
                self.agent_state_starts,
                self.agent_radii,
                self.dynamic_agent_clearance,
                start_index)
        return _joint_path_collides_3d_numba(
            joint_path,
            self.agent_state_starts,
            self.agent_radii,
            self.dynamic_agent_clearance,
            start_index)

    def first_joint_path_collision(self, joint_path, start_index=0):
        if self.agent_position_state_dim == 2:
            return _first_joint_path_collision_2d_numba(
                joint_path,
                self.agent_state_starts,
                self.agent_radii,
                self.dynamic_agent_clearance,
                start_index)
        return _first_joint_path_collision_3d_numba(
            joint_path,
            self.agent_state_starts,
            self.agent_radii,
            self.dynamic_agent_clearance,
            start_index)

    def get_agent_action_slice(self, agent_index):
        return slice(self.agent_action_starts[agent_index],
                     self.agent_action_starts[agent_index + 1])

    def get_agent_action(self, joint_action, agent_index):
        # return joint_action[self.get_agent_action_slice(agent_index)]
        start = self.agent_action_starts[agent_index]
        end = self.agent_action_starts[agent_index + 1]
        return joint_action[start:end]

    def set_agent_action(self, joint_action, agent_index, agent_action):
        # joint_action[self.get_agent_action_slice(agent_index)] = agent_action
        start = self.agent_action_starts[agent_index]
        end = self.agent_action_starts[agent_index + 1]
        joint_action[start:end] = agent_action

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
            return np.array([np.inf for _ in range(len(self.agents))])
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
            self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)
        else:
            self.tree = some_existing_tree[0]
            self._node_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            last_added_node_id = next(reversed(self.tree._node))
            self.last_added_node_id = last_added_node_id
            self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)
            
    def superstate_to_matrix_state(self, superstate):
        """Converts a superstate to a vector of just each agent's position
        values (i.e. not the attitude values)

        Args:
            superstate (np.array): vector of each agent's entire state (pos, attitude, etc.)

        Returns:
            np.array: vector of just each agent's position (i.e. x, y, z) 
        """
        return _joint_state_to_matrix_state_numba(
            superstate,self.agent_state_starts,len(self.agents),
            self.agent_position_state_dim,self.distance_metric_state_size)
     
    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost, reached_goals):
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
        random_state = np.empty(self.distance_metric_state_size)
        position_dim = self.agent_position_state_dim

        for agent_index in range(self.num_agents):
            position_start = agent_index * position_dim
            position_end = position_start + position_dim

            r = self.rng.uniform(0, 1)
            if r < 0.1 or (r < 0.7 and self.goal_seen_by_agent[agent_index]):
                random_state[position_start:position_end] = self.goal_matrix_state[position_start:position_end]
            else:
                random_state[position_start:position_end] = self.get_random_point_funcs[agent_index](
                    self.env,
                    self.static_circular_obstacles,
                    self.static_rectangular_obstacles,
                    self.rng)
        
        return random_state
    
    def check_for_goals(self, parent_node, random_time, random_action,
                        joint_path, new_joint_state):
        """
        Check endpoint goals and, only for a terminal joint solution, truncate
        the whole joint edge to a common finish time. Earlier-arriving agents
        may be held at their first goal state only on this terminal edge.
        """
        if self.branch_goal_parking:
            reached_goal = parent_node.reached_goals.copy()
        else:
            reached_goal = np.zeros(self.num_agents, dtype=np.bool_)
        costs = np.zeros(self.num_agents)

        for agent_index in range(self.num_agents):
            if self._agent_is_parked(parent_node, agent_index):
                continue

            agent = self.agents[agent_index]
            agent_path = self.get_agent_path(joint_path, agent_index)
            parent_agent_state = self.get_agent_state(parent_node.state, agent_index)
            agent_action = self.get_agent_action(random_action, agent_index)
            agent_state = self.get_agent_state(new_joint_state, agent_index)

            reached_goal_flag, _ = self.reached_goal_funcs[agent_index](
                agent_state, self.goals[agent_index],
                self.goal_radii[agent_index], agent)
            reached_goal[agent_index] = reached_goal_flag
            costs[agent_index] = self.cost_funcs[agent_index](
                self.env, agent, parent_agent_state,
                agent_action, random_time, agent_path)

        #If the branch_goal_parking mode is on, we consider goals to be 
        #branch-local and don't do any truncation checks.
        if self.branch_goal_parking:
            return joint_path, new_joint_state, costs, reached_goal, random_time

        #If all agents have reached their goals, we can return early 
        #without doing any truncation checks.
        if np.all(reached_goal):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        endpoint_matrix_state = self.superstate_to_matrix_state(new_joint_state)
        joint_goal_distance = euclidean_distance_numba_with_l(
            endpoint_matrix_state,
            self.goal_matrix_state,
            self.distance_metric_state_size)

        if self.threshold <= 0.0 or joint_goal_distance >= self.threshold:
            return joint_path, new_joint_state, costs, reached_goal, random_time

        first_goal_indices = np.full(self.num_agents, -1, dtype=np.int32)

        #Find the first index in the joint path where each agent reaches 
        #its goal. If an agent doesn't reach its goal at any point in the 
        #path, its index remains -1.
        for agent_index in range(self.num_agents):
            agent = self.agents[agent_index]
            agent_path = self.get_agent_path(joint_path, agent_index)
            for index in range(agent_path.shape[0]):
                goal_flag, _ = self.reached_goal_funcs[agent_index](
                    agent_path[index], self.goals[agent_index],
                    self.goal_radii[agent_index], agent)
                if goal_flag:
                    first_goal_indices[agent_index] = index
                    break

        #If any agent doesn't reach the goal at any point in the path,
        #return the original path and state without modification.
        #We only do this truncation process if all agents reach the goal 
        #at some point in the path and the endpoint is close enough 
        #to the goal in the same expansion step.
        if np.any(first_goal_indices < 0):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        #This becomes the terminal time for the whole joint edge. 
        joint_goal_index = int(np.max(first_goal_indices))
        first_collision_check_index = int(np.min(first_goal_indices))
        terminal_joint_path = np.empty(
            (joint_goal_index + 1, self.joint_state_size),
            dtype=joint_path.dtype)
        terminal_joint_path[:, :] = joint_path[:joint_goal_index + 1]

        #For agents that reach their goal before the joint goal index,
        #hold them at their first goal state for the rest of the path.
        for agent_index in range(self.num_agents):
            first_goal_index = int(first_goal_indices[agent_index])
            if first_goal_index < joint_goal_index:
                agent_slice = self.get_agent_state_slice(agent_index)
                first_goal_state = joint_path[first_goal_index, agent_slice]
                terminal_joint_path[first_goal_index + 1:, agent_slice] = first_goal_state

        #Check the truncated path for collisions. If it collides,
        #return the original path and state without modification.
        if self.joint_path_collides(terminal_joint_path,
                                    start_index=first_collision_check_index):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        #If the code reaches this point, the truncated path is valid 
        #and the last state of this path is a joint goal state.
        modified_edge_time = (joint_goal_index + 1) * self.minimum_time_step
        terminal_costs = np.zeros(self.num_agents)
        terminal_joint_state = terminal_joint_path[-1].copy()
        for agent_index in range(self.num_agents):
            agent = self.agents[agent_index]
            parent_agent_state = self.get_agent_state(parent_node.state, agent_index)
            agent_action = self.get_agent_action(random_action, agent_index)
            terminal_costs[agent_index] = self.cost_funcs[agent_index](
                self.env, agent, parent_agent_state,
                agent_action, modified_edge_time,
                self.get_agent_path(terminal_joint_path, agent_index))

        reached_goal[:] = True
        return terminal_joint_path, terminal_joint_state, terminal_costs, reached_goal, modified_edge_time

    def _select_best_joint_extension_candidate(self, parent_node, random_point):
        """
        Try several joint rollouts and keep the valid endpoint closest to
        the sampled joint random point, mirroring vanilla RRT's k trials.
        """
        best_candidate = None
        best_score = np.inf

        for _ in range(self.num_extension_trials):
            accept_new_state = True
            joint_action = np.empty(self.joint_action_size)
            new_joint_state = np.empty(self.joint_state_size)

            random_time = self.get_time()
            num_record_steps = max(1, round(random_time / self.minimum_time_step))
            random_time = num_record_steps * self.minimum_time_step
            joint_path = np.empty((num_record_steps, self.joint_state_size))

            for agent_index in range(self.num_agents):
                if self._agent_is_parked(parent_node, agent_index):
                    self._fill_parked_agent(
                        joint_action, joint_path, new_joint_state,
                        parent_node, agent_index)
                    continue

                agent = self.agents[agent_index]
                parent_agent_state = self.get_agent_state(parent_node.state, agent_index)
                agent_action = agent.get_random_action(self.rng)
                self.set_agent_action(joint_action, agent_index, agent_action)

                new_substate, path_to_new_state = agent.get_next_state(
                    parent_agent_state, agent_action,
                    random_time, num_record_steps)
                self.set_agent_state(new_joint_state, agent_index, new_substate)
                self.set_agent_path(joint_path, agent_index, path_to_new_state)

                accept_new_state = accept_new_state and self.isvalid[agent_index](
                    path_to_new_state, agent.radius, self.env.size,
                    self.static_circular_obstacles,
                    self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles,
                    agent.dynamic_limit_indices,
                    agent.dynamic_limit_values,
                    self.env.obstacle_buffer,
                    self.dynamic_agent_clearance,
                    self.env.boundary_buffer,
                    parent_node.time_elapsed,
                    random_time,
                    self.minimum_time_step)
                if not accept_new_state:
                    break

            if not accept_new_state:
                continue

            if self.joint_path_collides(joint_path):
                continue

            # The new joint state is valid and collision-free.
            matrix_state = self.superstate_to_matrix_state(new_joint_state)
            score = euclidean_distance_numba_with_l(matrix_state, random_point,
                self.distance_metric_state_size)

            if score < best_score:
                best_score = score
                best_candidate = (new_joint_state, joint_path,
                    joint_action, random_time)

        return best_candidate

    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
        Sample several joint actions and add the valid rollout whose endpoint
        is closest to the sampled joint random point.
        """
        best_candidate = self._select_best_joint_extension_candidate(
            parent_node, random_point)

        if best_candidate is None:
            return

        new_joint_state, joint_path, joint_action, random_time = best_candidate

        updated_path, new_joint_state, costs, reached_goal, edge_time = self.check_for_goals(
            parent_node,
            random_time,
            joint_action,
            joint_path,
            new_joint_state)

        total_elapsed_time = parent_node.time_elapsed + edge_time
        combined_cost = parent_node.cost_so_far + costs
        new_node_id = self.add_rrt_node(
            new_joint_state,parent_node_id,
            joint_action,edge_time,
            updated_path,
            total_elapsed_time,
            combined_cost,
            reached_goal)

        if np.any(reached_goal) and self.debug_flag:
            print("Agent found goal: " + str(np.flatnonzero(reached_goal)))
        self.goal_seen_by_agent |= reached_goal

        self.path_found = np.all(reached_goal)
        if self.path_found:
            self.goal_node_id = new_node_id
            self.path_cost = combined_cost
            self.path_time = total_elapsed_time

        return

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
        Return the time-discretized solution path for each agent.

        Internally, each tree edge now stores one 2D joint path with shape
        (num_timesteps, joint_state_size). This method preserves the old public
        return shape: a list of per-agent arrays.

        Returns:
            list(np.array): list of numpy arrays for each agent, each array
                is shape (num_timesteps, agent_state_dim)
        """

        if self.path_found == False:
            print(f"Path can't be found for agents {self.get_agent_id_order} because goal hasn't been reached!")
            return np.empty((0, self.distance_metric_state_size), dtype=np.float64)

        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_num_nodes = 0
        num_path_rows = 1

        node_id = self.goal_node_id
        while node_id != -1:
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_num_nodes] = rrt_node.id
            if rrt_node.parent_id != -1:
                num_path_rows += rrt_node.path_from_parent.shape[0]
            path_num_nodes += 1
            node_id = rrt_node.parent_id

        path_states = []
        for agent_index in range(self.num_agents):
            path_states.append(np.empty(
                (num_path_rows, self.agent_state_lengths[agent_index]),
                dtype=np.float64))

        ids = path_rrt_node_ids[:path_num_nodes][::-1]
        start_node = self.tree.nodes[ids[0]]['value']
        for agent_index in range(self.num_agents):
            path_states[agent_index][0] = self.get_agent_state(
                start_node.state, agent_index)

        curr_index = 1
        for node_id in ids[1:]:
            rrt_node = self.tree.nodes[node_id]['value']
            joint_path = rrt_node.path_from_parent
            len_path_to_node = joint_path.shape[0]
            next_index = curr_index + len_path_to_node
            for agent_index in range(self.num_agents):
                path_states[agent_index][curr_index:next_index] = (
                    self.get_agent_path(joint_path, agent_index))
            curr_index = next_index

        return path_states
        
    def get_high_resolution_paths(self):
        """
        Return the time-discretized solution path for each agent as dictionaries.

        Returns:
            list(dict(float, agent_state_type)): list of dicts from timestep to state
                for each agent 
        """
        high_res_dicts = []
        for _ in range(self.num_agents):
            high_res_dicts.append({})
        
        if self.path_found == False:
            print("Path can't be found because goal hasn't been reached!")
            return high_res_dicts

        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_length = 0

        node_id = self.goal_node_id
        while node_id != -1:
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_length += 1
            node_id = rrt_node.parent_id
        
        ids = path_rrt_node_ids[:path_length][::-1]
        start_node = self.tree.nodes[ids[0]]["value"]
        path_time = np.zeros(self.num_agents, dtype=np.float64)
        for i in range(self.num_agents):
            high_res_dicts[i][0.0] = self.get_agent_state(start_node.state, i).copy()

        roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        for node_id in ids[1:]:
            rrt_node = self.tree.nodes[node_id]['value']
            joint_path = rrt_node.path_from_parent
            for i in range(self.num_agents):
                agent_path = self.get_agent_path(joint_path, i)
                for point_index in range(agent_path.shape[0]):
                    path_time[i] = round(
                        path_time[i] + self.minimum_time_step,
                        roundoff_digits)
                    high_res_dicts[i][float(path_time[i])] = agent_path[point_index].copy()

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
            path_states.append(np.empty((max_nodes, len(state_data_type)), dtype=np.float16))
            path_controls.append(np.empty((max_nodes, self.agent_action_lengths[i]), dtype=np.float64))
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
                path_states[i][path_length] = self.get_agent_state(rrt_node.state, i)
                path_controls[i][path_length] = self.get_agent_action(rrt_node.parent_action, i)
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
        first_node_state = self.start_joint_state 
        self.add_rrt_node(first_node_state, 
                          -1, # parent node id 
                          np.zeros(self.joint_action_size), # parent action 
                          None, # parent action duration 
                          np.empty((0, self.joint_state_size)), # path from parent 
                          0.0, # time elapsed
                          np.zeros(len(self.agents)), # cost (per agent)
                          np.zeros(self.num_agents, dtype=np.bool_)) # nothing has reached the goal yet
        start_time = time.time()
        while curr_num_steps<=self.max_iter:
            # print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            curr_num_steps+=1            
            if self.path_found:   
                break

            if time.time() - start_time >= self.planning_time:
                break

        end_time = time.time()
        total_time = end_time - start_time
        if self.print_logs: 
            print("Total Planning Time after", curr_num_steps, "steps:", total_time)
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

        self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)

        if self.debug_flag or self.print_logs:
            print("Replanning the path...")

        curr_num_steps = 0
        start_time = time.time()
        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration: ", curr_num_steps)

            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1

        end_time = time.time()
        total_time = end_time - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            print("Total Planning Time after", curr_num_steps, "steps:", total_time)

        return total_time
