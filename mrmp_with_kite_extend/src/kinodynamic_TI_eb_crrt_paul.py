import networkx as nx
import numpy as np
from operator import add
from numba.typed import List
from numba import types

from cRRT_paul import CRRT
from kinodynamic_TI_eb_rrt import KinoTIEBTreeNode

class CrrtKinoTIEBTreeNode(KinoTIEBTreeNode):
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost, reached_goal=[]):
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,
                         path_from_parent, time_so_far, cost)
        self.reached_goal = reached_goal

class KinoTIEBCRRT(CRRT):
    def __init__(self, *, agents, starts, goals, goal_radii, env,
                    edge_bundle,
                    use_fixed_sampling_time=True, sampling_time_step=1.0, minimum_time_step=0.1, 
                    max_iter=1000, planning_time=10.0,
                    isvalid_function, 
                    cost_function, 
                    reached_goal_function, 
                    translate_function,
                    random_point_function, 
                    sort_edges_function,
                    udf_seed = 77, 
                    dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C')), 
                    collision_check_func = None,
                    num_random_edges=10, 
                    eb_kd_trees,
                    get_eb_kd_tree_query_funcs,
                    kd_tree_delta_radius = 0.5,
                    max_num_edges_per_node,
                    num_skip_edges=10,
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
            edge_bundle (list(EdgeBundle)): List of edge bundles for each agent
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
            translate_function (agent_state_type): List of functions for each agent that returns
                the end point from an edge bundle translated to start from a current agent state
                (agent_state_tuple, agent_state_tuple) -> agent_state_tuple
                :TODO: Not currently used, need to decide whether we want to use this to 
                    precompute paths/end states. 
            random_point_function (tuple(x float, y float)): List of functions for each agent that 
                generate a new point in the environment 
                (env object, list circular obstacles, list rectangular obstacles, rng object) -> tuple(x float, y float)
            sort_edges_functions: used to pick which edges get an agent closest to a random point
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
            num_random_edges (int, optional): Number of random edges to check for each agent if sorted edges fail. Defaults to 10.
            eb_kd_tree: KD tree for edge bundle lookups
            get_eb_kd_tree_query: Function to get KD tree query for edge bundle
            kd_tree_delta_radius: Radius to use for KD tree queries. Defaults to 0.5.
            max_num_edges_per_node: Max number of edges to consider per node
            num_skip_edges (int, optional): Number of edges to skip when checking sorted edges. Defaults to 10.
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false
        """

        CRRT.__init__(self, starts=starts, goals=goals, goal_radii=goal_radii, env=env, agents=agents,
                         use_fixed_sampling_time=use_fixed_sampling_time, 
                         sampling_time_step=sampling_time_step, 
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter, 
                         planning_time=planning_time,
                         isvalid_function=isvalid_function,
                         cost_function=cost_function, 
                         reached_goal_function=reached_goal_function,
                         random_point_function=random_point_function,
                         udf_seed=udf_seed,
                         print_logs=print_logs,
                         debug_flag=debug_flag,
                         collision_check_func=collision_check_func,
                         dynamic_obstacles=dynamic_obstacles)

        self.edge_bundles = edge_bundle
        self.eb_kd_trees = eb_kd_trees
        self.get_eb_kd_tree_query_funcs = get_eb_kd_tree_query_funcs
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.max_num_edges_per_node = max_num_edges_per_node
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.distance_array = [np.zeros(eb.num_edges) for eb in edge_bundle]
        self.random_indices = np.zeros(self.num_random_edges,dtype=int)
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.node_class = CrrtKinoTIEBTreeNode
        self.truncate_paths = truncate_paths

    def get_random_edge_index(self):
        return self.rng.randint(0,self.edge_bundles.num_edges)
    
    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost, reached_goal=[]):
        # new_node_id = len(self.tree.nodes)
        new_node_id = self.last_added_node_id + 1
        new_node = CrrtKinoTIEBTreeNode(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost, reached_goal)
        new_node.edge_bundle_indices = [None for _ in range(len(self.agents))]
        new_node.edge_bundle_mask = [None for _ in range(len(self.agents))]
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant position dimensions (e.g., x, y)
        self._node_matrix.append(self.superstate_to_matrix_state(state), new_node_id)

        return new_node_id

    
        
    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
        Sample a time duration and a random action for each agent.
        Extend the tree towards the random point with a new joint state if possible.

        Args:
            parent_node_id: Tree node id for the parent node, or the
            state to propagate from 
            parent_node: the RRT node representation of the parent node
            random_point: point to extend to.

        Returns:
            int: new node id, or very negative number if a new node couldn't be found
        """
        r = 0 # self.rng.integers(0,p)

        accept_new_state = True
        # list of paths to new state for each agent
        list_of_paths = []
        # list of actions for each agent
        selected_actions = []
        # new joint state after each agent has followed 
        # its path 
        new_state = []
        # list of costs for each agent 
        costs = []

        # the max and min numbers of steps on any path. All paths must
        # be extended to or retracted to one of these lengths. 
        max_steps = -1
        min_steps = 1e10
        # same for time
        max_time = -1.
        min_time = 1e10

        # iterate over each agent 
        for agent_index in range(len(self.agents)):
            num_samples = self.num_skip_edges
            if agent_index in parent_node.reached_goal:
                # this agent has already reached its goal, 
                # so just keep its state the same 
                new_state.append(parent_node.state[agent_index])
                selected_actions.append( (0.,) * self.agents[agent_index].action_length ) # no action 
                list_of_paths.append( np.array( [parent_node.state[agent_index]] ) ) # no path 
                continue

            found_valid_edge_curr_agent = False
            agent = self.agents[agent_index]
            eb = self.edge_bundles[agent_index]

            if parent_node.edge_bundle_indices[agent_index] is None:
                # Edges have not been found before for this agent for this node.
                query = self.get_eb_kd_tree_query_funcs[agent_index](parent_node.state[agent_index])
                edge_ids = self.eb_kd_trees[agent_index].radius_query(query, self.kd_tree_delta_radius)
                l = min(len(edge_ids), self.max_num_edges_per_node)
                parent_node.edge_bundle_indices[agent_index] = edge_ids[:l]
                parent_node.edge_bundle_mask[agent_index] = np.full((l,), False, dtype=bool)
    
            # Keeps track of all the edges from the bundle available for this node
            curr_edge_indices = parent_node.edge_bundle_indices[agent_index]
            # Keeps track of which edges have already been tried for this node
            curr_edge_mask = parent_node.edge_bundle_mask[agent_index]

            sorted_indices, num_valid_edges = self.sort_edges[agent_index](parent_node.state[agent_index], 
                                                          random_point[(agent_index*self.agent_position_state_dim):
                                                                       ((agent_index+1)*self.agent_position_state_dim)], 
                                                          eb.start_states,
                                                          eb.final_states,
                                                          curr_edge_indices, 
                                                          curr_edge_mask, 
                                                          self.distance_array[agent_index])
            
            p = max(1, num_valid_edges // num_samples)
            
            # check sorted edges for valid substate
            for idx in range(0, num_valid_edges, p):
                mask_index = sorted_indices[idx]
                edge_bundle_index = curr_edge_indices[mask_index]
                
                action = eb.actions[edge_bundle_index]
                timestep = eb.timesteps[edge_bundle_index]

                num_record_steps = round(timestep/self.minimum_time_step)
                new_substate, path_to_new_state = agent.get_next_state(parent_node.state[agent_index], 
                        action, timestep, num_record_steps)
                state_is_valid = self.isvalid[agent_index](
                        path_to_new_state, self.agents[agent_index].radius, self.env.size,
                        self.static_circular_obstacles,
                        self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles,
                        self.agents[agent_index].dynamic_limit_indices,
                        self.agents[agent_index].dynamic_limit_values,
                        self.env.obstacle_buffer,
                        self.env.boundary_buffer,
                        parent_node.time_elapsed,
                        timestep,
                        self.minimum_time_step)
                
                if(state_is_valid):
                    new_state.append(new_substate)
                    selected_actions.append(tuple(action))
                    list_of_paths.append(path_to_new_state)
                    found_valid_edge_curr_agent = True
                    max_steps = max(max_steps, num_record_steps)
                    min_steps = min(min_steps, num_record_steps)
                    max_time = max(max_time, timestep)
                    min_time = min(min_time, timestep)
                    break
                else:
                    # Mark this edge as tried
                    curr_edge_mask[mask_index] = True
                    if self.debug_flag:
                        print(f"~~~~~~~~~~Sampled New State in Kino TI EB cRRT for agent {agent_index} is invalid. Trying again!~~~~~~~~~~")
                        print("Invalid State :", new_substate)
            # end iter over sorted edges

            # if there are no valid edges from sorted edges, stop trying to find a valid substate and superstate
            if not found_valid_edge_curr_agent and num_valid_edges <= 0:
                if self.debug_flag:
                    print(f"~~~~~~~~~~No valid new substate options found for agent {agent_index}. Stop planning superstate!~~~~~~~~~~")
                    print("Parent Subst`ate :", parent_node.state[agent_index])
                accept_new_state = False
                break # loop over agents

            # check some random edges for valid substate if one hasn't been found yet
            if not found_valid_edge_curr_agent:
                num_samples = min(self.num_random_edges, num_valid_edges)
                random_slots = self.rng.integers(0, num_valid_edges, size=num_samples)

                for slot in random_slots:
                    mask_index = sorted_indices[slot]  # map 0..num_valid_edges-1 → actual index into bundle

                    # Optional: if we *ever* want to retry already-used edges, we can comment:
                    if curr_edge_mask[mask_index]:
                        continue # loop over random edges/slots

                    edge_bundle_index = curr_edge_indices[mask_index]
                    action = eb.actions[edge_bundle_index]
                    timestep = eb.timesteps[edge_bundle_index]
                    num_record_steps = round(timestep/self.minimum_time_step)
                    new_substate, path_to_new_state = agent.get_next_state(parent_node.state[agent_index], 
                            action, timestep, num_record_steps)
                    state_is_valid = self.isvalid[agent_index](
                            path_to_new_state, self.agents[agent_index].radius, self.env.size,
                            self.static_circular_obstacles,
                            self.static_rectangular_obstacles,
                            self.dynamic_agent_obstacles,
                            self.agents[agent_index].dynamic_limit_indices,
                            self.agents[agent_index].dynamic_limit_values,
                            self.env.obstacle_buffer,
                            self.env.boundary_buffer,
                            parent_node.time_elapsed,
                            timestep,
                            self.minimum_time_step)
                    
                    if(state_is_valid):
                        new_state.append(new_substate)
                        selected_actions.append(tuple(action))
                        list_of_paths.append(path_to_new_state)
                        found_valid_edge_curr_agent = True
                        max_steps = max(max_steps, num_record_steps)
                        min_steps = min(min_steps, num_record_steps)
                        max_time = max(max_time, timestep)      
                        min_time = min(min_time, timestep)
                        break # loop over random edges/slots
                    else:
                        # Mark this edge as tried
                        curr_edge_mask[mask_index] = True
                        if self.debug_flag:
                            print(f"~~~~~~~~~~Sampled New Random State in Kino TI EB cRRT for agent {agent_index} is invalid. Trying again!~~~~~~~~~~")
                            print("Invalid State :", new_substate)
            # end condition sorted edges did not produce a valid state
            
            # if the sorted edges and random edges both don't produce a valid
            # state/path for this agent, stop trying to find a valid superstate
            # for this iter. Return a bogus value and start again
            if not found_valid_edge_curr_agent:
                accept_new_state = False
                break
        # end loop over agents 


        # check if the new state should be accepted based on 
        # each agent's individual path fitness as well as if 
        # any of the paths resulted in collisions between agents
        accept_new_state = accept_new_state and not self.collision_check_func(self, list_of_paths)
        # print(accept_new_state,"FINAL") 
        if not accept_new_state:
            if self.debug_flag: print("Sampled New CRRT Node/Superstate is invalid. Trying again!")
            # new node is invalid, exit this round. No new nodes are added to the tree
            # and the caller will attempt to add a whole new node with a new random state, 
            # etc. 
            return -10000 # clearly not a valid node ID!
        else: # This new joint state is valid! 
            # Make path lengths equal by truncating or extending paths
            time_used = max_time
            steps_used = max_steps
            if self.truncate_paths:
                time_used = min_time
                steps_used = min_steps
            for path_idx in range(len(list_of_paths)):
                if path_idx in parent_node.reached_goal:
                    list_of_paths[path_idx] = np.full((steps_used, len(list_of_paths[path_idx][0])), \
                                                list_of_paths[path_idx][-1])
                elif self.truncate_paths:
                    # truncate longer paths to match min
                    list_of_paths[path_idx] = list_of_paths[path_idx][:min_steps]
                    new_state[path_idx] = list_of_paths[path_idx][-1]
                elif len(list_of_paths[path_idx]) < max_steps:
                    # extend shorter paths to match max
                    list_of_paths[path_idx] = np.concatenate(
                        (list_of_paths[path_idx], np.full((max_steps - len(list_of_paths[path_idx]), len(list_of_paths[path_idx][0])), 
                                                list_of_paths[path_idx][-1])))

            total_elapsed_time = parent_node.time_elapsed + time_used
            
            updated_paths, new_state, costs, reached_goal = self.check_for_goals(parent_node, 
                                                                      time_used,
                                                                      selected_actions,
                                                                      list_of_paths,
                                                                      new_state)
            
            cum_cost = list( map(add, parent_node.cost_so_far, costs) )
            new_node_id = self.add_rrt_node(new_state, parent_node_id, selected_actions, time_used,
                                                    updated_paths, total_elapsed_time, cum_cost, reached_goal)

            if len(reached_goal) > 0 and self.print_logs: print("Agents found goal " + str(reached_goal) 
                                                                + " out of those that have " + str(self.reached_goal))
            # set individual agent goal flags
            for agent_id in reached_goal: self.reached_goal[agent_id] = True 
            
            # check if total solution found!
            self.path_found = len(reached_goal) == len(self.agents)
            if(self.path_found):
                self.path_cost = cum_cost
                self.path_time = total_elapsed_time
 
            return new_node_id 
        
    
    
