import networkx as nx
import numpy as np
from operator import add
import time
from numba.typed import List
from numba import types

from cRRT_paul import CRRT, check_collisions
from utils import get_dtype_from_input,find_roundoff_decimal_digits, \
     point_circle_collision

class CRRT_EBType2(CRRT):
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
                    collision_check_func = check_collisions,
                    num_random_edges=10, 
                    num_skip_edges=10,
                    truncate_paths = False,
                    dynamic_agent_clearance=0.0,
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
            num_random
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false
        """

        # init the superclass, most of which will not be used
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
                         dynamic_agent_clearance=dynamic_agent_clearance,
                         dynamic_obstacles=dynamic_obstacles)

        self.edge_bundle = edge_bundle
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.distance_array = [np.zeros(eb.num_edges) for eb in edge_bundle]
        self.random_indices = np.zeros(self.num_random_edges,dtype=int)
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.truncate_paths = truncate_paths

        
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
        p = self.num_skip_edges
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

        # the max number of steps on any path. All paths must
        # be extended to this length. 
        max_steps = -1
        # same for time
        max_time = -1.

        # iterate over each agent 
        for agent_index in range(len(self.agents)):
            found_valid_edge_curr_agent = False
            agent = self.agents[agent_index]

            eb = self.edge_bundle[agent_index]
            sorted_indices = self.sort_edges[agent_index](parent_node.state[agent_index], 
                                                          random_point[(agent_index*self.agent_position_state_dim):
                                                                       ((agent_index+1)*self.agent_position_state_dim)], 
                                                          eb.final_states, 
                                                          self.distance_array[agent_index])
            
            # check sorted edges for valid substate
            for idx,x in enumerate(sorted_indices[r::p]):
                action = eb.actions[x]
                timestep = eb.timesteps[x]
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
                        self.dynamic_agent_clearance,
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
                    max_time = max(max_time, timestep)
                    break
            # end iter over sorted edges

            # check some random edges for valid substate if one hasn't been found yet
            if not found_valid_edge_curr_agent:
                random_indices = self.rng.integers(0,eb.num_edges,size=self.num_random_edges)

                for idx,x in enumerate(random_indices):
                    action = eb.actions[x]
                    timestep = eb.timesteps[x]
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
                            self.dynamic_agent_clearance,
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
                        max_time = max(max_time, timestep)      
                        break
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
        accept_new_state = accept_new_state and not self.collision_check_func(self, list_of_paths, start_time=parent_node.time_elapsed)
        # print(accept_new_state,"FINAL") 
        if not accept_new_state:
            # if self.debug_flag: print("Sampled New CRRT Node is invalid. Trying again!")
            # new node is invalid, exit this round. No new nodes are added to the tree
            # and the caller will attempt to add a whole new node with a new random state, 
            # etc. 
            return -10000 # clearly not a valid node ID!
        else: # This new joint state is valid! 
            # Extend shorter paths to match max
            for path_idx in range(len(list_of_paths)):
                agent_path = list_of_paths[path_idx]
                if len(agent_path) < max_steps:
                    list_of_paths[path_idx] = np.concatenate(
                        (list_of_paths[path_idx], np.full((max_steps - len(list_of_paths[path_idx]), len(list_of_paths[path_idx][0])), 
                                             list_of_paths[path_idx][-1])))

            total_elapsed_time = parent_node.time_elapsed + max_time
            
            updated_paths, new_state, costs, reached_goal = self.check_for_goals(parent_node, 
                                                                      max_time,
                                                                      selected_actions,
                                                                      list_of_paths,
                                                                      new_state)
            
            cum_cost = list( map(add, parent_node.cost_so_far, costs) )
            new_node_id = self.add_rrt_node(new_state, parent_node_id, selected_actions, max_time,
                                                    updated_paths, total_elapsed_time, cum_cost)

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
        
    
    
