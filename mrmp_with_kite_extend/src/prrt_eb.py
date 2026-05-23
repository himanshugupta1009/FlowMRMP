from edge_bundle_rrt import EdgeBundleType2RRT
from Environments import AgentObstacle
from utils import check_dynamic_collisions_to_end
import time
from numba.typed import List
from numba import types
import numpy as np


class EdgeBundlePRRT(EdgeBundleType2RRT):
    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
        Performs the edge bundle planning process:
            * sort edges in the bundle depending on their distance from the random point.
            * Pick the action and time corresponding to the edge with the minimum distance
                from the random point.
            * Use the action and time to extend the tree towards the random point (or don't
                if resulting point would be invalid).
            If that doesn't result in a valid point, pick a number of random edges and 
                attempt to propagate those using the same steps. 

        :MAINT: This is extended from the base RRT class as 
            obstacles for pRRT are time-dependent 
        :TODO: Use helper function for the sorted and random steps

        parent_node_id: Tree node id for the parent node, or the
            state to propagate from 
        parent_node: the RRT node representation of the parent node
        random_point: point to propagate towards
        """

        # get a number of edge bundles, sorted from those that 
        # will get the agent closest to the random_point to the
        # furthest. 
        eb = self.edge_bundle
        p = self.num_skip_edges
        sorted_indices = self.sort_edges(parent_node.state, random_point, eb.final_states, 
                                        self.distance_array)

        # iterate over the list of sorted edges
        for idx,x in enumerate(sorted_indices[::p]):
            # retrieve the action and time duration from the edge bundle 
            action = eb.actions[x]
            timestep = eb.timesteps[x]
            num_record_steps = round(timestep/self.minimum_time_step)

            # get a new state and the path from propagating 
            # the edge bundle 
            # step_env required for pybullet 
            self.env.step_environment([parent_node.state], [action], timestep, parent_node.time_elapsed, 
                                  num_steps=num_record_steps)
            new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, action, 
                                                        timestep, num_steps = num_record_steps)
            
            # check if new node is valid 
            accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                                self.static_circular_obstacles, self.static_rectangular_obstacles,
                                self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                                self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                                self.dynamic_agent_clearance,
                                self.env.boundary_buffer, parent_node.time_elapsed,
                                timestep, self.minimum_time_step)
            
            # check if new node is valid 
            if not accept_new_node:
                # new node is invalid, exit this round. No new nodes are added to the tree
                # and the caller will attempt to add a whole new node with a new random state, 
                # etc. 
                if self.debug_flag:
                    print("Node is invalid : " + str(new_state))
                continue
            else:
                reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
                if reached_goal_flag:

                    if check_dynamic_collisions_to_end(new_state, self.agent.radius, 
                                                    self.dynamic_agent_obstacles, 
                                                    self.dynamic_agent_clearance,
                                                    parent_node.time_elapsed + timestep,
                                                    self.minimum_time_step):
                        if self.debug_flag:
                            print("Goal state will collide with high-priority agent. Trying again!")
                        continue

                    edge_cost = self.cost(self.env, self.agent, parent_node.state, action, 
                                timestep, path_to_new_state)
                    total_elapsed_time = parent_node.time_elapsed + timestep
                    total_cost = parent_node.cost_so_far + edge_cost

                    new_node_id = self.add_rrt_node(new_state, parent_node_id, action, timestep,
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

                                if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius, 
                                                    self.dynamic_agent_obstacles, 
                                                    self.dynamic_agent_clearance,
                                                    total_elapsed_time,
                                                    self.minimum_time_step):
                                    if self.debug_flag:
                                        print("Goal state will collide with high-priority agent. Trying again!")
                                    continue

                                modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                                new_path_to_new_state = path_to_new_state[:index+1]
                                edge_cost = self.cost(self.env, self.agent, parent_node.state, action,
                                                    modified_edge_time, new_path_to_new_state)
                                total_cost = parent_node.cost_so_far + edge_cost

                                new_node_id = self.add_rrt_node(intermediate_state, parent_node_id, action, 
                                                                modified_edge_time, new_path_to_new_state, 
                                                                total_elapsed_time, total_cost)
                                self.path_found = True
                                if self.debug_flag:
                                    print("Goal Reached! Path found for ",self.agent.id)
                                self.goal_node_id = new_node_id
                                self.path_cost = total_cost
                                self.path_time = total_elapsed_time
                                return
                    edge_cost = self.cost(self.env, self.agent, parent_node.state, action, 
                                            timestep, path_to_new_state)
                    total_cost = parent_node.cost_so_far + edge_cost
                    total_elapsed_time = parent_node.time_elapsed + timestep
                    new_node_id = self.add_rrt_node(new_state, parent_node_id, action, timestep,
                                            path_to_new_state, total_elapsed_time, total_cost)
                    if self.debug_flag:
                       print("New Node Added to the RRT Tree: ", new_node_id)
                return 
        
        #Generate a list of random indices 
        random_indices = self.rng.integers(0,eb.num_edges,size=self.num_random_edges)

        for idx,x in enumerate(random_indices):
            # retrieve the action and time duration from the edge bundle 
            action = eb.actions[x]
            timestep = eb.timesteps[x]
            num_record_steps = round(timestep/self.minimum_time_step)

            # get a new state and the path from propagating 
            # the edge bundle 
            self.env.step_environment([parent_node.state], [action], timestep, parent_node.time_elapsed, 
                                  num_steps=num_record_steps)
            new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, action, 
                                                        timestep, num_steps = num_record_steps)
            
            # check if new node is valid 
            accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                                self.static_circular_obstacles, self.static_rectangular_obstacles,
                                self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                                self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                                self.dynamic_agent_clearance,
                                self.env.boundary_buffer, parent_node.time_elapsed,
                                timestep, self.minimum_time_step)
            
            # check if new node is valid 
            if not accept_new_node:
                # new node is invalid, exit this round. No new nodes are added to the tree
                # and the caller will attempt to add a whole new node with a new random state, 
                # etc. 
                if self.debug_flag:
                    print("Node is invalid : " + str(new_state))
                continue
            else:
                reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
                if reached_goal_flag:

                    if check_dynamic_collisions_to_end(new_state, self.agent.radius, 
                                                    self.dynamic_agent_obstacles, 
                                                    self.dynamic_agent_clearance,
                                                    parent_node.time_elapsed + timestep,
                                                    self.minimum_time_step):
                        if self.debug_flag:
                            print("Goal state will collide with high-priority agent. Trying again!")
                        continue

                    edge_cost = self.cost(self.env, self.agent, parent_node.state, action, 
                                timestep, path_to_new_state)
                    total_elapsed_time = parent_node.time_elapsed + timestep
                    total_cost = parent_node.cost_so_far + edge_cost

                    new_node_id = self.add_rrt_node(new_state, parent_node_id, action, timestep,
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
                                if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius, 
                                                    self.dynamic_agent_obstacles, 
                                                    self.dynamic_agent_clearance,
                                                    total_elapsed_time, 
                                                    self.minimum_time_step):
                                    if self.debug_flag:
                                        print("Goal state will collide with high-priority agent. Trying again!")
                                    return
                                
                                modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                                new_path_to_new_state = path_to_new_state[:index+1]
                                edge_cost = self.cost(self.env, self.agent, parent_node.state, action,
                                                    modified_edge_time, new_path_to_new_state)
                                total_cost = parent_node.cost_so_far + edge_cost

                                new_node_id = self.add_rrt_node(intermediate_state, parent_node_id, action, 
                                                                modified_edge_time, new_path_to_new_state, 
                                                                total_elapsed_time, total_cost)
                                self.path_found = True
                                if self.debug_flag:
                                    print("Goal Reached! Path found for ",self.agent.id)
                                self.goal_node_id = new_node_id
                                self.path_cost = total_cost
                                self.path_time = total_elapsed_time
                                return
                    edge_cost = self.cost(self.env, self.agent, parent_node.state, action, 
                                            timestep, path_to_new_state)
                    total_cost = parent_node.cost_so_far + edge_cost
                    total_elapsed_time = parent_node.time_elapsed + timestep
                    new_node_id = self.add_rrt_node(new_state, parent_node_id, action, timestep,
                                            path_to_new_state, total_elapsed_time, total_cost)
                    if self.debug_flag:
                       print("New Node Added to the RRT Tree: ", new_node_id)

                return
        return

    @staticmethod
    def plan_multi(*, agents, starts, goals, goal_radii, env, edge_bundle,
                    use_fixed_sampling_time=True, sampling_time_step=1.0, max_iter=1000, planning_time=10.0,
                    isvalid_function, cost_function, reached_goal_function, translate_function, random_point_function,
                    sort_edges_functions, 
                    num_skip_edges = 10,
                    udf_seed = 77,
                    obs_type = AgentObstacle, 
                    num_rand_edges = 1,
                    dynamic_agent_clearance=0.0,
                    print_logs=False, 
                    debug_flag=False):
        """Perform Priority RRT (pRRT) with Edge Bundles 
        Each agent plans its path individually in a priority order. Subsequent agents 
        must plan using the previous agents' positions along their paths as obstacles. 

        Args:
            agents (list(agent objects)): list of agents
            starts (list(agent_state_type)): agent starts
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
                (env object, agent object, agent_state_type start state, agent_action_type action,
                    float time_delta, list(agent_state_type) path_to_new_state) -> float
            reached_goal_function (bool): List of functions for each agent that returns true if the agent 
                has reached its goal, false else. 
                (agent_state_type state, tuple(x float, y float) goal center, float goal radius, 
                    agent object) -> bool  
            translate_function (agent_state_type): List of functions for each agent that returns
                the end point from an edge bundle translated to start from a current agent state
                (agent_state_tuple, agent_state_tuple) -> agent_state_tuple
            random_point_function (tuple(x float, y float)): List of functions for each agent that 
                generate a new point in the environment 
                (env object, list circular obstacles, list rectangular obstacles, rng object) -> tuple(x float, y float)
            sort_edges_functions: used to pick which edges get an agent closest to a random point
            num_skip_edges (int, optional): number of edges to skip when iterating through edge bundles
            use_fixed_sampling_time (bool, optional): Don't use random timesteps. Defaults to True.
            sampling_time_step (float, optional): Max sampling time to use. Defaults to 1.0.
            max_iter (int, optional): Maximum number of planning iterations before failure. Defaults to 1000.
            planning_time (float, optional): maximum time to plan before failure. Defaults to 10.0.
            udf_seed (int, optional): local rng seed. Defaults to 77.
            obs_type (Obstacle, optional): Type of obstacle to use to represent agents that have found
                goals for other agents to avoid. Defaults to AgentObstacle.
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false

        Returns:
            tuple(list, list, list, list, list): lists of:
                Successful path node ids for each agent
                Successful path state tuples for each agent
                Successful path control inputs for each agent
                Successful path timesteps, i.e. the time delta between each node in the path, for each agent
                    :MAINT: The timesteps are NOT cumulative!
                The pRRT object for each agent. 
        """
        # list of return values for each agent
        paths = []
        states = []
        rrts = []
        controls = []
        timesteps = []

        dyn_obs = List.empty_list(types.Array(types.float64, 2, 'C'))

        seed_rng = np.random.default_rng(udf_seed)
        planner_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, size=len(agents))

        start_time = time.time()
        # iterate over each agent, planning for it
        for i, agent, start, goal, goal_radius in zip(range(len(agents)), agents, starts, goals, goal_radii):
            if(print_logs):
                print("Planning for agent", i, "with id", agent.id)

            env.add_agent(agent, goal=(goal, goal_radius))
            # create a new PRRT object 
            rrt = EdgeBundlePRRT( start=start, goal=goal, goal_radius=goal_radius, 
                env = env, agent=agent, 
                edge_bundle=edge_bundle[i],
                use_fixed_sampling_time=use_fixed_sampling_time,
                sampling_time_step=sampling_time_step,
                max_iter = max_iter, planning_time=planning_time - (time.time() - start_time),         
                isvalid_function=isvalid_function[i], 
                cost_function=cost_function[i],
                random_point_function=random_point_function[i], 
                reached_goal_function = reached_goal_function[i],
                translate_function=translate_function[i],
                sort_edges_function=sort_edges_functions[i],
                num_skip_edges=num_skip_edges,
                udf_seed = int(planner_seeds[i]),
                print_logs=print_logs,
                debug_flag=debug_flag,
                dynamic_obstacles=dyn_obs,
                dynamic_agent_clearance=dynamic_agent_clearance,
                num_random_edges=num_rand_edges
                )
            # Find the path with the PRRT object
            rrt.plan_path()

            # Get path information for each agent 
            path_state_ids, path_states, path_controls, path_timesteps = rrt.get_path()
            paths.append(path_state_ids)
            states.append(path_states)
            controls.append(path_controls)
            timesteps.append(path_timesteps)
            rrts.append(rrt)

            # add new agent obstacle if there was a path found for this 
            # agent 
            if(rrt.path_found and time.time() - start_time <= planning_time):
                dyn_obs.append(obs_type(agent, rrt).to_np())
            else:
                end_time = time.time() - start_time 
                print("PRRT failed to find a path for agent", i)
                print("Failed to solve the current MRMP problem using pRRT.")
                return ([], [], [], [], [], end_time)

        end_time = time.time() - start_time 
        return (paths, states, rrts, controls, timesteps, end_time)
