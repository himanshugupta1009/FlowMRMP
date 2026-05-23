from rrt import *

class EdgeBundleType2RRT(RRT):
    def __init__(self, * , start, goal, goal_radius, env, agent, 
                    edge_bundle,
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
                    goal_sampling_probability=0.1,
                    num_random_edges=10, 
                    num_skip_edges=10,
                    translate_function,
                    sort_edges_function, 
                    debug_flag=False,
                    print_logs=False,
                    dynamic_agent_clearance=0.0,
                    dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C'))
                    ):

        super().__init__(start=start, goal=goal,
                        goal_radius=goal_radius, env=env, agent=agent,
                        use_fixed_sampling_time=use_fixed_sampling_time,
                        sampling_time_step=sampling_time_step,
                        minimum_time_step=minimum_time_step,
                        max_iter=max_iter, planning_time=planning_time,
                        isvalid_function=isvalid_function,
                        cost_function=cost_function,
                        random_point_function=random_point_function,
                        reached_goal_function=reached_goal_function,
                        udf_seed=udf_seed,
                        goal_sampling_probability=goal_sampling_probability,
                        debug_flag=debug_flag,
                        print_logs=print_logs, 
                        dynamic_agent_clearance=dynamic_agent_clearance,
                        dynamic_obstacles=dynamic_obstacles)

        self.edge_bundle = edge_bundle
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.distance_array = np.zeros(self.edge_bundle.num_edges)
        self.random_indices = np.zeros(self.num_random_edges,dtype=int)
        self.translate = translate_function
        self.sort_edges = sort_edges_function

    def get_random_edge_index(self):
        return self.rng.randint(0,self.edge_bundle.num_edges)

    def _try_edge_from_bundle(self, edge_idx, parent_node_id, parent_node, debug_prefix=""):
        
        eb = self.edge_bundle

        action = eb.actions[edge_idx]
        timestep = eb.timesteps[edge_idx]
        num_record_steps = round(timestep / self.minimum_time_step)

        # Propagate dynamics
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state,
                                action, timestep, num_steps=num_record_steps)

        # Collision check
        accept_new_node = self.isvalid(path_to_new_state,self.agent.radius,
                    self.env.size,self.static_circular_obstacles,
                    self.static_rectangular_obstacles,self.dynamic_agent_obstacles,
                    self.env.obstacle_buffer,self.dynamic_agent_clearance,
                    self.env.boundary_buffer,
                    parent_node.time_elapsed,timestep,
                    self.minimum_time_step)

        if not accept_new_node:
            if self.debug_flag:
                print(f"{debug_prefix}~~~~~~~~~~Sampled New Edge Bundle RRT Node is invalid. Trying again!~~~~~~~~~~")
                print("Invalid State :", new_state)
            return False 
        
        # Check goal at the final state
        reached_goal_flag, goal_distance = self.reached_goal(new_state,
                        self.goal, self.goal_radius, self.agent)

        if reached_goal_flag:
            edge_cost = self.cost(self.env, self.agent, parent_node.state, 
                            action, timestep, path_to_new_state)
            total_cost = parent_node.cost_so_far + edge_cost
            total_elapsed_time = parent_node.time_elapsed + timestep

            new_node_id = self.add_rrt_node(new_state,parent_node_id,action,
                    timestep,path_to_new_state,total_elapsed_time,total_cost)

            self.path_found = True
            self.goal_node_id = new_node_id
            self.path_time = total_elapsed_time
            self.path_cost = total_cost

            if self.debug_flag:
                print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

            return True

        # Check if goal is hit along the path to new_state
        if goal_distance < self.threshold:
            total_elapsed_time = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path_to_new_state):
                total_elapsed_time += self.minimum_time_step
                goal_flag, _ = self.reached_goal(intermediate_state, self.goal,
                                    self.goal_radius, self.agent)
                if goal_flag:
                    modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                    new_path_to_new_state = path_to_new_state[:index + 1]

                    edge_cost = self.cost(self.env,self.agent,parent_node.state,
                        action,modified_edge_time,new_path_to_new_state,)
                    total_cost = parent_node.cost_so_far + edge_cost

                    new_node_id = self.add_rrt_node(intermediate_state,
                        parent_node_id,action,modified_edge_time,new_path_to_new_state,
                        total_elapsed_time,total_cost)

                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time

                    if self.debug_flag:
                        print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

                    return True

        # Otherwise: valid node, no goal → just add it
        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                    action, timestep, path_to_new_state)
        total_cost = parent_node.cost_so_far + edge_cost
        total_elapsed_time = parent_node.time_elapsed + timestep

        new_node_id = self.add_rrt_node(new_state,parent_node_id,action,timestep,
            path_to_new_state,total_elapsed_time,total_cost)

        if self.debug_flag:
            print(f"{debug_prefix}New Node Added to the RRT Tree: ", new_node_id)
            print("Valid Node:", new_state)

        return True

    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
        if use_edge_bundle:
            * sort edges in the bundle depending on their distance from the random point.
            * Pick the action and time corresponding to the edge with the minimum distance
            from the random point.
            * Use the action and time to extend the tree towards the random point (or don't).
        """
        eb = self.edge_bundle
        p = self.num_skip_edges

        # sorted_indices = self.sort_edges(parent_node.state, random_point)
        sorted_indices = self.sort_edges(parent_node.state, random_point, 
                        eb.final_states, self.distance_array)

        for idx, x in enumerate(sorted_indices[::p]):
            if self._try_edge_from_bundle(x, parent_node_id, parent_node, debug_prefix="[sorted] "):
                return

        # Generate a list of random indices 
        random_indices = self.rng.integers(0, eb.num_edges, size=self.num_random_edges)

        for idx, x in enumerate(random_indices):
            if self._try_edge_from_bundle(x, parent_node_id, parent_node, debug_prefix="[random] "):
                return

        # print("Sampled New RRT Node is invalid. Trying again!")    
        return


"""
while True:
    sample a new point
    find the nearest node in the tree to the sample point
    extend the tree towards the sample point
    if the new node is in collision, continue
    if the new node is not in collision, add the new node to the tree
    if the maximum number of iterations is reached, break
    if the new node is within the goal radius, break    

"""
