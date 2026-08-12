from math import dist
from rrt import *
from edge_bundle_rrt import *
from kinodynamic_TI_eb_rrt import *
from db.db_rrt import DbRRTPlanner
from numba import njit
from utils import euclidean_distance_numba_with_l

# Function to check collisions when not using a high-level sim
# After the new optimization changes for KCBS, it is not being used anywhere.
# Delete it eventually.
def is_collision_math(first_agent, first_agent_position, second_agent, second_agent_position):
    return (first_agent.check_collision(first_agent_position, second_agent_position) \
                        or second_agent.check_collision(second_agent_position, first_agent_position))

@njit
def next_power_of_two(n):
    """ Returns the next power of two greater than or equal to n """

    l = int(np.round(np.log2(n)) + 1)
    return 2**l

@njit
def constraint_satisfaction_numba(constraints, start_index, end_index, 
                                  path_to_new_state,path_start_time,
                                  curr_agent_radius, delta_t, distance_metric_state_size,
                                  dynamic_agent_clearance=0.1,roundoff_digits=1):

    # Constraints is a list of constraints/conflicts for the current agent.
    # Each constraint is a tuple that has the following format:
    # (collision_keys, collision_agent_position_array, collision_agent_radius) 

    if len(constraints) == 0:
        return True

    path_length = len(path_to_new_state)
    path_end_time = round(path_start_time + path_length*delta_t, roundoff_digits)
    # constraint_debugging = False

    for i in range(start_index, end_index):
        collision_keys, collision_agent_position_array, collision_agent_radius = constraints[i]
        start_conflict_time = collision_keys[0]
        end_conflict_time = collision_keys[-1]

        # Check if a conflict is even possible in the path time range.
        intersect = not(start_conflict_time > path_end_time or end_conflict_time < path_start_time)
        # if constraint_debugging:
        #     print("Checking for conflicts...")
        #     print("Start Conflict Time: ", start_conflict_time, " End Conflict Time: ", end_conflict_time)
        #     print("Path Start Time: ", path_start_time, " Path End Time: ", path_end_time)
        #     print("Intersection: ", intersect)
        if intersect == False:
            # There is no conflict.
            continue

        # States occur at path_start_time + (state_idx + 1) * delta_t.
        # Limit the scan to indices that can overlap the conflict window,
        # padded on both sides so correctness does not depend on float
        # rounding; the exact per-state time check below remains authoritative.
        state_idx_lo = int((start_conflict_time - path_start_time) / delta_t) - 2
        state_idx_hi = int((end_conflict_time - path_start_time) / delta_t) + 1
        if state_idx_lo < 0:
            state_idx_lo = 0
        if state_idx_hi > path_length - 1:
            state_idx_hi = path_length - 1

        for state_idx in range(state_idx_lo, state_idx_hi + 1):
            agent_state = path_to_new_state[state_idx]
            curr_t = round(path_start_time + (state_idx+1)*delta_t, roundoff_digits)
            # if constraint_debugging:
            #     print("Reached A")
            #     print("Current Time: ", curr_t)
            if curr_t < start_conflict_time or curr_t > end_conflict_time:
                # This state is not in the conflict time range.
                continue
            # if constraint_debugging:
            #     print("Reached B")
            # This state is in the conflict time range.
            pos_index = int(round((curr_t - start_conflict_time)/delta_t, roundoff_digits))
            collision_agent_pos = collision_agent_position_array[pos_index]

            threshold = collision_agent_radius + curr_agent_radius + dynamic_agent_clearance
            dist = euclidean_distance_numba_with_l(agent_state, collision_agent_pos, 
                                                   distance_metric_state_size)
            is_collision = dist <= threshold
            # if constraint_debugging:
            #     print("Reached C")
            #     print("Collision Check: ", is_collision, " Distance: ", dist)
            #     print("Current Time: ", curr_t)
            #     print("Agent State: ", agent_state[:distance_metric_state_size])
            #     print("Collision Agent Array Position Index: ", pos_index)
            #     print("Collision Agent State: ", collision_agent_pos[:distance_metric_state_size], "Time: ", curr_t)
            #     print("Numpy Norm Distance: ", np.linalg.norm(agent_state[:distance_metric_state_size] - collision_agent_pos[:distance_metric_state_size]))
            if is_collision:
                # Collision found.
                return False

    return True


@njit
def constraint_satisfaction_parked_numba(
        constraints, parked_state, parked_start_time, curr_agent_radius,
        delta_t, distance_metric_state_size, dynamic_agent_clearance=0.1,
        roundoff_digits=1):
    """Check the stationary path tail assumed after goal arrival by KCBS."""
    if len(constraints) == 0:
        return True

    for i in range(len(constraints)):
        collision_keys, collision_agent_positions, collision_agent_radius = constraints[i]
        if collision_keys[-1] < parked_start_time:
            continue

        threshold = collision_agent_radius + curr_agent_radius + dynamic_agent_clearance
        for key_index in range(len(collision_keys)):
            if collision_keys[key_index] < parked_start_time:
                continue
            distance = euclidean_distance_numba_with_l(
                parked_state,
                collision_agent_positions[key_index],
                distance_metric_state_size,
            )
            if distance <= threshold:
                return False

    return True


class ConstrainedRRT(RRT):
    def __init__(self, * , start, goal, goal_radius, env, agent,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1, 
                    max_iter=1000, 
                    planning_time=10.0,
                    num_extension_trials=1,
                    isvalid_function,
                    cost_function, 
                    reached_goal_function, 
                    random_point_function, 
                    udf_seed = 77,
                    goal_sampling_probability=0.1,
                    use_goal_parking_fix=True,
                    dynamic_agent_clearance=0.0,
                    debug_flag=False,
                    print_logs=False,
                    reuse_tree=False,
                    is_collision_func=is_collision_math
                    ):

        super().__init__(start=start, goal=goal, goal_radius=goal_radius, 
                         env=env, agent=agent,
                         use_fixed_sampling_time=use_fixed_sampling_time, 
                         sampling_time_step=sampling_time_step,
                         minimum_time_step=minimum_time_step, 
                         max_iter=max_iter, 
                         planning_time=planning_time,
                         num_extension_trials=num_extension_trials,
                         isvalid_function=isvalid_function,
                         cost_function=cost_function, 
                         reached_goal_function=reached_goal_function, 
                         random_point_function=random_point_function, 
                         udf_seed=udf_seed,
                         goal_sampling_probability=goal_sampling_probability,
                         use_goal_parking_fix=use_goal_parking_fix,
                         dynamic_agent_clearance=dynamic_agent_clearance,
                         debug_flag=debug_flag,
                         print_logs=print_logs)

        self.constraints = List()
        dummy_constraint = (np.empty(0), np.empty((0,0)), 0.0)  # Placeholder for constraints
        self.constraints.append(dummy_constraint)
        self.constraints.pop()

        self.is_collision_func = is_collision_func
        self.dynamic_agent_clearance = dynamic_agent_clearance
        self.reuse_tree = bool(reuse_tree)


    def set_constraints(self, cons):
        self.constraints = cons


    def copy_retained_node_bookkeeping(self, old_node, new_node):
        """Copy planner-specific search state when retaining a tree node."""
        # Design note: node reconstruction and this hook could be combined
        # into a polymorphic add_retained_rrt_node(old_node, new_parent_id)
        # method, with KiTE overriding it to copy candidate bookkeeping. That
        # would encapsulate the two operations but would not reduce Python
        # calls unless add_rrt_node() were duplicated/inlined, so the simpler
        # separated design is retained for now.
        return


    def constraint_satisfaction(self,path_to_new_state,path_start_time):
    
        return constraint_satisfaction_numba(self.constraints, 0, len(self.constraints),
                            path_to_new_state,path_start_time, self.agent.radius, 
                            self.minimum_time_step,self.distance_metric_state_size,
                            self.dynamic_agent_clearance, self.roundoff_digits)

    def parking_blocked(self, state, arrival_time):
        if RRT.parking_blocked(self, state, arrival_time):
            return True
        if not self.use_goal_parking_fix:
            return False
        return not constraint_satisfaction_parked_numba(
            self.constraints,
            state,
            arrival_time,
            self.agent.radius,
            self.minimum_time_step,
            self.distance_metric_state_size,
            self.dynamic_agent_clearance,
            self.roundoff_digits,
        )


    def _select_best_extension_candidate(self, parent_node, random_point):
        """
        Try self.num_extension_trials rollouts from parent_node and 
        return the best valid one.

        Returns
        -------
        None
            If all trials are invalid.
        (new_state, path_to_new_state, random_action, random_time)
            Best valid candidate according to a simple score.
        """

        best_candidate = None
        best_score = np.inf

        for _ in range(self.num_extension_trials):
            random_action = self.agent.get_random_action(self.rng)
            random_time = self.get_time()
            num_record_steps = round(random_time / self.minimum_time_step)

            new_state, path_to_new_state = self.agent.get_next_state(
                parent_node.state,
                random_action,
                random_time,
                num_steps=num_record_steps
            )

            new_state_valid = self.isvalid(
                path_to_new_state,
                self.agent.radius,
                self.env.size,
                self.static_circular_obstacles,
                self.static_rectangular_obstacles,
                self.dynamic_agent_obstacles,
                self.agent.dynamic_limit_indices, 
                self.agent.dynamic_limit_values,
                self.env.obstacle_buffer,
                self.dynamic_agent_clearance,
                self.env.boundary_buffer,
                parent_node.time_elapsed,
                random_time,
                self.minimum_time_step
            )
            constraint_satisfied = self.constraint_satisfaction(path_to_new_state,
                                                        parent_node.time_elapsed)

            if not (new_state_valid and constraint_satisfied):
                continue

            # Score: distance to the sampled point (classic RRT heuristic)
            score = euclidean_distance_numba_with_l(new_state,random_point,
                                        self.distance_metric_state_size)

            if score < best_score:
                best_score = score
                best_candidate = (
                    new_state,
                    path_to_new_state,
                    random_action,
                    random_time
                )

        return best_candidate


    def get_conflict_edge(self, tree, goal_node_id, time_value):
        """
        Get the edge in the path that overlaps with the input time value.
        """
        # Get the edge in the tree that has the conflict/constraint.
        # The edge is a tuple (x,y) where x is the start node and y is the end node.
        # The edge is returned as a tuple (x,y)

        if goal_node_id is None:
            print("[ERROR get_conflict_edge] The retained RRT tree snapshot "
                  "has no solution goal node id. KCBS tree bookkeeping needs "
                  "to be checked.")
            return (-100, -100)

        if goal_node_id not in tree:
            print("[ERROR get_conflict_edge] Solution goal node",
                  goal_node_id, "is missing from its retained RRT tree "
                  "snapshot. KCBS tree bookkeeping needs to be checked.")
            return (-100, -100)

        # Edge is parent_id -> child_id
        child_id = goal_node_id
        parent_id = tree.nodes[child_id]['value'].parent_id
        if parent_id == -1 or parent_id not in tree:
            return (-100, -100)

        rrt_parent_node = tree.nodes[parent_id]['value']
        if rrt_parent_node.time_elapsed < time_value:
            #It means the conflict happens from the goal node's parent to the goal node.
            return (parent_id, child_id)

        while parent_id != -1:
            rrt_parent_node = tree.nodes[parent_id]['value']
            rrt_child_node = tree.nodes[child_id]['value']
            if( rrt_parent_node.time_elapsed < time_value <= rrt_child_node.time_elapsed):
                #Time interval for checking is this -> (start_time, end_time]
                return (parent_id, child_id)
            child_id = parent_id
            parent_id = rrt_parent_node.parent_id
        
        return (-100, -100) #Code should not reach here. 


    def plan_path_with_constraints(self, curr_tree_structure, constraints):
        """
        This function is called when the agent had already found a path 
        with the set of old constraints, but now has a new set of constraints.
        It should be used to replan the path for the agent.

        Note: New set of constraints = Old set of constraints U {New conflict}.
        New Conflict = (collision_keys, collision_agent_position_array, collision_agent_radius)
        New Conflict is appended to the end of the constraints list.

        This method validates retained edges using self.constraints. KCBS must
        therefore call set_constraints(constraints) immediately before calling
        this method. The constraints argument identifies the newly appended
        conflict, while self.constraints is the active set used by
        constraint_satisfaction_numba(). Calling this method directly without
        first installing the same constraint list can validate against stale
        constraints.
        """

        if not self.reuse_tree:
            self.reset_tree()
            return self.plan_path()
        else:

            """
            We need to create a new tree from the old tree by removing the parts of the tree
            that are affected by the new constraints. Then we need to rewire and grow the tree
            from the point where the new constraints start affecting the tree.
            The steps are as follows:
            0. Find the start time of the new constraint/conflict.
            1. Find the edge in the tree that has the conflict/constraint. Edge->(x,y)
            2. Find all the nodes in the tree that are descendants of y.
            3. Remove all the descendants of y from the tree.
            4. Delete the edge (x,y) from the tree.
            5. Start rewiring and growing the new tree from x.
            """

            #This means a path was already found which avoids the old constraints.

            #Find the start time of the new constraint/conflict.
            new_conflict = constraints[-1]
            conflict_start_time = new_conflict[0][0]
            num_constraints = len(constraints)
            curr_tree = curr_tree_structure[0]

            # Find the edge on the solution path that overlaps the conflict.
            # The goal id travels with its snapshot because the last inserted
            # tree node is not necessarily the goal for that solution.
            goal_node_id = curr_tree_structure[2]

            edge_start_node_id, edge_end_node_id = self.get_conflict_edge(curr_tree, goal_node_id, conflict_start_time)
            
            if edge_start_node_id == -100 and edge_end_node_id == -100:
                print("[ERROR plan_path_with_constraints] Could not locate "
                      "the new conflict on the retained solution path "
                      "(conflict start time:", conflict_start_time, "). "
                      "KCBS tree bookkeeping and conflict timing need to be "
                      "checked. Falling back to a scratch replan.")
                self.reset_tree()
                return self.plan_path()
            else:
                # descendants = self.find_descendants(edge_end_node_id)
                self.tree = nx.DiGraph()
                num_curr_tree_nodes = len(curr_tree.nodes)
                capacity = next_power_of_two(num_curr_tree_nodes)
                self._node_matrix = DynamicMatrix(initial_capacity=capacity, 
                                                dim=self.distance_metric_state_size)
                self.last_added_node_id = -1
                self._node_matrix.count = 0
                
                new_id_dict = {} #Old node id -> New node id
                # get_conflict_edge() returns x -> y, where the new
                # constraint first affects the stored solution path. The old
                # subtree below y cannot be reused. However, x -> y itself
                # still needs validation: when the collision occurs after
                # arrival while the agent is implicitly parked at y, the
                # incoming edge can remain valid and y can be retained as an
                # ordinary transit node.
                descendants = {edge_end_node_id}
                copy_retained_bookkeeping = self.copy_retained_node_bookkeeping

                #Add the root node to the new tree.
                root_node = curr_tree.nodes[0]['value']
                new_root_id = self.add_rrt_node(root_node.state, -1, root_node.parent_action,
                                root_node.parent_action_duration, root_node.path_from_parent,
                                root_node.time_elapsed, root_node.cost_so_far)
                new_root_node = self.tree.nodes[new_root_id]['value']
                copy_retained_bookkeeping(root_node, new_root_node)
                new_id_dict[0] = new_root_id

                # for node_id in range(1, num_curr_tree_nodes): #Start from 1 to avoid the root node.
                for node_id in curr_tree.nodes: 
                    if(node_id == 0): #Avoid the root node.
                        continue
                    curr_node = curr_tree.nodes[node_id]['value']
                    parent_id = curr_node.parent_id
                    # A node whose parent is in the discarded subtree cannot
                    # be retained. Do not also skip node_id merely because it
                    # is y: the constraint check below distinguishes a
                    # collision on x -> y from one on y's parked tail.
                    if parent_id in descendants:
                        descendants.add(node_id)
                    else:
                        # NetworkX preserves node insertion order, and RRT
                        # children are always inserted after their parents.
                        # Therefore every retained parent has already been
                        # remapped before its children are processed.
                        new_parent_id = new_id_dict[parent_id]
                        path_start_time = curr_tree.nodes[parent_id]['value'].time_elapsed
                        satisfy_new_constraint = constraint_satisfaction_numba(self.constraints, 
                                            num_constraints-1, num_constraints,
                                            curr_node.path_from_parent, path_start_time, 
                                            self.agent.radius, self.minimum_time_step,
                                            self.distance_metric_state_size,
                                            self.dynamic_agent_clearance, self.roundoff_digits)
                        
                        if satisfy_new_constraint:
                            new_node_id = self.add_rrt_node(curr_node.state, new_parent_id, curr_node.parent_action,
                                            curr_node.parent_action_duration, curr_node.path_from_parent,
                                            curr_node.time_elapsed, curr_node.cost_so_far)
                            new_node = self.tree.nodes[new_node_id]['value']
                            copy_retained_bookkeeping(curr_node, new_node)
                            new_id_dict[node_id] = new_node_id
                        else:
                            #Do not add this node or any of its descendants to the new tree.
                            descendants.add(node_id)
                            continue

                if self.debug_flag:
                    print("Number of nodes in the new tree: ", len(self.tree.nodes))
                    print("Number of nodes in the old tree: ", len(curr_tree.nodes))
                    print("Number of descendants removed: ", len(descendants))

            # Start growing toward the goal from immediately before the
            # conflict when that parent survived full-tree revalidation.
            if edge_start_node_id in new_id_dict:
                rewiring_node_id = new_id_dict[edge_start_node_id]
                rewiring_node = self.tree.nodes[rewiring_node_id]['value']
                self.extend_tree(rewiring_node_id, rewiring_node, self.goal)
            
            #Plan for a new path with this modified tree.
            return self.replan_path()


class ConstrainedRRTPB(ConstrainedRRT):        
    def plan_path(self):
        path = super().plan_path()
        self.agent.set_agent_oob()
        self.env.step_pb()
        return path
    
    def replan_path(self):
        path = super().replan_path()
        self.agent.set_agent_oob()
        self.env.step_pb()
        return path
        

class ConstrainedEdgeBundleType2RRT(ConstrainedRRT,EdgeBundleType2RRT):
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
                    use_goal_parking_fix=True,
                    reuse_tree=False,
                    dynamic_agent_clearance=0.0,
                    num_random_edges=10, 
                    num_skip_edges=10, 
                    translate_function,
                    sort_edges_function,
                    debug_flag = False,
                    print_logs = False,
                    is_collision_func = is_collision_math
                    ):

        EdgeBundleType2RRT.__init__(self, start=start, goal=goal, goal_radius=goal_radius,
                                    env=env, agent=agent,
                                    edge_bundle=edge_bundle,
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
                                    goal_sampling_probability=goal_sampling_probability,
                                    num_random_edges=num_random_edges, 
                                    num_skip_edges=num_skip_edges, 
                                    translate_function=translate_function,
                                    sort_edges_function=sort_edges_function,
                                    dynamic_agent_clearance=dynamic_agent_clearance,
                                    debug_flag=debug_flag,
                                    print_logs=print_logs,
                                    )

        # True checks whether a goal arrival can remain there safely and keeps
        # an unsafe arrival as a transit node. False accepts a valid goal
        # arrival without checking its stationary tail.
        self.use_goal_parking_fix = bool(use_goal_parking_fix)
        self.reuse_tree = bool(reuse_tree)
        self.constraints = List()
        dummy_constraint = (np.empty(0), np.empty((0,0)), 0.0)  # Placeholder for constraints
        self.constraints.append(dummy_constraint)
        self.constraints.pop()

        self.is_collision_func = is_collision_func
        self.dynamic_agent_clearance = dynamic_agent_clearance

    def _try_edge_from_bundle(self, edge_idx, parent_node_id, parent_node, debug_prefix=""):
        
        eb = self.edge_bundle

        action = eb.actions[edge_idx]
        timestep = eb.timesteps[edge_idx]
        num_record_steps = round(timestep / self.minimum_time_step)

        # Propagate dynamics
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state,
                                action, timestep, num_steps=num_record_steps)

        # Collision check
        new_state_valid = self.isvalid(path_to_new_state,self.agent.radius,
                        self.env.size,self.static_circular_obstacles,
                        self.static_rectangular_obstacles,self.dynamic_agent_obstacles,
                        self.agent.dynamic_limit_indices,self.agent.dynamic_limit_values,
                        self.env.obstacle_buffer,self.dynamic_agent_clearance,
                        self.env.boundary_buffer,
                        parent_node.time_elapsed,timestep,
                        self.minimum_time_step)
        
        constraint_satisfied = self.constraint_satisfaction(path_to_new_state,
                                                        parent_node.time_elapsed)
        accept_new_node = new_state_valid and constraint_satisfied

        if not accept_new_node:
            if self.debug_flag:
                print(f"{debug_prefix}~~~~~~~~~~Sampled New Edge Bundle RRT Node is invalid. Trying again!~~~~~~~~~~")
                print("Invalid State :", new_state)
                print("Path to Invalid State: ", path_to_new_state)
            return False

        # Check goal at the final state
        reached_goal_flag, goal_distance = self.reached_goal(new_state,
                        self.goal, self.goal_radius, self.agent)

        if reached_goal_flag:
            total_elapsed_time = parent_node.time_elapsed + timestep
            if (self.use_goal_parking_fix
                    and self.parking_blocked(new_state, total_elapsed_time)):
                if self.debug_flag:
                    print(f"{debug_prefix}Goal state cannot be parked safely yet. Adding it as a transit node.")
            else:
                edge_cost = self.cost(self.env, self.agent, parent_node.state,
                                action, timestep, path_to_new_state)
                total_cost = parent_node.cost_so_far + edge_cost

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
        if not reached_goal_flag and goal_distance < self.threshold:
            total_elapsed_time = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path_to_new_state):
                total_elapsed_time += self.minimum_time_step
                goal_flag, _ = self.reached_goal(intermediate_state, self.goal,
                                    self.goal_radius, self.agent)
                if goal_flag:
                    if (self.use_goal_parking_fix
                            and self.parking_blocked(intermediate_state, total_elapsed_time)):
                        if self.debug_flag:
                            print(f"{debug_prefix}Intermediate goal state violates a later constraint window. Trying again!")
                        continue
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

    def extend_tree(self, *args, **kwargs):
        return EdgeBundleType2RRT.extend_tree(self, *args, **kwargs)

class ConstrainedKinoTIEBRRT(ConstrainedRRT,KinoTIEBRRT):
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
                    translate_function,
                    sort_edges_function,
                    is_collision_func = is_collision_math,
                    max_num_edges_per_node=1000,
                    num_skip_edges=50,
                    rank_candidates=True,
                    use_geometric_candidate_schedule=True,
                    num_random_edges=1,
                    epsilon_random=0.01,
                    eb_kd_tree,
                    get_eb_kd_tree_query,
                    kd_tree_delta_radius=0.5,
                    goal_sampling_probability=0.1,
                    use_goal_parking_fix=True,
                    reuse_tree=False,
                    dynamic_agent_clearance=0.0,
                    udf_seed,
                    debug_flag=False,
                    print_logs=False,
                    ):

        KinoTIEBRRT.__init__(self, start=start, goal=goal, 
                    goal_radius=goal_radius,
                    env=env, agent=agent,
                    edge_bundle=edge_bundle,
                    use_fixed_sampling_time=use_fixed_sampling_time, 
                    sampling_time_step=sampling_time_step,
                    minimum_time_step=minimum_time_step, 
                    max_iter=max_iter, 
                    planning_time=planning_time,
                    isvalid_function=isvalid_function,
                    cost_function=cost_function, 
                    reached_goal_function=reached_goal_function, 
                    random_point_function=random_point_function,
                    translate_function=translate_function,
                    sort_edges_function=sort_edges_function,
                    max_num_edges_per_node=max_num_edges_per_node,
                    num_skip_edges=num_skip_edges,
                    rank_candidates=rank_candidates,
                    use_geometric_candidate_schedule=(
                        use_geometric_candidate_schedule),
                    num_random_edges=num_random_edges,
                    epsilon_random=epsilon_random,
                    eb_kd_tree=eb_kd_tree,
                    get_eb_kd_tree_query=get_eb_kd_tree_query,
                    kd_tree_delta_radius=kd_tree_delta_radius, 
                    udf_seed=udf_seed,
                    goal_sampling_probability=goal_sampling_probability,
                    dynamic_agent_clearance=dynamic_agent_clearance,
                    debug_flag=debug_flag,
                    print_logs=print_logs,
                    )

        # True checks whether a goal arrival can remain there safely and keeps
        # an unsafe arrival as a transit node. False accepts a valid goal
        # arrival without checking its stationary tail.
        self.use_goal_parking_fix = bool(use_goal_parking_fix)
        self.reuse_tree = bool(reuse_tree)
        self.constraints = List()
        dummy_constraint = (np.empty(0), np.empty((0,0)), 0.0)  # Placeholder for constraints
        self.constraints.append(dummy_constraint)
        self.constraints.pop()

        self.is_collision_func = is_collision_func
        self.dynamic_agent_clearance = dynamic_agent_clearance

    def copy_retained_node_bookkeeping(self, old_node, new_node):
        # Candidate ids are selected once for this state and are read-only, so
        # retained CBS branches can share that array. The mask is updated in
        # place as primitives are tried, so each branch needs its own copy.
        new_node.edge_bundle_indices = old_node.edge_bundle_indices
        if old_node.edge_bundle_mask is not None:
            new_node.edge_bundle_mask = old_node.edge_bundle_mask.copy()

    def _try_edge_from_bundle(self,edge_bundle_index,parent_node,
                parent_node_id,mask_index,curr_edge_mask,debug_prefix=""):

        eb = self.edge_bundle

        action = eb.actions[edge_bundle_index]
        timestep = eb.timesteps[edge_bundle_index]
        num_record_steps = round(timestep / self.minimum_time_step)

        # Propagate dynamics
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state,
                                action, timestep, num_steps=num_record_steps)

        # Collision check
        new_state_valid = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices, 
                        self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                        self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        timestep, self.minimum_time_step)
        
        constraint_satisfied = self.constraint_satisfaction(path_to_new_state,
                                                        parent_node.time_elapsed)
        accept_new_node = new_state_valid and constraint_satisfied

        if not accept_new_node:
            # Mark this edge as tried
            curr_edge_mask[mask_index] = True
            if self.debug_flag:
                print(f"{debug_prefix}~~~~~~~~~~Sampled New State in Kino TI EB RRT is invalid. Trying again!~~~~~~~~~~")
                print("Invalid State :", new_state)
            return False  # nothing added; caller should continue

        # Check goal at the final state
        reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                            self.goal_radius, self.agent)

        if reached_goal_flag:
            total_elapsed_time = parent_node.time_elapsed + timestep
            if (self.use_goal_parking_fix
                    and self.parking_blocked(new_state, total_elapsed_time)):
                if self.debug_flag:
                    print(f"{debug_prefix}Goal state cannot be parked safely yet. Adding it as a transit node.")
            else:
                edge_cost = self.cost(self.env, self.agent, parent_node.state,
                                action, timestep, path_to_new_state)
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state,parent_node_id,
                            action,timestep,path_to_new_state,
                            total_elapsed_time,total_cost)

                self.path_found = True
                self.goal_node_id = new_node_id
                self.path_time = total_elapsed_time
                self.path_cost = total_cost
                curr_edge_mask[mask_index] = True

                if self.debug_flag:
                    print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

                return True  # node added

        # Check if we hit the goal along the path
        if not reached_goal_flag and goal_distance < self.threshold:
            total_elapsed_time = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path_to_new_state):
                total_elapsed_time += self.minimum_time_step
                goal_flag, d = self.reached_goal(intermediate_state, self.goal,
                            self.goal_radius, self.agent)
                if goal_flag:
                    if (self.use_goal_parking_fix
                            and self.parking_blocked(intermediate_state, total_elapsed_time)):
                        if self.debug_flag:
                            print(f"{debug_prefix}Intermediate goal state violates a later constraint window. Trying again!")
                        continue
                    modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                    new_path_to_new_state = path_to_new_state[:index + 1]

                    edge_cost = self.cost(self.env,self.agent,parent_node.state,
                                action,modified_edge_time,new_path_to_new_state)
                    total_cost = parent_node.cost_so_far + edge_cost

                    new_node_id = self.add_rrt_node(intermediate_state,
                        parent_node_id,action,modified_edge_time,
                        new_path_to_new_state,total_elapsed_time,total_cost)

                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time
                    curr_edge_mask[mask_index] = True

                    if self.debug_flag:
                        print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

                    return True  # node added

        # Otherwise: valid node, no goal → add full edge
        edge_cost = self.cost(self.env, self.agent, parent_node.state, 
                        action, timestep, path_to_new_state)
        total_cost = parent_node.cost_so_far + edge_cost
        total_elapsed_time = parent_node.time_elapsed + timestep

        new_node_id = self.add_rrt_node(new_state,parent_node_id,
                    action,timestep,path_to_new_state,
                    total_elapsed_time,total_cost)

        curr_edge_mask[mask_index] = True

        if self.debug_flag:
            print(f"{debug_prefix}New Node Added to the RRT Tree: ", new_node_id)
            print("Valid Node:", new_state)

        return True  # node added

    def _try_random_control(self, parent_node, parent_node_id,
        random_point, debug_prefix="[random-control] "):

        action = self.agent.get_random_action(self.rng)
        timestep = self.get_time()
        num_record_steps = round(timestep / self.minimum_time_step)

        # Propagate dynamics
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state,
                                action, timestep, num_steps=num_record_steps)

        # Collision and constraint checks
        new_state_valid = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                        self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                        self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        timestep, self.minimum_time_step)

        constraint_satisfied = self.constraint_satisfaction(path_to_new_state,
                                                        parent_node.time_elapsed)
        accept_new_node = new_state_valid and constraint_satisfied

        if not accept_new_node:
            if self.debug_flag:
                print(f"{debug_prefix}~~~~~~~~~~Sampled random-control state in Kino TI EB RRT is invalid. Trying again!~~~~~~~~~~")
                print("Invalid State :", new_state)
            return False

        # Check goal at the final state
        reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal,
                            self.goal_radius, self.agent)

        if reached_goal_flag:
            total_elapsed_time = parent_node.time_elapsed + timestep
            if (self.use_goal_parking_fix
                    and self.parking_blocked(new_state, total_elapsed_time)):
                if self.debug_flag:
                    print(f"{debug_prefix}Goal state cannot be parked safely yet. Adding it as a transit node.")
            else:
                edge_cost = self.cost(self.env, self.agent, parent_node.state,
                                action, timestep, path_to_new_state)
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state,parent_node_id,
                            action,timestep,path_to_new_state,
                            total_elapsed_time,total_cost)

                self.path_found = True
                self.goal_node_id = new_node_id
                self.path_time = total_elapsed_time
                self.path_cost = total_cost

                if self.debug_flag:
                    print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

                return True

        # Check if we hit the goal along the path
        if not reached_goal_flag and goal_distance < self.threshold:
            total_elapsed_time = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path_to_new_state):
                total_elapsed_time += self.minimum_time_step
                goal_flag, d = self.reached_goal(intermediate_state, self.goal,
                            self.goal_radius, self.agent)
                if goal_flag:
                    if (self.use_goal_parking_fix
                            and self.parking_blocked(intermediate_state, total_elapsed_time)):
                        if self.debug_flag:
                            print(f"{debug_prefix}Intermediate goal state violates a later constraint window. Trying again!")
                        continue
                    modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                    new_path_to_new_state = path_to_new_state[:index + 1]

                    edge_cost = self.cost(self.env,self.agent,parent_node.state,
                                action,modified_edge_time,new_path_to_new_state)
                    total_cost = parent_node.cost_so_far + edge_cost

                    new_node_id = self.add_rrt_node(intermediate_state,
                        parent_node_id,action,modified_edge_time,
                        new_path_to_new_state,total_elapsed_time,total_cost)

                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time

                    if self.debug_flag:
                        print(f"{debug_prefix}Goal Reached! Path found for ", self.agent.id)

                    return True

        # Otherwise: valid node, no goal -> add full random-control edge
        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                        action, timestep, path_to_new_state)
        total_cost = parent_node.cost_so_far + edge_cost
        total_elapsed_time = parent_node.time_elapsed + timestep

        new_node_id = self.add_rrt_node(new_state,parent_node_id,
                    action,timestep,path_to_new_state,
                    total_elapsed_time,total_cost)

        if self.debug_flag:
            print(f"{debug_prefix}New random-control node added to the RRT Tree: ", new_node_id)
            print("Valid Node:", new_state)

        return True

    def extend_tree(self, *args, **kwargs):
        return KinoTIEBRRT.extend_tree(self, *args, **kwargs)

class ConstrainedDbRRTPlanner(DbRRTPlanner):
    """
    Constrained Db-RRT planner for KCBS / CBS-style low-level search.

    This mirrors the constrained single-agent planners in this module: every
    candidate high-resolution path segment must satisfy both the base validity
    checks and the currently active time-indexed constraints.
    """

    def __init__(self, *args, dynamic_agent_clearance: float = 0.0,
                 use_goal_parking_fix=True, **kwargs):
        super().__init__(*args, **kwargs)
        # True also requires the stationary goal tail to satisfy future
        # constraints. False retains only the legacy superclass safety check.
        self.use_goal_parking_fix = bool(use_goal_parking_fix)
        self.constraints = List()
        dummy_constraint = (np.empty(0), np.empty((0, 0)), 0.0)
        self.constraints.append(dummy_constraint)
        self.constraints.pop()
        self.dynamic_agent_clearance = float(dynamic_agent_clearance)
        self.optimizer_function = None
        self.optimized_result = None
        self.optimized_path_view = None
        self.trajectory_source = None
        self.optimizer_output_feasible = False
        self.raw_path_found = False
        self.raw_goal_node_id = None
        self.raw_path_cost = float("inf")
        self.raw_path_time = 0.0
        # Keep both success and failure flags for backward compatibility and
        # to distinguish three cases:
        #   attempted=False, succeeded=False, failed=False -> no optimizer ran
        #   attempted=True,  succeeded=True,  failed=False -> optimizer accepted
        #   attempted=True,  succeeded=False, failed=True  -> optimizer rejected
        self.optimization_attempted = False
        self.optimization_succeeded = False
        self.optimization_failed = False

    def set_optimizer(self, optimizer_function):
        self.optimizer_function = optimizer_function

    def _clear_optimized_result(self):
        self.optimized_result = None
        self.optimized_path_view = None
        self.trajectory_source = None
        self.optimizer_output_feasible = False
        self.optimization_attempted = False
        self.optimization_succeeded = False
        self.optimization_failed = False

    def _clear_raw_result(self):
        self.raw_path_found = False
        self.raw_goal_node_id = None
        self.raw_path_cost = float("inf")
        self.raw_path_time = 0.0

    def _record_raw_result(self):
        self.raw_path_found = bool(self.path_found)
        self.raw_goal_node_id = self.goal_node_id
        self.raw_path_cost = float(self.path_cost)
        self.raw_path_time = float(self.path_time)

    def _maybe_optimize(self):
        self._clear_optimized_result()
        if not self.path_found or self.optimizer_function is None:
            return
        self.optimization_attempted = True
        opt_result = self.optimizer_function(self)
        self.trajectory_source = getattr(opt_result, "source", None)
        self.optimizer_output_feasible = bool(
            getattr(opt_result, "optimizer_output_feasible", getattr(opt_result, "feasible", False))
        )
        if getattr(opt_result, "feasible", False):
            self.optimization_succeeded = True
            self.optimized_result = opt_result
            self.optimized_path_view = getattr(opt_result, "path_view", None)
            if self.optimized_path_view is not None:
                self.path_time = round(self.optimized_path_view.path_time, self.roundoff_digits)
                self.path_cost = self.path_time
            else:
                self.path_cost = float(self.raw_path_cost)
        else:
            self.optimization_failed = True
            self.path_found = False
            self.goal_node_id = None
            self.path_cost = float("inf")
            self.path_time = 0.0

        if self.print_logs or self.debug_flag:
            agent_msg = ""
            if hasattr(self.agent, "id"):
                agent_msg = f" for agent {self.agent.id}"
            opt_src = self.trajectory_source
            if opt_src is not None and str(opt_src).startswith("cpp_dynoplan"):
                opt_src = "dynoplan"
            if self.debug_flag:
                print(
                    f"Optimizer result{agent_msg}: "
                    f"attempted={self.optimization_attempted}, "
                    f"succeeded={self.optimization_succeeded}, "
                    f"opt_src={opt_src}, "
                    f"optimizer_output_feasible={self.optimizer_output_feasible}, "
                    f"final_path_found={self.path_found}, "
                    f"raw_cost={self.raw_path_cost}, final_cost={self.path_cost}"
                )
            else:
                print(
                    f"Optimizer result{agent_msg}: "
                    f"succeeded={self.optimization_succeeded}, "
                    f"opt_src={opt_src}, "
                    f"cost={self.raw_path_cost}->{self.path_cost}"
                )

    def set_constraints(self, cons):
        if isinstance(cons, list) and len(cons) == 0:
            self.constraints = List()
            dummy_constraint = (np.empty(0), np.empty((0, 0)), 0.0)
            self.constraints.append(dummy_constraint)
            self.constraints.pop()
            return
        self.constraints = cons

    def constraint_satisfaction(self, path_to_new_state, path_start_time):
        return constraint_satisfaction_numba(
            self.constraints,
            0,
            len(self.constraints),
            path_to_new_state,
            path_start_time,
            self.agent.radius,
            self.minimum_time_step,
            self.distance_metric_state_size,
            self.dynamic_agent_clearance,
            self.roundoff_digits,
        )

    def _is_path_valid(self, path, start_time, duration):
        if not super()._is_path_valid(path, start_time, duration):
            return False
        return self.constraint_satisfaction(path, start_time)

    def _goal_state_is_safe(self, state, arrival_time):
        if not super()._goal_state_is_safe(state, arrival_time):
            return False
        if not self.use_goal_parking_fix:
            return True
        return constraint_satisfaction_parked_numba(
            self.constraints,
            state,
            arrival_time,
            self.agent.radius,
            self.minimum_time_step,
            self.distance_metric_state_size,
            self.dynamic_agent_clearance,
            self.roundoff_digits,
        )

    def _search_current_tree(self):
        self.path_found = False
        self.goal_node_id = None
        self.path_cost = float("inf")
        self.path_time = 0.0
        self.last_added_node_id = next(reversed(self.tree._node)) if len(self.tree.nodes) > 0 else -1

        curr_num_steps = 0
        start_time = time.time()

        while curr_num_steps < self.max_iter:
            if time.time() - start_time >= self.planning_time:
                break

            random_point, is_goal_sample = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            new_node_id = self._expand_once(
                nearest_node_id,
                nearest_node,
                random_point,
                is_goal_sample,
            )

            if new_node_id is not None:
                if self.path_found:
                    break

            curr_num_steps += 1

        total_wall = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations"
            print(planning_time_msg + ": ", total_wall)

    def plan_path(self):
        self._clear_raw_result()
        self._clear_optimized_result()
        result = super().plan_path()
        self._record_raw_result()
        self._maybe_optimize()
        return result

    def plan_path_with_constraints(self, curr_tree_structure, constraints):
        self.set_constraints(constraints)
        self.reset_tree()
        return self.plan_path()

    def replan_path(self):
        self._clear_optimized_result()
        if self.raw_path_found:
            return self.plan_path()

        if len(self.tree.nodes) == 0:
            return self.plan_path()

        self._clear_raw_result()
        self._search_current_tree()
        self._record_raw_result()
        self._maybe_optimize()
        return None

    def get_high_resolution_path_numpy_array(self):
        if self.optimized_path_view is not None:
            return self.optimized_path_view.get_high_resolution_path_numpy_array()
        return super().get_high_resolution_path_numpy_array()

    def get_high_resolution_path(self):
        if self.optimized_path_view is not None:
            return self.optimized_path_view.get_high_resolution_path()
        return super().get_high_resolution_path()

    def get_path(self):
        if self.optimized_path_view is not None:
            return self.optimized_path_view.get_path()
        return super().get_path()

    def get_high_resolution_path_and_actions(self):
        if self.optimized_path_view is not None:
            return self.optimized_path_view.get_high_resolution_path_and_actions()
        return super().get_high_resolution_path_and_actions()
