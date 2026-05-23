from rrt import *
from utils import check_dynamic_collisions_to_end, check_dynamic_collisions_to_end_3d


class KinoTIEBTreeNode:
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
        self.edge_bundle_indices = None
        self.edge_bundle_mask = None



class KinoTIEBRRT(RRT):
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
                    max_num_edges_per_node=1000,
                    num_skip_edges=10,
                    num_random_edges=1, #Corresponds to num_extension_trials in the main RRT loop
                    epsilon_random=0.01,
                    eb_kd_tree,
                    get_eb_kd_tree_query,
                    kd_tree_delta_radius=0.5,
                    udf_seed,
                    goal_sampling_probability=0.1,
                    dynamic_agent_clearance=0.0,
                    debug_flag=False,
                    print_logs=False,
                    dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C'))
                    ):

        super().__init__(start=start, goal=goal,
                        goal_radius=goal_radius, env=env, agent=agent,
                        use_fixed_sampling_time=use_fixed_sampling_time,
                        sampling_time_step=sampling_time_step,
                        minimum_time_step=minimum_time_step,
                        max_iter=max_iter, planning_time=planning_time,
                        num_extension_trials=num_random_edges,
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
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        if epsilon_random < 0.0 or epsilon_random > 1.0:
            raise ValueError("epsilon_random must be between 0.0 and 1.0")
        self.epsilon_random = epsilon_random
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.max_num_edges_per_node = max_num_edges_per_node
        self.distance_array = np.zeros((self.max_num_edges_per_node,), dtype=np.float64)
        self.random_indices = np.zeros((self.num_random_edges,), dtype=np.int64)
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.node_class = KinoTIEBTreeNode

        if self.distance_metric_state_size == 2:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end
        elif self.distance_metric_state_size == 3:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end_3d

    def get_random_edge_index(self):
        return self.rng.randint(0,self.edge_bundle.num_edges)

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
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                        self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                        self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        timestep, self.minimum_time_step)

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
            if self.dynamic_col_checker_to_end(new_state, self.agent.radius, 
                                            self.dynamic_agent_obstacles, 
                                            self.dynamic_agent_clearance,
                                            total_elapsed_time,
                                            self.minimum_time_step):
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
                    if self.dynamic_col_checker_to_end(intermediate_state, self.agent.radius, 
                                        self.dynamic_agent_obstacles, 
                                        self.dynamic_agent_clearance,
                                        total_elapsed_time,
                                        self.minimum_time_step):
                        if self.debug_flag:
                            print(f"{debug_prefix}Intermediate goal state will collide \
                                  with high-priority agent. Trying again!")
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

        # Collision check
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                        self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                        self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        timestep, self.minimum_time_step)

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
            if self.dynamic_col_checker_to_end(new_state, self.agent.radius,
                                            self.dynamic_agent_obstacles,
                                            self.dynamic_agent_clearance,
                                            total_elapsed_time,
                                            self.minimum_time_step):
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
                    if self.dynamic_col_checker_to_end(intermediate_state, self.agent.radius,
                                        self.dynamic_agent_obstacles,
                                        self.dynamic_agent_clearance,
                                        total_elapsed_time,
                                        self.minimum_time_step):
                        if self.debug_flag:
                            print(f"{debug_prefix}Intermediate goal state will collide with high-priority agent. Trying again!")
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

    def extend_tree(self, parent_node_id, parent_node, random_point):

        if self.epsilon_random > 0.0 and self.rng.random() < self.epsilon_random:
            for _ in range(self.num_random_edges):
                if self._try_random_control(parent_node, parent_node_id, random_point):
                    return  # node added
            return

        eb = self.edge_bundle
        # p = self.num_skip_edges
        num_samples = self.num_skip_edges

        if parent_node.edge_bundle_indices is None:
            # Edges have not been found before for this node.
            query = self.get_eb_kd_tree_query(parent_node.state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            l = min(len(edge_ids), self.max_num_edges_per_node)
            parent_node.edge_bundle_indices = edge_ids[:l]
            parent_node.edge_bundle_mask = np.full((l,), False, dtype=bool)

        # Keeps track of all the edges from the bundle available for this node
        curr_edge_indices = parent_node.edge_bundle_indices
        # Keeps track of which edges have already been tried for this node
        curr_edge_mask = parent_node.edge_bundle_mask

        sorted_indices, num_valid_edges = self.sort_edges(parent_node.state,
            random_point, eb.start_states, eb.final_states, curr_edge_indices,
            curr_edge_mask, self.distance_array)

        p = max(1, num_valid_edges // num_samples)
        # 1) Greedy / sorted pass over valid edges, skipping every p-th
        for idx in range(0, num_valid_edges, p):
            x = sorted_indices[idx]     # index into curr_edge_indices / mask
            edge_bundle_index = curr_edge_indices[x]

            if self._try_edge_from_bundle(edge_bundle_index,
                parent_node, parent_node_id, x, curr_edge_mask,
                debug_prefix="[sorted] "):
                return  # node added

        for _ in range(self.num_random_edges):
            if self._try_random_control(parent_node, parent_node_id,
                                        random_point):
                return  # node added

        return




"""
import sys
sys.path.append('./src')
import numpy as np
from edge_bundle import EdgeBundle

edge_bundle_file_location = 'edge_bundles/eb_unicycle_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)

from kd_tree_unicycle import CircularAngleIndexNumba
edge_ids = np.arange(kd_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kd_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
kd_tree_kino_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)

θq, δ = 6.25, 0.05
ids = kd_tree_kino_TI_eb_unicycle.radius_query(θq, δ)           # np.ndarray of IDs



"""




"""
class A:
    def __init__(self,a):
        self.x = a
    def set_x(self,b):
        self.x[:] = b

a1 = np.array([43,53,64,75,86])
a2 = np.array([21,31,41,51,61])
o = A(a1)
print(o.x)
v = o.x
print(v)
o.set_x(a2)
print(o.x)
print(v)


"""
