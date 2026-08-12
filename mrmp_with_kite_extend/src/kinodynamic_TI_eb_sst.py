import numpy as np
from utils import check_dynamic_collisions_to_end
from sst import SSTNodeMatrix, SSTWitnessMatrix, SST
from numba.typed import List
from numba import types, njit


class KiteSSTNodeMatrix(SSTNodeMatrix):
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length,
                 max_num_edges_per_node=1000):
        super().__init__(initial_capacity, state_dim, action_dim, max_sub_path_length)
        self.edge_bundle_indices = np.full((initial_capacity, max_num_edges_per_node), 
                                           -1, dtype=np.int64)
        self.edge_bundle_mask = np.full((initial_capacity, max_num_edges_per_node), 
                                        False, dtype=bool)
        self.edge_bundle_count = np.full((initial_capacity,), -1, dtype=np.int64)
        self.max_num_edges_per_node = max_num_edges_per_node


    def set_edge_bundle_indices(self, node_index, edge_bundle_indices: np.ndarray):
        l = min(len(edge_bundle_indices), self.max_num_edges_per_node)
        self.edge_bundle_indices[node_index][:l] = edge_bundle_indices[:l]
        self.edge_bundle_count[node_index] = l


    def set_edge_bundle_mask(self, node_index, edge_index):
        self.edge_bundle_mask[node_index][edge_index] = True


    def _grow(self):
        super()._grow()
        old_cap = self.edge_bundle_indices.shape[0]
        new_cap = old_cap * 2

        new_edge_bundle_indices = np.full((new_cap, self.max_num_edges_per_node), 
                                         -1, dtype=np.int64)
        new_edge_bundle_mask = np.full((new_cap, self.max_num_edges_per_node), 
                                      False, dtype=bool)
        new_edge_bundle_count = np.full((new_cap,), -1, dtype=np.int64)
        
        new_edge_bundle_indices[:old_cap] = self.edge_bundle_indices
        new_edge_bundle_mask[:old_cap] = self.edge_bundle_mask
        new_edge_bundle_count[:old_cap] = self.edge_bundle_count
                
        self.edge_bundle_indices = new_edge_bundle_indices
        self.edge_bundle_mask = new_edge_bundle_mask
        self.edge_bundle_count = new_edge_bundle_count

    def get_edges_for_node(self, node_index):
        l = self.edge_bundle_count[node_index]
        if l < 0:
            return None, None
        return self.edge_bundle_indices[node_index][:l], self.edge_bundle_mask[node_index][:l]


class KiteSST(SST):
    def __init__(self, * , start, goal, goal_radius, env, agent,
                edge_bundle,
                use_fixed_sampling_time=True,
                sampling_time_step=1.5,
                minimum_time_step=0.1,
                max_iter = 10000,
                planning_time = 10.0,
                isvalid_function,
                cost_function,
                reached_goal_function,
                random_point_function,
                translate_function,
                sort_edges_function,
                best_near_radius=2.0,
                prune_radius=0.5,
                mrmp_planning=False,
                witness_time_radius=None,
                max_num_edges_per_node=1000,
                num_skip_edges=50,
                num_random_edges=10,
                epsilon_random=0.01,
                eb_kd_tree,
                get_eb_kd_tree_query,
                kd_tree_delta_radius=0.5,
                udf_seed,
                goal_sampling_probability=0.1,
                dynamic_agent_clearance=0.0,
                debug_flag=False,
                print_logs=False,
                dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))
                ):

        super().__init__(start=start, goal=goal, goal_radius=goal_radius,
                         env=env, agent=agent,
                         use_fixed_sampling_time=use_fixed_sampling_time,
                         sampling_time_step=sampling_time_step,
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter,
                         planning_time=planning_time,
                         isvalid_function=isvalid_function,
                         cost_function=cost_function,
                         reached_goal_function=reached_goal_function,
                         random_point_function=random_point_function,
                         best_near_radius=best_near_radius,
                         prune_radius=prune_radius,
                         mrmp_planning=mrmp_planning,
                         witness_time_radius=witness_time_radius,
                         udf_seed=udf_seed,
                         goal_sampling_probability=goal_sampling_probability,
                         dynamic_agent_clearance=dynamic_agent_clearance,
                         debug_flag=debug_flag,
                         print_logs=print_logs,
                         dynamic_obstacles=dynamic_obstacles
                        )


        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        if epsilon_random < 0.0 or epsilon_random > 1.0:
            raise ValueError("epsilon_random must be between 0.0 and 1.0")
        if num_skip_edges <= 0:
            raise ValueError("num_skip_edges must be greater than 0")
        self.epsilon_random = epsilon_random
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.max_num_edges_per_node = max_num_edges_per_node
        self.distance_array = np.zeros((self.max_num_edges_per_node,), dtype=np.float64)
        self.random_indices = np.zeros((self.num_random_edges,), dtype=np.int64)
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        
        self._node_matrix = KiteSSTNodeMatrix(initial_capacity=1024,
                                state_dim=self.state_length,
                                action_dim=self.action_length,
                                max_sub_path_length=self.max_sub_path_length,
                                max_num_edges_per_node=max_num_edges_per_node)
        self.type_node_matrix = type(self._node_matrix)
        
    
    def extend_tree(self, parent_node_index, random_point):
        """
        Return a KiteSST extension candidate without inserting it. Base
        SST.plan_path() performs witness acceptance and inserts accepted nodes.
        """

        eb = self.edge_bundle
        num_samples = self.num_skip_edges
        tree_nodes = self._node_matrix
        parent_state = tree_nodes.state[parent_node_index]
        parent_time_elapsed = tree_nodes.time_elapsed[parent_node_index]
        parent_cost = tree_nodes.cost[parent_node_index]

        if self.epsilon_random > 0.0 and self.rng.random() < self.epsilon_random:
            for _ in range(self.num_random_edges):
                candidate = self._try_random_control(parent_node_index, parent_state,
                                                     parent_time_elapsed, parent_cost)
                if candidate is not None:
                    return candidate
            return None

        curr_edge_indices, curr_edge_mask = tree_nodes.get_edges_for_node(parent_node_index)
        if curr_edge_indices is None:
            query = self.get_eb_kd_tree_query(parent_state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            if len(edge_ids) > self.max_num_edges_per_node:
                # Radius-query results have a structured KD-index order. Draw a
                # uniform subset only when the per-node cache cap is exceeded.
                edge_ids = self.rng.choice(
                    edge_ids,
                    size=self.max_num_edges_per_node,
                    replace=False,
                )
            self._node_matrix.set_edge_bundle_indices(parent_node_index, edge_ids)
            curr_edge_indices, curr_edge_mask = tree_nodes.get_edges_for_node(parent_node_index)

        sorted_indices, num_valid_edges = self.sort_edges(parent_state, random_point,
                        eb.start_states, eb.final_states, eb.timesteps,
                        curr_edge_indices, curr_edge_mask, self.distance_array)

        if num_valid_edges > 0:
            p = max(1, num_valid_edges // num_samples)
            for idx in range(0, num_valid_edges, p):
                x = sorted_indices[idx]
                edge_idx = curr_edge_indices[x]
                candidate = self._build_candidate_from_action(parent_node_index, parent_state,
                                                              parent_time_elapsed, parent_cost,
                                                              eb.actions[edge_idx], eb.timesteps[edge_idx])
                curr_edge_mask[x] = True
                if candidate is not None:
                    return candidate

        for _ in range(self.num_random_edges):
            candidate = self._try_random_control(parent_node_index, parent_state,
                                                 parent_time_elapsed, parent_cost)
            if candidate is not None:
                return candidate

        return None


    def _try_random_control(self, parent_node_index, parent_state,
                            parent_time_elapsed, parent_cost):
        action = self.agent.get_random_action(self.rng)
        timestep = self.get_time()
        return self._build_candidate_from_action(parent_node_index, parent_state,
                                                 parent_time_elapsed, parent_cost,
                                                 action, timestep)


    def _build_candidate_from_action(self, parent_node_index, parent_state, parent_time_elapsed,
                                     parent_cost, action, timestep):
        num_record_steps = round(timestep / self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_state, action,
                                                timestep, num_steps=num_record_steps)
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius,
                                    self.env.size,
                                    self.static_circular_obstacles,
                                    self.static_rectangular_obstacles,
                                    self.dynamic_agent_obstacles,
                                    self.agent.dynamic_limit_indices,
                                    self.agent.dynamic_limit_values,
                                    self.env.obstacle_buffer,
                                    self.dynamic_agent_clearance,
                                    self.env.boundary_buffer,
                                    parent_time_elapsed,
                                    timestep,
                                    self.minimum_time_step)
        if not accept_new_node:
            if self.debug_flag:
                print("~~~~~~~~~~Sampled New KiteSST Node is invalid. Trying again!~~~~~~~~~~")
                print("Invalid Node : ", new_state)
            return None

        # Check goal at the final state. If the state reaches the goal but
        # cannot be parked safely until the end, keep the edge as a transit
        # candidate instead of rejecting it. This matches RRT/KinoTIEBRRT.
        reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal,
                                                             self.goal_radius, self.agent)
        if reached_goal_flag:
            total_elapsed_time = parent_time_elapsed + timestep
            if self.dynamic_col_checker_to_end(new_state, self.agent.radius,
                                               self.dynamic_agent_obstacles,
                                               self.dynamic_agent_clearance,
                                               total_elapsed_time,
                                               self.minimum_time_step):
                if self.debug_flag:
                    print("Goal state cannot be parked safely yet. Adding it as a transit candidate.")
            else:
                edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                      timestep, path_to_new_state)
                total_cost = parent_cost + edge_cost
                if self.debug_flag:
                    print("Goal reached by proposed KiteSST candidate for ", self.agent.id)
                return (new_state, path_to_new_state, action, timestep,
                        total_elapsed_time, total_cost, True)

        # Check whether the rollout hits the goal at an intermediate state.
        # Unsafe parking at one intermediate goal hit should not invalidate the
        # whole edge; keep scanning later states, then fall through to transit.
        if goal_distance < self.threshold:
            total_elapsed_time = parent_time_elapsed
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
                            print("Intermediate goal state will collide with high-priority agent. Trying again!")
                        continue

                    modified_edge_time = total_elapsed_time - parent_time_elapsed
                    new_path_to_new_state = path_to_new_state[:index+1]
                    edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                          modified_edge_time, new_path_to_new_state)
                    total_cost = parent_cost + edge_cost
                    if self.debug_flag:
                        print("Goal reached by proposed KiteSST candidate for ", self.agent.id)
                    return (intermediate_state, new_path_to_new_state, action,
                            modified_edge_time, total_elapsed_time, total_cost, True)

        edge_cost = self.cost(self.env, self.agent, parent_state, action,
                              timestep, path_to_new_state)
        total_elapsed_time = parent_time_elapsed + timestep
        total_cost = parent_cost + edge_cost
        if self.debug_flag:
            print("New KiteSST candidate generated")
        return (new_state, path_to_new_state, action, timestep,
                total_elapsed_time, total_cost, False)







"""
import sys
sys.path.append('./src')
import numpy as np
from edge_bundle import EdgeBundle

edge_bundle_file_location = 'edge_bundles/eb_unicycle_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)

from kd_tree_unicycle import CircularAngleIndexNumba
edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
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
