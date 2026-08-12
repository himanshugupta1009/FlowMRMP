import numpy as np
from numba.typed import List
from numba import types

from cSST import CSST, CSSTNodeMatrix
from utils import euclidean_distance_numba_with_l


class KiteCSSTNodeMatrix(CSSTNodeMatrix):
    def __init__(self, initial_capacity, joint_state_dim, matrix_state_dim,
                 joint_action_dim, max_sub_path_length, num_agents,
                 max_num_edges_per_node=1000):
        super().__init__(initial_capacity, joint_state_dim, matrix_state_dim,
                         joint_action_dim, max_sub_path_length, num_agents)
        self.edge_bundle_indices = np.full(
            (initial_capacity, num_agents, max_num_edges_per_node),
            -1, dtype=np.int64)
        self.edge_bundle_mask = np.full(
            (initial_capacity, num_agents, max_num_edges_per_node),
            False, dtype=bool)
        self.edge_bundle_count = np.full(
            (initial_capacity, num_agents), -1, dtype=np.int64)
        self.max_num_edges_per_node = max_num_edges_per_node

    def set_edge_bundle_indices(self, node_index, agent_index, edge_bundle_indices):
        l = min(len(edge_bundle_indices), self.max_num_edges_per_node)
        self.edge_bundle_indices[node_index, agent_index, :l] = edge_bundle_indices[:l]
        self.edge_bundle_mask[node_index, agent_index, :l] = False
        self.edge_bundle_count[node_index, agent_index] = l

    def get_edges_for_node_agent(self, node_index, agent_index):
        l = self.edge_bundle_count[node_index, agent_index]
        if l < 0:
            return None, None
        return (self.edge_bundle_indices[node_index, agent_index, :l],
                self.edge_bundle_mask[node_index, agent_index, :l])

    def _grow(self):
        old_cap = self.state.shape[0]
        super()._grow()
        new_cap = self.state.shape[0]

        new_edge_bundle_indices = np.full(
            (new_cap, self.num_agents, self.max_num_edges_per_node),
            -1, dtype=np.int64)
        new_edge_bundle_mask = np.full(
            (new_cap, self.num_agents, self.max_num_edges_per_node),
            False, dtype=bool)
        new_edge_bundle_count = np.full(
            (new_cap, self.num_agents), -1, dtype=np.int64)

        new_edge_bundle_indices[:old_cap] = self.edge_bundle_indices
        new_edge_bundle_mask[:old_cap] = self.edge_bundle_mask
        new_edge_bundle_count[:old_cap] = self.edge_bundle_count

        self.edge_bundle_indices = new_edge_bundle_indices
        self.edge_bundle_mask = new_edge_bundle_mask
        self.edge_bundle_count = new_edge_bundle_count


class KiteCSST(CSST):
    def __init__(self, *, agents, starts, goals, goal_radii, env,
                 edge_bundle,
                 use_fixed_sampling_time=True,
                 sampling_time_step=1.0,
                 minimum_time_step=0.1,
                 max_iter=1000,
                 planning_time=10.0,
                 num_extension_trials=1,
                 truncation_check_threshold=1.0,
                 isvalid_function,
                 cost_function,
                 reached_goal_function,
                 translate_function,
                 random_point_function,
                 sort_edges_function,
                 udf_seed=77,
                 goal_sampling_probability=0.1,
                 best_near_radius=1.0,
                 prune_radius=0.5,
                 mrmp_planning=True,
                 witness_time_radius=None,
                 objective="sum_of_costs",
                 dynamic_agent_clearance=0.0,
                 dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C')),
                 eb_kd_trees=None,
                 get_eb_kd_tree_query_funcs=None,
                 kd_tree_delta_radius=0.5,
                 max_num_edges_per_node=None,
                 num_edge_candidates_per_agent=10,
                 max_joint_edge_trials=20,
                 epsilon_random=0.01,
                 fallback_to_random_control=True,
                 truncate_paths=False,
                 branch_goal_parking=True,
                 print_logs=False,
                 debug_flag=False):
        if eb_kd_trees is None:
            raise ValueError("eb_kd_trees must be provided")
        if get_eb_kd_tree_query_funcs is None:
            raise ValueError("get_eb_kd_tree_query_funcs must be provided")
        if max_num_edges_per_node is None:
            raise ValueError("max_num_edges_per_node must be provided")
        if epsilon_random < 0.0 or epsilon_random > 1.0:
            raise ValueError("epsilon_random must be between 0.0 and 1.0")

        CSST.__init__(
            self,
            agents=agents,
            starts=starts,
            goals=goals,
            goal_radii=goal_radii,
            env=env,
            use_fixed_sampling_time=use_fixed_sampling_time,
            sampling_time_step=sampling_time_step,
            minimum_time_step=minimum_time_step,
            max_iter=max_iter,
            planning_time=planning_time,
            num_extension_trials=num_extension_trials,
            truncation_check_threshold=truncation_check_threshold,
            isvalid_function=isvalid_function,
            cost_function=cost_function,
            reached_goal_function=reached_goal_function,
            random_point_function=random_point_function,
            udf_seed=udf_seed,
            goal_sampling_probability=goal_sampling_probability,
            best_near_radius=best_near_radius,
            prune_radius=prune_radius,
            mrmp_planning=mrmp_planning,
            witness_time_radius=witness_time_radius,
            objective=objective,
            dynamic_agent_clearance=dynamic_agent_clearance,
            dynamic_obstacles=dynamic_obstacles,
            truncate_paths=truncate_paths,
            branch_goal_parking=branch_goal_parking,
            print_logs=print_logs,
            debug_flag=debug_flag)

        self.edge_bundles = edge_bundle
        self.eb_kd_trees = eb_kd_trees
        self.get_eb_kd_tree_query_funcs = get_eb_kd_tree_query_funcs
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.max_num_edges_per_node = max_num_edges_per_node
        self.num_edge_candidates_per_agent = num_edge_candidates_per_agent
        self.max_joint_edge_trials = max_joint_edge_trials
        self.epsilon_random = epsilon_random
        self.fallback_to_random_control = fallback_to_random_control
        self.distance_array = [np.zeros(self.max_num_edges_per_node, dtype=np.float64)
                               for _ in edge_bundle]
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.type_node_matrix = KiteCSSTNodeMatrix
        self.reset_tree()

    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree is None:
            self._node_matrix = self.type_node_matrix(
                initial_capacity=1024,
                joint_state_dim=self.joint_state_size,
                matrix_state_dim=self.distance_metric_state_size,
                joint_action_dim=self.joint_action_size,
                max_sub_path_length=self.max_sub_path_length,
                num_agents=self.num_agents,
                max_num_edges_per_node=self.max_num_edges_per_node)
            self._witness_matrix = self.type_witness_matrix(
                initial_capacity=1024,
                state_dim=self.distance_metric_state_size)
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = np.full(self.num_agents, np.inf, dtype=np.float64)
            self.path_scalar_cost = np.inf
            self.path_time = 0.0
            self.first_solution_node_id = None
            self.first_solution_planning_time = np.inf
            self.first_solution_path_time = 0.0
            self.first_solution_path_cost = np.inf
            self.first_solution_per_agent_cost = np.full(self.num_agents, np.inf, dtype=np.float64)
            self.last_added_node_id = -1
            self.last_added_witness_id = -1
            self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)
        else:
            self._node_matrix = some_existing_tree[0]
            self._witness_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = np.full(self.num_agents, np.inf, dtype=np.float64)
            self.path_scalar_cost = np.inf
            self.path_time = 0.0
            self.first_solution_node_id = None
            self.first_solution_planning_time = np.inf
            self.first_solution_path_time = 0.0
            self.first_solution_path_cost = np.inf
            self.first_solution_per_agent_cost = np.full(self.num_agents, np.inf, dtype=np.float64)
            self.last_added_node_id = self._node_matrix.count - 1
            self.last_added_witness_id = self._witness_matrix.count - 1
            self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)

    def _active_agent_indices(self, parent_index):
        if not self.branch_goal_parking:
            return list(range(self.num_agents))
        return [i for i in range(self.num_agents)
                if not self._node_matrix.reached_goals[parent_index, i]]

    def get_random_edge_index(self, agent_index):
        return self.rng.integers(0, self.edge_bundles[agent_index].num_edges)

    def _ensure_agent_edge_cache(self, parent_index, agent_index):
        edge_ids, _ = self._node_matrix.get_edges_for_node_agent(parent_index, agent_index)
        if edge_ids is not None:
            return

        parent_agent_state = self.get_agent_state(
            self._node_matrix.state[parent_index], agent_index)
        query = self.get_eb_kd_tree_query_funcs[agent_index](parent_agent_state)
        edge_ids = self.eb_kd_trees[agent_index].radius_query(
            query, self.kd_tree_delta_radius)
        if len(edge_ids) > self.max_num_edges_per_node:
            # Radius-query results have a structured KD-index order. Draw a
            # uniform subset only when the per-node cache cap is exceeded.
            edge_ids = self.rng.choice(
                edge_ids,
                size=self.max_num_edges_per_node,
                replace=False,
            )
        self._node_matrix.set_edge_bundle_indices(
            parent_index, agent_index, edge_ids)

    def _try_agent_edge(self, parent_index, agent_index, edge_bundle_index,
                        mask_index, curr_edge_mask, debug_prefix):
        agent = self.agents[agent_index]
        eb = self.edge_bundles[agent_index]
        parent_agent_state = self.get_agent_state(
            self._node_matrix.state[parent_index], agent_index)

        action = eb.actions[edge_bundle_index]
        timestep = eb.timesteps[edge_bundle_index]
        num_record_steps = round(timestep / self.minimum_time_step)
        if num_record_steps <= 0:
            curr_edge_mask[mask_index] = True
            return None

        new_substate, path_to_new_state = agent.get_next_state(
            parent_agent_state, action, timestep, num_record_steps)

        state_is_valid = self.isvalid[agent_index](
            path_to_new_state, agent.radius, self.env.size,
            self.static_circular_obstacles,
            self.static_rectangular_obstacles,
            self.dynamic_agent_obstacles,
            agent.dynamic_limit_indices,
            agent.dynamic_limit_values,
            self.env.obstacle_buffer,
            self.dynamic_agent_clearance,
            self.env.boundary_buffer,
            self._node_matrix.time_elapsed[parent_index],
            timestep,
            self.minimum_time_step)

        if not state_is_valid:
            curr_edge_mask[mask_index] = True
            if self.debug_flag:
                print(f"{debug_prefix}Sampled KiteCSST edge for agent "
                      f"{agent_index} is invalid. Trying again!")
                print("Invalid State:", new_substate)
            return None

        return action, timestep, path_to_new_state

    def _collect_agent_candidate_mask_indices(self, parent_index, agent_index,
                                              random_point):
        self._ensure_agent_edge_cache(parent_index, agent_index)

        eb = self.edge_bundles[agent_index]
        curr_edge_indices, curr_edge_mask = self._node_matrix.get_edges_for_node_agent(
            parent_index, agent_index)
        if curr_edge_indices.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        parent_agent_state = self.get_agent_state(
            self._node_matrix.state[parent_index], agent_index)
        position_start = agent_index * self.agent_position_state_dim
        position_end = position_start + self.agent_position_state_dim
        agent_random_point = random_point[position_start:position_end]

        sorted_indices, num_valid_edges = self.sort_edges[agent_index](
            parent_agent_state, agent_random_point,
            eb.start_states, eb.final_states, eb.timesteps,
            curr_edge_indices, curr_edge_mask,
            self.distance_array[agent_index])

        if num_valid_edges <= 0:
            if self.debug_flag:
                print(f"No valid edge-bundle options found for agent {agent_index}.")
                print("Parent Substate:", parent_agent_state)
            return np.empty(0, dtype=np.int64)

        num_samples = max(1, self.num_edge_candidates_per_agent)
        stride = max(1, num_valid_edges // num_samples)
        num_candidates = min(num_samples, ((num_valid_edges - 1) // stride) + 1)
        candidate_mask_indices = np.empty(num_candidates, dtype=np.int64)

        write_index = 0
        for sorted_index in range(0, num_valid_edges, stride):
            if write_index >= num_candidates:
                break
            candidate_mask_indices[write_index] = sorted_indices[sorted_index]
            write_index += 1

        return candidate_mask_indices

    def _get_or_try_agent_candidate(self, parent_index, agent_index,
                                    candidate_mask_indices,
                                    candidate_cache,
                                    local_candidate_index):
        cached_candidate = candidate_cache[local_candidate_index]
        if cached_candidate is False:
            return None
        if cached_candidate is not None:
            return cached_candidate

        curr_edge_indices, curr_edge_mask = self._node_matrix.get_edges_for_node_agent(
            parent_index, agent_index)
        mask_index = candidate_mask_indices[local_candidate_index]
        edge_bundle_index = curr_edge_indices[mask_index]

        candidate = self._try_agent_edge(
            parent_index, agent_index, edge_bundle_index, mask_index,
            curr_edge_mask, "")
        if candidate is None:
            candidate_cache[local_candidate_index] = False
            return None

        candidate_cache[local_candidate_index] = candidate
        return candidate

    def _build_joint_candidate_from_selected_edges(self, selected_edges,
                                                   random_point):
        edge_time = np.inf
        for agent_index in range(self.num_agents):
            timestep = selected_edges[agent_index][1]
            if timestep < edge_time:
                edge_time = timestep

        num_steps = round(edge_time / self.minimum_time_step)
        if num_steps <= 0:
            return None
        edge_time = num_steps * self.minimum_time_step

        joint_action = np.empty(self.joint_action_size, dtype=np.float64)
        joint_path = np.empty((num_steps, self.joint_state_size), dtype=np.float64)
        new_joint_state = np.empty(self.joint_state_size, dtype=np.float64)

        for agent_index in range(self.num_agents):
            action, _, path_to_new_state = selected_edges[agent_index]
            agent_path = path_to_new_state[:num_steps]
            if agent_path.shape[0] != num_steps:
                return None
            self.set_agent_action(joint_action, agent_index, action)
            self.set_agent_path(joint_path, agent_index, agent_path)
            self.set_agent_state(new_joint_state, agent_index, agent_path[-1])

        matrix_state = self.superstate_to_matrix_state(new_joint_state)
        score = euclidean_distance_numba_with_l(
            matrix_state, random_point, self.distance_metric_state_size)
        return new_joint_state, matrix_state, joint_path, joint_action, edge_time, score

    def _build_branch_goal_joint_candidate_from_selected_edges(
            self, parent_index, selected_edges, active_agent_indices,
            random_point):
        edge_time = np.inf
        for agent_index in active_agent_indices:
            timestep = selected_edges[agent_index][1]
            if timestep < edge_time:
                edge_time = timestep

        num_steps = round(edge_time / self.minimum_time_step)
        if num_steps <= 0:
            return None
        edge_time = num_steps * self.minimum_time_step

        joint_action = np.zeros(self.joint_action_size, dtype=np.float64)
        joint_path = np.empty((num_steps, self.joint_state_size), dtype=np.float64)
        new_joint_state = np.empty(self.joint_state_size, dtype=np.float64)

        for agent_index in range(self.num_agents):
            if self._agent_is_parked_index(parent_index, agent_index):
                self._fill_parked_agent_index(
                    joint_action, joint_path, new_joint_state,
                    parent_index, agent_index)
                continue

            action, _, path_to_new_state = selected_edges[agent_index]
            agent_path = path_to_new_state[:num_steps]
            if agent_path.shape[0] != num_steps:
                return None
            self.set_agent_action(joint_action, agent_index, action)
            self.set_agent_path(joint_path, agent_index, agent_path)
            self.set_agent_state(new_joint_state, agent_index, agent_path[-1])

        matrix_state = self.superstate_to_matrix_state(new_joint_state)
        score = euclidean_distance_numba_with_l(
            matrix_state, random_point, self.distance_metric_state_size)
        return new_joint_state, matrix_state, joint_path, joint_action, edge_time, score

    def _increment_collision_agent(self, candidate_indices,
                                   agent_candidate_mask_indices,
                                   first_agent, second_agent):
        first_active = first_agent >= 0 and first_agent in agent_candidate_mask_indices
        second_active = second_agent >= 0 and second_agent in agent_candidate_mask_indices

        if first_active and second_active:
            if self.rng.random() < 0.5:
                ordered_agents = (first_agent, second_agent)
            else:
                ordered_agents = (second_agent, first_agent)
        elif first_active:
            ordered_agents = (first_agent,)
        elif second_active:
            ordered_agents = (second_agent,)
        else:
            return False

        for agent_index in ordered_agents:
            if (candidate_indices[agent_index] + 1 <
                    agent_candidate_mask_indices[agent_index].shape[0]):
                candidate_indices[agent_index] += 1
                return True
        return False

    def _select_edge_bundle_joint_candidate(self, parent_index, random_point):
        active_agent_indices = self._active_agent_indices(parent_index)
        if len(active_agent_indices) == 0:
            return None

        if self.branch_goal_parking:
            agent_candidate_mask_indices = {}
            agent_candidate_caches = {}
        else:
            agent_candidate_mask_indices = []
            agent_candidate_caches = []

        for agent_index in active_agent_indices:
            candidate_mask_indices = self._collect_agent_candidate_mask_indices(
                parent_index, agent_index, random_point)
            if candidate_mask_indices.shape[0] == 0:
                return None
            if self.branch_goal_parking:
                agent_candidate_mask_indices[agent_index] = candidate_mask_indices
                agent_candidate_caches[agent_index] = [None for _ in range(
                    candidate_mask_indices.shape[0])]
            else:
                agent_candidate_mask_indices.append(candidate_mask_indices)
                agent_candidate_caches.append([None for _ in range(
                    candidate_mask_indices.shape[0])])

        if self.branch_goal_parking:
            candidate_indices = {agent_index: 0 for agent_index in active_agent_indices}
        else:
            candidate_indices = np.zeros(self.num_agents, dtype=np.int64)
        selected_edges = [None for _ in range(self.num_agents)]

        for _ in range(self.max_joint_edge_trials):
            for agent_index in active_agent_indices:
                candidate_mask_indices = agent_candidate_mask_indices[agent_index]
                candidate_cache = agent_candidate_caches[agent_index]
                while candidate_indices[agent_index] < candidate_mask_indices.shape[0]:
                    candidate = self._get_or_try_agent_candidate(
                        parent_index, agent_index, candidate_mask_indices,
                        candidate_cache, candidate_indices[agent_index])
                    if candidate is not None:
                        selected_edges[agent_index] = candidate
                        break
                    candidate_indices[agent_index] += 1
                if candidate_indices[agent_index] >= candidate_mask_indices.shape[0]:
                    return None

            if self.branch_goal_parking:
                candidate = self._build_branch_goal_joint_candidate_from_selected_edges(
                    parent_index, selected_edges, active_agent_indices, random_point)
            else:
                candidate = self._build_joint_candidate_from_selected_edges(
                    selected_edges, random_point)
            if candidate is None:
                return None

            collides, first_agent, second_agent, _ = self.first_joint_path_collision(
                candidate[2])
            if not collides:
                return candidate

            if self.branch_goal_parking:
                if not self._increment_collision_agent(
                        candidate_indices, agent_candidate_mask_indices,
                        first_agent, second_agent):
                    return None
                continue

            if self.rng.random() < 0.5:
                primary_agent = first_agent
                secondary_agent = second_agent
            else:
                primary_agent = second_agent
                secondary_agent = first_agent

            if (candidate_indices[primary_agent] + 1 <
                    agent_candidate_mask_indices[primary_agent].shape[0]):
                candidate_indices[primary_agent] += 1
            elif (candidate_indices[secondary_agent] + 1 <
                    agent_candidate_mask_indices[secondary_agent].shape[0]):
                candidate_indices[secondary_agent] += 1
            else:
                return None

        return None

    def _select_random_joint_candidate(self, parent_index, random_point):
        best_candidate = super()._select_best_joint_extension_candidate(
            parent_index, random_point)
        if best_candidate is None:
            return None
        new_joint_state, matrix_state, joint_path, joint_action, random_time = best_candidate
        score = euclidean_distance_numba_with_l(
            matrix_state, random_point, self.distance_metric_state_size)
        return new_joint_state, matrix_state, joint_path, joint_action, random_time, score

    def _select_best_joint_extension_candidate(self, parent_index, random_point):
        if self.rng.random() < self.epsilon_random:
            random_candidate = self._select_random_joint_candidate(parent_index, random_point)
            if random_candidate is not None:
                return random_candidate[:-1]
            return None

        candidate = self._select_edge_bundle_joint_candidate(parent_index, random_point)
        if candidate is not None:
            return candidate[:-1]

        if self.fallback_to_random_control:
            random_candidate = self._select_random_joint_candidate(parent_index, random_point)
            if random_candidate is not None:
                return random_candidate[:-1]

        return None
