import numpy as np
from numba.typed import List
from numba import types

from cRRT import CRRT
from kinodynamic_TI_eb_rrt import KinoTIEBTreeNode
from utils import euclidean_distance_numba_with_l


class CrrtKinoTIEBTreeNode(KinoTIEBTreeNode):
    def __init__(self, sid, state, parent_id, parent_action,
                 parent_action_duration, path_from_parent, time_so_far,
                 cost, reached_goals):
        super().__init__(sid, state, parent_id, parent_action,
                         parent_action_duration, path_from_parent,
                         time_so_far, cost)
        self.reached_goals = reached_goals


class KinoTIEBCRRT(CRRT):
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
        """
        Centralized kinodynamic time-invariant edge-bundle RRT.

        This version is compatible with cRRT.CRRT: tree states, actions,
        and edge paths are flattened joint numpy arrays. Each agent still
        chooses its edge independently from its bundle. The joint edge executes
        for the minimum selected edge duration, so every stored state is a real
        prefix of that agent's kinodynamic rollout.

        Goal handling follows cRRT: branch_goal_parking=True with
        truncate_paths=False is the default MRMP mode.
        """
        if eb_kd_trees is None:
            raise ValueError("eb_kd_trees must be provided")
        if get_eb_kd_tree_query_funcs is None:
            raise ValueError("get_eb_kd_tree_query_funcs must be provided")
        if max_num_edges_per_node is None:
            raise ValueError("max_num_edges_per_node must be provided")
        if epsilon_random < 0.0 or epsilon_random > 1.0:
            raise ValueError("epsilon_random must be between 0.0 and 1.0")

        CRRT.__init__(self, starts=starts, goals=goals,
                      goal_radii=goal_radii, env=env, agents=agents,
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
                      dynamic_agent_clearance=dynamic_agent_clearance,
                      print_logs=print_logs,
                      debug_flag=debug_flag,
                      dynamic_obstacles=dynamic_obstacles,
                      truncate_paths=truncate_paths,
                      branch_goal_parking=branch_goal_parking)

        self.edge_bundles = edge_bundle
        self.eb_kd_trees = eb_kd_trees
        self.get_eb_kd_tree_query_funcs = get_eb_kd_tree_query_funcs
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.max_num_edges_per_node = max_num_edges_per_node

        # EB-specific expansion controls. Each agent prepares a small,
        # approximately evenly spaced set of sorted bundle candidates, then the
        # centralized planner repairs inter-agent collisions by changing only
        # agents that actually collided.
        self.num_edge_candidates_per_agent = num_edge_candidates_per_agent
        self.max_joint_edge_trials = max_joint_edge_trials
        self.epsilon_random = epsilon_random
        self.fallback_to_random_control = fallback_to_random_control

        # Scratch buffers/helpers used by the per-agent sort functions. The
        # distance arrays are reused so sorting candidate EB edges does not
        # allocate a fresh work array on every expansion.
        self.distance_array = [np.zeros(eb.num_edges) for eb in edge_bundle]
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.node_class = CrrtKinoTIEBTreeNode
        self.truncate_paths = truncate_paths

    def _active_agent_indices(self, parent_node):
        # In normal mode, every agent participates in every joint expansion.
        # In branch-goal-parking mode, agents that already reached their goal
        # on this branch are parked and excluded from EB/random action search.
        # A list fits this Python-side loop/dict-key use better than a NumPy
        # array, since these indices are not used for vectorized math.
        if not self.branch_goal_parking:
            return list(range(self.num_agents))
        return [i for i in range(self.num_agents)
                if not parent_node.reached_goals[i]]

    def get_random_edge_index(self, agent_index):
        return self.rng.integers(0, self.edge_bundles[agent_index].num_edges)

    def add_rrt_node(self, state, parent_node_id, parent_action,
                     parent_action_duration, path_from_parent, time_elapsed,
                     cost, reached_goals):
        # Store a normal cRRT node. CrrtKinoTIEBTreeNode inherits the single
        # agent EB cache fields from KinoTIEBTreeNode, and we leave those fields
        # as None here. If this node is later expanded, they are replaced by
        # per-agent cache lists in _ensure_agent_edge_cache.
        new_node_id = self.last_added_node_id + 1
        new_node = CrrtKinoTIEBTreeNode(
            new_node_id, state, parent_node_id, parent_action,
            parent_action_duration, path_from_parent,
            round(time_elapsed, self.roundoff_digits), cost, reached_goals)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id
        self._node_matrix.append(self.superstate_to_matrix_state(state),
                                 new_node_id)
        return new_node_id

    def _ensure_agent_edge_cache(self, parent_node, agent_index):
        # The inherited EB cache fields start as None. Centralized planning
        # needs one cache slot per agent, so allocate those top-level lists only
        # when this node is actually expanded.
        if parent_node.edge_bundle_indices is None:
            parent_node.edge_bundle_indices = [None for _ in range(self.num_agents)]
            parent_node.edge_bundle_mask = [None for _ in range(self.num_agents)]

        # For each node/agent pair, query the EB KD-tree once and cache nearby
        # edge ids. Later expansions from the same node reuse this list and only
        # update the mask for individually invalid edges.
        if parent_node.edge_bundle_indices[agent_index] is not None:
            return

        parent_agent_state = self.get_agent_state(parent_node.state,
                                                  agent_index)
        query = self.get_eb_kd_tree_query_funcs[agent_index](parent_agent_state)
        edge_ids = self.eb_kd_trees[agent_index].radius_query(query, 
                                                self.kd_tree_delta_radius)
        num_edges = min(len(edge_ids), self.max_num_edges_per_node)
        parent_node.edge_bundle_indices[agent_index] = edge_ids[:num_edges]
        parent_node.edge_bundle_mask[agent_index] = np.full(
            (num_edges,), False, dtype=bool)

    def _try_agent_edge(self, parent_node, agent_index, edge_bundle_index,
                        mask_index, curr_edge_mask, debug_prefix):
        # Test one edge-bundle primitive for one agent only. The edge is
        # re-simulated from the parent state instead of directly copying the
        # stored EB trajectory, so the bundle supplies the action and duration
        # while the actual rollout remains anchored at this RRT node.
        agent = self.agents[agent_index]
        eb = self.edge_bundles[agent_index]
        parent_agent_state = self.get_agent_state(parent_node.state,
                                                  agent_index)

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
            parent_node.time_elapsed,
            timestep,
            self.minimum_time_step)

        if not state_is_valid:
            # Only permanently mask edges that are individually infeasible for
            # this node/agent, such as obstacle, boundary, or dynamic-limit
            # failures. Inter-agent conflicts are handled later with a local
            # joint repair loop and should not poison this edge for the node.
            curr_edge_mask[mask_index] = True
            if self.debug_flag:
                print(f"{debug_prefix}Sampled Kino TI EB cRRT state for agent "
                      f"{agent_index} is invalid. Trying again!")
                print("Invalid State:", new_substate)
            return None

        return action, timestep, path_to_new_state

    def _collect_agent_candidate_mask_indices(self, parent_node, agent_index,
                                              random_point):
        # Build only the p-spaced candidate index set for one agent. This does
        # not propagate any edge yet; propagation is delayed until the joint
        # repair loop actually selects a candidate. The sort function orders
        # nearby EB edges by the distance between their translated endpoint and
        # this agent's slice of the sampled joint random point.
        self._ensure_agent_edge_cache(parent_node, agent_index)

        eb = self.edge_bundles[agent_index]
        curr_edge_indices = parent_node.edge_bundle_indices[agent_index]
        curr_edge_mask = parent_node.edge_bundle_mask[agent_index]

        if len(curr_edge_indices) == 0:
            return np.empty(0, dtype=np.int64)

        parent_agent_state = self.get_agent_state(parent_node.state,
                                                  agent_index)
        position_start = agent_index * self.agent_position_state_dim
        position_end = position_start + self.agent_position_state_dim
        agent_random_point = random_point[position_start:position_end]

        sorted_indices, num_valid_edges = self.sort_edges[agent_index](
            parent_agent_state,agent_random_point,
            eb.start_states,eb.final_states,
            curr_edge_indices,curr_edge_mask,
            self.distance_array[agent_index])

        if num_valid_edges <= 0:
            if self.debug_flag:
                print(f"No valid edge-bundle options found for agent "
                      f"{agent_index}.")
                print("Parent Substate:", parent_agent_state)
            return np.empty(0, dtype=np.int64)

        num_samples = max(1, self.num_edge_candidates_per_agent)
        stride = max(1, num_valid_edges // num_samples)
        num_candidates = min(num_samples,
                             ((num_valid_edges - 1) // stride) + 1)
        candidate_mask_indices = np.empty(num_candidates, dtype=np.int64)

        # Walk through the sorted list at an even stride, collecting up to
        # num_edge_candidates_per_agent edge ids. This keeps the local search
        # small without paying for dynamics propagation until a candidate is
        # actually used in a joint trial.
        write_index = 0
        for sorted_index in range(0, num_valid_edges, stride):
            if write_index >= num_candidates:
                break
            candidate_mask_indices[write_index] = sorted_indices[sorted_index]
            write_index += 1

        return candidate_mask_indices

    def _get_or_try_agent_candidate(self, parent_node, agent_index,
                                    candidate_mask_indices,
                                    candidate_cache,
                                    local_candidate_index):
        # Lazily propagate one agent's selected EB candidate. Within this joint
        # expansion, cache both valid rollouts and invalid results so repeated
        # repair attempts do not re-simulate the same primitive.
        cached_candidate = candidate_cache[local_candidate_index]
        if cached_candidate is False:
            return None
        if cached_candidate is not None:
            return cached_candidate

        curr_edge_indices = parent_node.edge_bundle_indices[agent_index]
        curr_edge_mask = parent_node.edge_bundle_mask[agent_index]
        mask_index = candidate_mask_indices[local_candidate_index]
        edge_bundle_index = curr_edge_indices[mask_index]

        candidate = self._try_agent_edge(
            parent_node, agent_index, edge_bundle_index, mask_index,
            curr_edge_mask, "")
        if candidate is None:
            candidate_cache[local_candidate_index] = False
            return None

        candidate_cache[local_candidate_index] = candidate
        return candidate

    def _build_joint_candidate_from_selected_edges(self, selected_edges,
                                                   random_point):
        # Convert one selected per-agent EB candidate into the flattened joint
        # state/action/path representation used by cRRT. Agents may have
        # selected different edge durations, so the joint edge executes only up
        # to the minimum duration. This preserves kinodynamic validity because
        # every stored state is a prefix of an actual rollout, not artificial
        # padding at an endpoint.
        edge_time = np.inf
        for agent_index in range(self.num_agents):
            timestep = selected_edges[agent_index][1]
            if timestep < edge_time:
                edge_time = timestep

        num_steps = round(edge_time / self.minimum_time_step)
        if num_steps <= 0:
            return None
        edge_time = num_steps * self.minimum_time_step

        joint_action = np.empty(self.joint_action_size)
        joint_path = np.empty((num_steps, self.joint_state_size))
        new_joint_state = np.empty(self.joint_state_size)

        for agent_index in range(self.num_agents):
            action, _, path_to_new_state = selected_edges[agent_index]
            agent_path = path_to_new_state[:num_steps]
            if agent_path.shape[0] != num_steps:
                return None
            self.set_agent_action(joint_action, agent_index, action)
            self.set_agent_path(joint_path, agent_index, agent_path)
            self.set_agent_state(new_joint_state, agent_index,
                                 agent_path[-1])

        matrix_state = self.superstate_to_matrix_state(new_joint_state)
        score = euclidean_distance_numba_with_l(
            matrix_state, random_point, self.distance_metric_state_size)
        return new_joint_state, joint_path, joint_action, edge_time, score

    def _build_branch_goal_joint_candidate_from_selected_edges(
            self, parent_node, selected_edges, active_agent_indices,
            random_point):
        # Determine the common joint-edge duration from active agents only.
        # Parked agents can wait for any duration, so they should not shorten
        # or lengthen the selected active-agent rollout.
        edge_time = np.inf
        for agent_index in active_agent_indices:
            timestep = selected_edges[agent_index][1]
            if timestep < edge_time:
                edge_time = timestep

        # Snap the edge duration to an integer number of record steps. This
        # keeps the stored edge time consistent with the discrete joint path.
        num_steps = round(edge_time / self.minimum_time_step)
        if num_steps <= 0:
            return None
        edge_time = num_steps * self.minimum_time_step

        # Allocate flattened joint containers. The action starts at zero so
        # parked agents naturally have zero controls.
        joint_action = np.zeros(self.joint_action_size)
        joint_path = np.empty((num_steps, self.joint_state_size))
        new_joint_state = np.empty(self.joint_state_size)

        # Fill parked agents with repeated parent states, and active agents
        # with the selected EB rollout prefix.
        for agent_index in range(self.num_agents):
            if self._agent_is_parked(parent_node, agent_index):
                self._fill_parked_agent(
                    joint_action, joint_path, new_joint_state,
                    parent_node, agent_index)
                continue

            action, _, path_to_new_state = selected_edges[agent_index]
            agent_path = path_to_new_state[:num_steps]
            if agent_path.shape[0] != num_steps:
                return None
            self.set_agent_action(joint_action, agent_index, action)
            self.set_agent_path(joint_path, agent_index, agent_path)
            self.set_agent_state(new_joint_state, agent_index,
                                 agent_path[-1])

        # Score the candidate by how close its endpoint is to the sampled joint
        # random point, matching the non-branch-parking candidate builder.
        matrix_state = self.superstate_to_matrix_state(new_joint_state)
        score = euclidean_distance_numba_with_l(
            matrix_state, random_point, self.distance_metric_state_size)
        return new_joint_state, joint_path, joint_action, edge_time, score

    def _increment_collision_agent(self, candidate_indices,
                                   agent_candidate_mask_indices,
                                   first_agent, second_agent):
        # In branch-parking mode, a collision can involve a parked agent. Only
        # active agents have candidate lists that can be advanced.
        first_active = first_agent >= 0 and (
            first_agent in agent_candidate_mask_indices)
        second_active = second_agent >= 0 and (
            second_agent in agent_candidate_mask_indices)

        # If both colliding agents are active, randomly choose which candidate
        # to advance first. If only one is active, that one must move around the
        # parked/static participant.
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

        # Advance the first active colliding agent that still has an unused
        # local EB candidate. If neither can advance, this joint repair failed.
        for agent_index in ordered_agents:
            if (candidate_indices[agent_index] + 1 <
                    agent_candidate_mask_indices[agent_index].shape[0]):
                candidate_indices[agent_index] += 1
                return True
        return False

    def _select_edge_bundle_joint_candidate(self, parent_node, random_point):
        # First create a small sorted EB candidate-index list for every agent.
        # These are just indices into the node-local EB cache, not propagated
        # paths. Dynamics rollouts are performed lazily below.
        active_agent_indices = self._active_agent_indices(parent_node)
        if len(active_agent_indices) == 0:
            return None

        # Normal mode uses dense per-agent arrays indexed by agent order.
        # Branch-parking mode uses dicts because parked agents are skipped and
        # have no EB candidate list to repair.
        if self.branch_goal_parking:
            agent_candidate_mask_indices = {}
            agent_candidate_caches = {}
        else:
            agent_candidate_mask_indices = []
            agent_candidate_caches = []

        # Collect sorted, p-spaced EB candidate indices for active agents only.
        for agent_index in active_agent_indices:
            candidate_mask_indices = self._collect_agent_candidate_mask_indices(
                parent_node, agent_index, random_point)
            if candidate_mask_indices.shape[0] == 0:
                return None
            if self.branch_goal_parking:
                agent_candidate_mask_indices[agent_index] = (
                    candidate_mask_indices)
                agent_candidate_caches[agent_index] = (
                    [None for _ in range(candidate_mask_indices.shape[0])])
            else:
                agent_candidate_mask_indices.append(candidate_mask_indices)
                agent_candidate_caches.append(
                    [None for _ in range(candidate_mask_indices.shape[0])])

        # Track which local candidate each active agent is currently using.
        if self.branch_goal_parking:
            candidate_indices = {
                agent_index: 0 for agent_index in active_agent_indices}
        else:
            candidate_indices = np.zeros(self.num_agents, dtype=np.int64)
        selected_edges = [None for _ in range(self.num_agents)]

        # Start with every agent's best candidate. If the joint path collides,
        # identify the first colliding pair and advance one of those agents to
        # its next local candidate. This is deliberately local: a bundle edge
        # that collides with one partner edge may still work with a different
        # partner, so inter-agent conflicts do not update the persistent mask.
        for _ in range(self.max_joint_edge_trials):
            # Lazily materialize one valid EB primitive for each active agent.
            # Individually invalid primitives are cached and skipped.
            for agent_index in active_agent_indices:
                candidate_mask_indices = agent_candidate_mask_indices[agent_index]
                candidate_cache = agent_candidate_caches[agent_index]
                while candidate_indices[agent_index] < candidate_mask_indices.shape[0]:
                    candidate = self._get_or_try_agent_candidate(
                        parent_node, agent_index, candidate_mask_indices,
                        candidate_cache, candidate_indices[agent_index])
                    if candidate is not None:
                        selected_edges[agent_index] = candidate
                        break
                    candidate_indices[agent_index] += 1
                if candidate_indices[agent_index] >= candidate_mask_indices.shape[0]:
                    return None

            # Build a full joint path. Branch-parking mode also fills in parked
            # agents as stationary participants before collision checking.
            if self.branch_goal_parking:
                candidate = self._build_branch_goal_joint_candidate_from_selected_edges(
                    parent_node, selected_edges, active_agent_indices,
                    random_point)
            else:
                candidate = self._build_joint_candidate_from_selected_edges(
                    selected_edges, random_point)
            if candidate is None:
                return None

            collides, first_agent, second_agent, _ = (
                self.first_joint_path_collision(candidate[1]))
            if not collides:
                return candidate

            # In branch-parking mode, repair only active agents involved in the
            # first collision. Parked agents cannot be advanced.
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

    def _select_random_joint_candidate(self, parent_node, random_point):
        # Random joint controls provide exploration when EB guidance gets stuck.
        # Unlike EB candidates, all agents use the same sampled duration here,
        # matching the vanilla cRRT joint expansion behavior.
        best_candidate = None
        best_score = np.inf
        active_agent_indices = self._active_agent_indices(parent_node)
        if len(active_agent_indices) == 0:
            return None

        for _ in range(self.num_extension_trials):
            accept_new_state = True
            # Parked agents should have zero controls, so branch-parking mode
            # starts the joint action at zero. Normal mode will fill every
            # action slice below.
            if self.branch_goal_parking:
                joint_action = np.zeros(self.joint_action_size)
            else:
                joint_action = np.empty(self.joint_action_size)
            new_joint_state = np.empty(self.joint_state_size)

            random_time = self.get_time()
            num_record_steps = max(
                1, round(random_time / self.minimum_time_step))
            random_time = num_record_steps * self.minimum_time_step
            joint_path = np.empty((num_record_steps, self.joint_state_size))

            # Simulate active agents with random controls. Parked agents are
            # copied forward as repeated states and still participate in the
            # joint collision check below.
            for agent_index in range(self.num_agents):
                if self._agent_is_parked(parent_node, agent_index):
                    self._fill_parked_agent(
                        joint_action, joint_path, new_joint_state,
                        parent_node, agent_index)
                    continue

                agent = self.agents[agent_index]
                parent_agent_state = self.get_agent_state(
                    parent_node.state, agent_index)
                agent_action = agent.get_random_action(self.rng)
                self.set_agent_action(joint_action, agent_index,
                                      agent_action)

                new_substate, path_to_new_state = agent.get_next_state(
                    parent_agent_state, agent_action,
                    random_time, num_record_steps)
                self.set_agent_state(new_joint_state, agent_index,
                                     new_substate)
                self.set_agent_path(joint_path, agent_index,
                                    path_to_new_state)

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

            # Reject random-control candidates that collide between agents,
            # including collisions with parked agents.
            if self.joint_path_collides(joint_path):
                continue

            # Keep the valid random-control candidate whose endpoint is closest
            # to the sampled joint random point.
            matrix_state = self.superstate_to_matrix_state(new_joint_state)
            score = euclidean_distance_numba_with_l(
                matrix_state, random_point, self.distance_metric_state_size)
            if score < best_score:
                best_score = score
                best_candidate = (new_joint_state, joint_path, joint_action,
                                  random_time, score)

        return best_candidate

    def _select_best_joint_extension_candidate(self, parent_node, random_point):
        # Epsilon-random exploration is tried before EB guidance. Otherwise,
        # run one EB-guided joint expansion; that call already performs the
        # bounded inter-agent repair attempts controlled by
        # max_joint_edge_trials. If EB cannot find a collision-free
        # combination, optionally fall back to random controls.
        if self.rng.random() < self.epsilon_random:
            random_candidate = self._select_random_joint_candidate(
                parent_node, random_point)
            if random_candidate is not None:
                return random_candidate[:-1]
            return None

        candidate = self._select_edge_bundle_joint_candidate(
            parent_node, random_point)
        if candidate is not None:
            return candidate[:-1]

        if self.fallback_to_random_control:
            random_candidate = self._select_random_joint_candidate(
                parent_node, random_point)
            if random_candidate is not None:
                return random_candidate[:-1]

        return None

    def extend_tree(self, parent_node_id, parent_node, random_point):
        # Select one valid joint rollout, then reuse CRRT's goal processing so
        # terminal path truncation and per-agent cost accounting stay consistent
        # with the base centralized planner.
        best_candidate = self._select_best_joint_extension_candidate(
            parent_node, random_point)
        if best_candidate is None:
            if self.debug_flag:
                print("Sampled Kino TI EB cRRT joint edge is invalid. "
                      "Trying again!")
            return

        new_joint_state, joint_path, joint_action, edge_time = best_candidate

        updated_path, new_joint_state, costs, reached_goals, edge_time = (
            self.check_for_goals(parent_node, edge_time, joint_action,
                                 joint_path, new_joint_state))

        total_elapsed_time = parent_node.time_elapsed + edge_time
        combined_cost = parent_node.cost_so_far + costs
        new_node_id = self.add_rrt_node(
            new_joint_state, parent_node_id, joint_action, edge_time,
            updated_path, total_elapsed_time, combined_cost, reached_goals)

        if np.any(reached_goals) and self.debug_flag:
            print("Agent found goal:", np.flatnonzero(reached_goals))
        self.goal_seen_by_agent |= reached_goals

        self.path_found = np.all(reached_goals)
        if self.path_found:
            self.goal_node_id = new_node_id
            self.path_cost = combined_cost
            self.path_time = total_elapsed_time

        return new_node_id
