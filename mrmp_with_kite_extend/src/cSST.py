import time

import numpy as np
from numba.typed import List
from numba import types

from cRRT import (
    _joint_state_to_matrix_state_numba,
    _joint_path_collides_2d_numba,
    _joint_path_collides_3d_numba,
    _first_joint_path_collision_2d_numba,
    _first_joint_path_collision_3d_numba,
)
from rrt import RRT
from sst import (
    SSTWitnessMatrix,
    get_best_or_nearest_active_index,
    get_nearest_active_index,
    get_nearest_witness_index,
    get_nearest_space_time_witness_index,
)
from utils import (
    find_roundoff_decimal_digits,
    get_dtype_from_input,
    euclidean_distance_numba_with_l,
)


class CSSTNodeMatrix:
    def __init__(self, initial_capacity, joint_state_dim, matrix_state_dim,
                 joint_action_dim, max_sub_path_length, num_agents):
        self.id = np.full(initial_capacity, -1, dtype=np.int32)
        self.state = np.zeros((initial_capacity, joint_state_dim), dtype=np.float64)
        self.matrix_state = np.zeros((initial_capacity, matrix_state_dim), dtype=np.float64)
        self.parent = np.full(initial_capacity, -1, dtype=np.int32)
        self.parent_action = np.zeros((initial_capacity, joint_action_dim), dtype=np.float64)
        self.action_duration = np.full(initial_capacity, -1, dtype=np.float32)
        self.path_from_parent = np.zeros(
            (initial_capacity, max_sub_path_length, joint_state_dim), dtype=np.float64)
        self.sub_path_length = np.full(initial_capacity, -1, dtype=np.int32)
        self.time_elapsed = np.full(initial_capacity, -1, dtype=np.float32)
        self.cost = np.full(initial_capacity, np.inf, dtype=np.float64)
        self.per_agent_cost = np.full((initial_capacity, num_agents), np.inf, dtype=np.float64)
        self.reached_goals = np.zeros((initial_capacity, num_agents), dtype=np.bool_)
        self.active = np.zeros(initial_capacity, dtype=np.uint8)
        self.count = 0
        self.joint_state_dim = joint_state_dim
        self.matrix_state_dim = matrix_state_dim
        self.joint_action_dim = joint_action_dim
        self.max_sub_path_length = max_sub_path_length
        self.num_agents = num_agents

    def append(self, node_id, state, matrix_state, parent, action, action_duration,
               path_from_parent, time_elapsed, scalar_cost, per_agent_cost,
               reached_goals):
        if self.count >= self.state.shape[0]:
            self._grow()
        curr_index = self.count
        self.id[curr_index] = node_id
        self.state[curr_index] = state
        self.matrix_state[curr_index] = matrix_state
        self.parent[curr_index] = parent
        self.parent_action[curr_index] = action
        self.action_duration[curr_index] = action_duration
        len_path_from_parent = path_from_parent.shape[0]
        self.path_from_parent[curr_index, :len_path_from_parent] = path_from_parent
        self.sub_path_length[curr_index] = len_path_from_parent
        self.time_elapsed[curr_index] = time_elapsed
        self.cost[curr_index] = scalar_cost
        self.per_agent_cost[curr_index] = per_agent_cost
        self.reached_goals[curr_index] = reached_goals
        self.active[curr_index] = 0
        self.count += 1
        return curr_index

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.joint_state_dim), dtype=np.float64)
        new_matrix_state = np.zeros((new_cap, self.matrix_state_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action = np.zeros((new_cap, self.joint_action_dim), dtype=np.float64)
        new_action_duration = np.full(new_cap, -1, dtype=np.float32)
        new_path_from_parent = np.zeros(
            (new_cap, self.max_sub_path_length, self.joint_state_dim), dtype=np.float64)
        new_sub_path_length = np.full(new_cap, -1, dtype=np.int32)
        new_time_elapsed = np.full(new_cap, -1, dtype=np.float32)
        new_cost = np.full(new_cap, np.inf, dtype=np.float64)
        new_per_agent_cost = np.full((new_cap, self.num_agents), np.inf, dtype=np.float64)
        new_reached_goals = np.zeros((new_cap, self.num_agents), dtype=np.bool_)
        new_active = np.zeros(new_cap, dtype=np.uint8)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_matrix_state[:old_cap] = self.matrix_state
        new_parent[:old_cap] = self.parent
        new_parent_action[:old_cap] = self.parent_action
        new_action_duration[:old_cap] = self.action_duration
        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length
        new_time_elapsed[:old_cap] = self.time_elapsed
        new_cost[:old_cap] = self.cost
        new_per_agent_cost[:old_cap] = self.per_agent_cost
        new_reached_goals[:old_cap] = self.reached_goals
        new_active[:old_cap] = self.active

        self.id = new_id
        self.state = new_state
        self.matrix_state = new_matrix_state
        self.parent = new_parent
        self.parent_action = new_parent_action
        self.action_duration = new_action_duration
        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length
        self.time_elapsed = new_time_elapsed
        self.cost = new_cost
        self.per_agent_cost = new_per_agent_cost
        self.reached_goals = new_reached_goals
        self.active = new_active

    def get_valid_states(self):
        return self.state[:self.count]

    def get_valid_ids(self):
        return self.id[:self.count]


class CSST(RRT):
    def __init__(self, *, agents, starts, goals, goal_radii, env,
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
                 random_point_function,
                 udf_seed=77,
                 goal_sampling_probability=0.1,
                 best_near_radius=1.0,
                 prune_radius=0.5,
                 mrmp_planning=True,
                 witness_time_radius=None,
                 objective="sum_of_costs",
                 dynamic_agent_clearance=0.0,
                 dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C')),
                 truncate_paths=False,
                 branch_goal_parking=True,
                 print_logs=False,
                 debug_flag=False):
        if branch_goal_parking and truncate_paths:
            raise ValueError("branch_goal_parking requires truncate_paths=False")
        if objective in ("sum_of_costs", "soc"):
            use_sum_of_costs = True
        elif objective == "makespan":
            use_sum_of_costs = False
        else:
            raise ValueError(
                "objective must be 'sum_of_costs', 'soc', or 'makespan'")

        RRT.__init__(
            self,
            start=starts[0], goal=goals[0], goal_radius=goal_radii[0],
            env=env, agent=agents[0],
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
            goal_sampling_probability=goal_sampling_probability,
            dynamic_agent_clearance=dynamic_agent_clearance,
            print_logs=print_logs,
            debug_flag=debug_flag,
            dynamic_obstacles=dynamic_obstacles)

        self.agents = agents
        self.starts = [np.asarray(start, dtype=np.float64) for start in starts]
        self.goals = [np.asarray(goal, dtype=np.float64) for goal in goals]
        self.goal_radii = goal_radii
        self.num_agents = len(self.agents)
        self.agent_id_to_index = {agent.id: index for index, agent in enumerate(self.agents)}

        self.agent_state_lengths = np.array([len(start) for start in self.starts], dtype=np.int32)
        self.agent_state_starts = np.empty(self.num_agents + 1, dtype=np.int32)
        self.agent_state_starts[0] = 0
        for agent_index in range(self.num_agents):
            self.agent_state_starts[agent_index + 1] = (
                self.agent_state_starts[agent_index] + self.agent_state_lengths[agent_index])
        self.joint_state_size = int(self.agent_state_starts[-1])
        self.start_joint_state = self.agent_states_to_joint_state(self.starts)

        self.agent_radii = np.array([agent.radius for agent in self.agents], dtype=np.float64)
        self.agent_action_lengths = np.array([agent.action_length for agent in self.agents], dtype=np.int32)
        # Reused when an agent is parked; set_agent_action copies these values
        # into the corresponding joint-action slice.
        self._zero_actions = [
            np.zeros(agent.action_length) for agent in self.agents
        ]
        self.agent_action_starts = np.empty(self.num_agents + 1, dtype=np.int32)
        self.agent_action_starts[0] = 0
        for agent_index in range(self.num_agents):
            self.agent_action_starts[agent_index + 1] = (
                self.agent_action_starts[agent_index] + self.agent_action_lengths[agent_index])
        self.joint_action_size = int(self.agent_action_starts[-1])

        self.isvalid = isvalid_function
        self.cost_funcs = cost_function
        self.reached_goal_funcs = reached_goal_function
        self.get_random_point_funcs = random_point_function
        self.branch_goal_parking = bool(branch_goal_parking)
        self.truncate_paths = bool(truncate_paths)
        self.use_sum_of_costs = bool(use_sum_of_costs)
        self.best_near_radius = best_near_radius
        self.prune_radius = prune_radius
        self.mrmp_planning = mrmp_planning
        self.goal_seen_by_agent = np.zeros(self.num_agents, dtype=np.bool_)
        self.path_scalar_cost = np.inf
        self.first_solution_node_id = None
        self.first_solution_planning_time = np.inf
        self.first_solution_path_time = 0.0
        self.first_solution_path_cost = np.inf
        self.first_solution_per_agent_cost = np.full(self.num_agents, np.inf, dtype=np.float64)

        self.agent_position_state_dim = len(env.size)
        if self.agent_position_state_dim not in (2, 3):
            raise ValueError("CSST only supports 2D and 3D state positions")
        self.distance_metric_state_size = self.agent_position_state_dim * self.num_agents
        self.goal_matrix_state = np.empty(self.distance_metric_state_size, dtype=np.float64)
        for agent_index in range(self.num_agents):
            position_start = agent_index * self.agent_position_state_dim
            position_end = position_start + self.agent_position_state_dim
            self.goal_matrix_state[position_start:position_end] = (
                self.goals[agent_index][:self.agent_position_state_dim])

        if truncate_paths:
            self.threshold = truncation_check_threshold * np.sqrt(self.num_agents)
        else:
            self.threshold = 0.0

        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        if witness_time_radius is None:
            witness_time_radius = sampling_time_step
        self.witness_time_radius = round(witness_time_radius, self.roundoff_digits)
        self.max_sub_path_length = int(round(
            self.max_sample_T / self.minimum_time_step, self.roundoff_digits))
        self.type_node_matrix = CSSTNodeMatrix
        self.type_witness_matrix = SSTWitnessMatrix
        self._node_matrix = self.type_node_matrix(
            initial_capacity=1024,
            joint_state_dim=self.joint_state_size,
            matrix_state_dim=self.distance_metric_state_size,
            joint_action_dim=self.joint_action_size,
            max_sub_path_length=self.max_sub_path_length,
            num_agents=self.num_agents)
        self._witness_matrix = self.type_witness_matrix(
            initial_capacity=1024,
            state_dim=self.distance_metric_state_size)

    def _agent_is_parked_index(self, parent_index, agent_index):
        return (self.branch_goal_parking and
                self._node_matrix.reached_goals[parent_index, agent_index])

    def _fill_parked_agent_index(self, joint_action, joint_path, new_joint_state,
                                 parent_index, agent_index):
        parent_state = self._node_matrix.state[parent_index]
        agent_state = self.get_agent_state(parent_state, agent_index)
        self.set_agent_action(
            joint_action,
            agent_index,
            self._zero_actions[agent_index])
        self.set_agent_state(new_joint_state, agent_index, agent_state)
        joint_path[:, self.get_agent_state_slice(agent_index)] = agent_state

    def get_agent_state_slice(self, agent_index):
        return slice(self.agent_state_starts[agent_index],
                     self.agent_state_starts[agent_index + 1])

    def get_agent_state(self, joint_state, agent_index):
        start = self.agent_state_starts[agent_index]
        end = self.agent_state_starts[agent_index + 1]
        return joint_state[start:end]

    def set_agent_state(self, joint_state, agent_index, agent_state):
        start = self.agent_state_starts[agent_index]
        end = self.agent_state_starts[agent_index + 1]
        joint_state[start:end] = agent_state

    def agent_states_to_joint_state(self, agent_states):
        joint_state = np.empty(self.joint_state_size, dtype=np.float64)
        for agent_index in range(self.num_agents):
            self.set_agent_state(joint_state, agent_index, agent_states[agent_index])
        return joint_state

    def joint_state_to_agent_states(self, joint_state):
        agent_states = []
        for agent_index in range(self.num_agents):
            agent_states.append(np.copy(self.get_agent_state(joint_state, agent_index)))
        return agent_states

    def get_agent_path(self, joint_path, agent_index):
        return joint_path[:, self.get_agent_state_slice(agent_index)]

    def set_agent_path(self, joint_path, agent_index, agent_path):
        joint_path[:, self.get_agent_state_slice(agent_index)] = agent_path

    def joint_path_collides(self, joint_path, start_index=0):
        if self.agent_position_state_dim == 2:
            return _joint_path_collides_2d_numba(
                joint_path, self.agent_state_starts, self.agent_radii,
                self.dynamic_agent_clearance, start_index)
        return _joint_path_collides_3d_numba(
            joint_path, self.agent_state_starts, self.agent_radii,
            self.dynamic_agent_clearance, start_index)

    def first_joint_path_collision(self, joint_path, start_index=0):
        if self.agent_position_state_dim == 2:
            return _first_joint_path_collision_2d_numba(
                joint_path, self.agent_state_starts, self.agent_radii,
                self.dynamic_agent_clearance, start_index)
        return _first_joint_path_collision_3d_numba(
            joint_path, self.agent_state_starts, self.agent_radii,
            self.dynamic_agent_clearance, start_index)

    def get_agent_action_slice(self, agent_index):
        return slice(self.agent_action_starts[agent_index],
                     self.agent_action_starts[agent_index + 1])

    def get_agent_action(self, joint_action, agent_index):
        start = self.agent_action_starts[agent_index]
        end = self.agent_action_starts[agent_index + 1]
        return joint_action[start:end]

    def set_agent_action(self, joint_action, agent_index, agent_action):
        start = self.agent_action_starts[agent_index]
        end = self.agent_action_starts[agent_index + 1]
        joint_action[start:end] = agent_action

    def get_agent_id_order(self):
        agent_id_order = np.empty(self.num_agents, dtype=np.int32)
        for index, agent in enumerate(self.agents):
            agent_id_order[index] = agent.id
        return agent_id_order

    def superstate_to_matrix_state(self, superstate):
        return _joint_state_to_matrix_state_numba(
            superstate, self.agent_state_starts, self.num_agents,
            self.agent_position_state_dim, self.distance_metric_state_size)

    def _scalar_cost(self, per_agent_cost, time_elapsed):
        if self.use_sum_of_costs:
            return float(np.sum(per_agent_cost))
        return float(time_elapsed)

    def add_csst_node(self, state, parent_index, parent_action, parent_action_duration,
                      path_from_parent, time_elapsed, scalar_cost, per_agent_cost,
                      reached_goals):
        new_node_id = self.last_added_node_id + 1
        matrix_state = self.superstate_to_matrix_state(state)
        added_index = self._node_matrix.append(
            new_node_id,
            state,
            matrix_state,
            parent_index,
            parent_action,
            parent_action_duration,
            path_from_parent,
            round(time_elapsed, self.roundoff_digits),
            scalar_cost,
            per_agent_cost,
            reached_goals)
        self.last_added_node_id = new_node_id
        return added_index

    def add_csst_witness(self, matrix_state, time_elapsed, rep_index):
        new_witness_id = self.last_added_witness_id + 1
        added_index = self._witness_matrix.append(
            new_witness_id,
            matrix_state,
            round(time_elapsed, self.roundoff_digits),
            rep_index)
        self.last_added_witness_id = new_witness_id
        return added_index

    def get_nearest_witness(self, matrix_state, time_elapsed):
        witnesses = self._witness_matrix
        if self.mrmp_planning:
            return get_nearest_space_time_witness_index(
                witnesses.state, witnesses.time_elapsed, witnesses.count,
                matrix_state, time_elapsed, self.distance_metric_state_size,
                self.witness_time_radius)
        return get_nearest_witness_index(
            witnesses.state, witnesses.count, matrix_state,
            self.distance_metric_state_size)

    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree is None:
            self._node_matrix = self.type_node_matrix(
                initial_capacity=1024,
                joint_state_dim=self.joint_state_size,
                matrix_state_dim=self.distance_metric_state_size,
                joint_action_dim=self.joint_action_size,
                max_sub_path_length=self.max_sub_path_length,
                num_agents=self.num_agents)
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

    def get_tree_structure(self):
        return self._node_matrix, self._witness_matrix

    def sample_random_point(self):
        random_state = np.empty(self.distance_metric_state_size, dtype=np.float64)
        position_dim = self.agent_position_state_dim
        for agent_index in range(self.num_agents):
            position_start = agent_index * position_dim
            position_end = position_start + position_dim
            r = self.rng.uniform(0, 1)
            if (r < self.goal_sampling_probability or
                    (r < 0.7 and self.goal_seen_by_agent[agent_index])):
                random_state[position_start:position_end] = (
                    self.goal_matrix_state[position_start:position_end])
            else:
                random_state[position_start:position_end] = self.get_random_point_funcs[agent_index](
                    self.env,
                    self.static_circular_obstacles,
                    self.static_rectangular_obstacles,
                    self.rng)
        return random_state

    def check_for_goals(self, parent_index, random_time, joint_action,
                        joint_path, new_joint_state):
        nodes = self._node_matrix
        if self.branch_goal_parking:
            reached_goal = nodes.reached_goals[parent_index].copy()
        else:
            reached_goal = np.zeros(self.num_agents, dtype=np.bool_)
        costs = np.zeros(self.num_agents, dtype=np.float64)

        parent_joint_state = nodes.state[parent_index]
        for agent_index in range(self.num_agents):
            if self._agent_is_parked_index(parent_index, agent_index):
                continue

            agent = self.agents[agent_index]
            agent_path = self.get_agent_path(joint_path, agent_index)
            parent_agent_state = self.get_agent_state(parent_joint_state, agent_index)
            agent_action = self.get_agent_action(joint_action, agent_index)
            agent_state = self.get_agent_state(new_joint_state, agent_index)

            reached_goal_flag, _ = self.reached_goal_funcs[agent_index](
                agent_state, self.goals[agent_index], self.goal_radii[agent_index], agent)
            reached_goal[agent_index] = reached_goal_flag
            costs[agent_index] = self.cost_funcs[agent_index](
                self.env, agent, parent_agent_state, agent_action,
                random_time, agent_path)

        if self.branch_goal_parking:
            return joint_path, new_joint_state, costs, reached_goal, random_time

        if np.all(reached_goal):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        endpoint_matrix_state = self.superstate_to_matrix_state(new_joint_state)
        joint_goal_distance = euclidean_distance_numba_with_l(
            endpoint_matrix_state, self.goal_matrix_state,
            self.distance_metric_state_size)

        if self.threshold <= 0.0 or joint_goal_distance >= self.threshold:
            return joint_path, new_joint_state, costs, reached_goal, random_time

        first_goal_indices = np.full(self.num_agents, -1, dtype=np.int32)
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

        if np.any(first_goal_indices < 0):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        joint_goal_index = int(np.max(first_goal_indices))
        first_collision_check_index = int(np.min(first_goal_indices))
        terminal_joint_path = np.empty(
            (joint_goal_index + 1, self.joint_state_size), dtype=joint_path.dtype)
        terminal_joint_path[:, :] = joint_path[:joint_goal_index + 1]

        for agent_index in range(self.num_agents):
            first_goal_index = int(first_goal_indices[agent_index])
            if first_goal_index < joint_goal_index:
                agent_slice = self.get_agent_state_slice(agent_index)
                first_goal_state = joint_path[first_goal_index, agent_slice]
                terminal_joint_path[first_goal_index + 1:, agent_slice] = first_goal_state

        if self.joint_path_collides(
                terminal_joint_path, start_index=first_collision_check_index):
            return joint_path, new_joint_state, costs, reached_goal, random_time

        modified_edge_time = (joint_goal_index + 1) * self.minimum_time_step
        terminal_costs = np.zeros(self.num_agents, dtype=np.float64)
        terminal_joint_state = terminal_joint_path[-1].copy()
        for agent_index in range(self.num_agents):
            agent = self.agents[agent_index]
            parent_agent_state = self.get_agent_state(parent_joint_state, agent_index)
            agent_action = self.get_agent_action(joint_action, agent_index)
            terminal_costs[agent_index] = self.cost_funcs[agent_index](
                self.env, agent, parent_agent_state, agent_action,
                modified_edge_time,
                self.get_agent_path(terminal_joint_path, agent_index))

        reached_goal[:] = True
        return terminal_joint_path, terminal_joint_state, terminal_costs, reached_goal, modified_edge_time

    def _select_best_joint_extension_candidate(self, parent_index, random_point):
        nodes = self._node_matrix
        parent_joint_state = nodes.state[parent_index]
        parent_time_elapsed = nodes.time_elapsed[parent_index]
        best_candidate = None
        best_score = np.inf

        for _ in range(self.num_extension_trials):
            accept_new_state = True
            joint_action = np.empty(self.joint_action_size, dtype=np.float64)
            new_joint_state = np.empty(self.joint_state_size, dtype=np.float64)

            random_time = self.get_time()
            num_record_steps = max(1, round(random_time / self.minimum_time_step))
            random_time = num_record_steps * self.minimum_time_step
            joint_path = np.empty((num_record_steps, self.joint_state_size), dtype=np.float64)

            for agent_index in range(self.num_agents):
                if self._agent_is_parked_index(parent_index, agent_index):
                    self._fill_parked_agent_index(
                        joint_action, joint_path, new_joint_state,
                        parent_index, agent_index)
                    continue

                agent = self.agents[agent_index]
                parent_agent_state = self.get_agent_state(parent_joint_state, agent_index)
                agent_action = agent.get_random_action(self.rng)
                self.set_agent_action(joint_action, agent_index, agent_action)

                new_substate, path_to_new_state = agent.get_next_state(
                    parent_agent_state, agent_action, random_time, num_record_steps)
                self.set_agent_state(new_joint_state, agent_index, new_substate)
                self.set_agent_path(joint_path, agent_index, path_to_new_state)

                accept_new_state = accept_new_state and self.isvalid[agent_index](
                    path_to_new_state,
                    agent.radius,
                    self.env.size,
                    self.static_circular_obstacles,
                    self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles,
                    agent.dynamic_limit_indices,
                    agent.dynamic_limit_values,
                    self.env.obstacle_buffer,
                    self.dynamic_agent_clearance,
                    self.env.boundary_buffer,
                    parent_time_elapsed,
                    random_time,
                    self.minimum_time_step)
                if not accept_new_state:
                    break

            if not accept_new_state:
                continue
            if self.joint_path_collides(joint_path):
                continue

            matrix_state = self.superstate_to_matrix_state(new_joint_state)
            score = euclidean_distance_numba_with_l(
                matrix_state, random_point, self.distance_metric_state_size)
            if score < best_score:
                best_score = score
                best_candidate = (new_joint_state, matrix_state, joint_path,
                                  joint_action, random_time)

        return best_candidate

    def extend_tree(self, parent_index, random_point):
        candidate = self._select_best_joint_extension_candidate(parent_index, random_point)
        if candidate is None:
            return None

        new_joint_state, matrix_state, joint_path, joint_action, random_time = candidate
        updated_path, new_joint_state, edge_costs, reached_goals, edge_time = self.check_for_goals(
            parent_index, random_time, joint_action, joint_path, new_joint_state)
        if updated_path is not joint_path:
            matrix_state = self.superstate_to_matrix_state(new_joint_state)

        nodes = self._node_matrix
        total_elapsed_time = nodes.time_elapsed[parent_index] + edge_time
        total_per_agent_cost = nodes.per_agent_cost[parent_index] + edge_costs
        scalar_cost = self._scalar_cost(total_per_agent_cost, total_elapsed_time)
        reached_goal_flag = bool(np.all(reached_goals))

        return (new_joint_state, matrix_state, updated_path, joint_action, edge_time,
                total_elapsed_time, scalar_cost, total_per_agent_cost,
                reached_goals, reached_goal_flag)

    def _accept_extension_candidate(self, parent_index, extension_candidate,
                                    planning_start_time):
        (new_joint_state, matrix_state, joint_path, joint_action, edge_time,
         total_elapsed_time, scalar_cost, total_per_agent_cost,
         reached_goals, reached_goal_flag) = extension_candidate

        nodes = self._node_matrix
        nearest_witness_index, nearest_witness_distance = self.get_nearest_witness(
            matrix_state, total_elapsed_time)

        node_accepted = False
        new_node_index = -1
        if nearest_witness_index == -1 or nearest_witness_distance > self.prune_radius:
            new_node_index = self.add_csst_node(
                new_joint_state, parent_index, joint_action, edge_time,
                joint_path, total_elapsed_time, scalar_cost,
                total_per_agent_cost, reached_goals)
            self.add_csst_witness(matrix_state, total_elapsed_time, new_node_index)
            nodes.active[new_node_index] = 1
            node_accepted = True
        else:
            curr_rep_index = self._witness_matrix.rep_index[nearest_witness_index]
            if scalar_cost < nodes.cost[curr_rep_index]:
                new_node_index = self.add_csst_node(
                    new_joint_state, parent_index, joint_action, edge_time,
                    joint_path, total_elapsed_time, scalar_cost,
                    total_per_agent_cost, reached_goals)
                nodes.active[curr_rep_index] = 0
                self._witness_matrix.rep_index[nearest_witness_index] = new_node_index
                nodes.active[new_node_index] = 1
                node_accepted = True

        if not node_accepted:
            return -1

        self.goal_seen_by_agent |= reached_goals
        if reached_goal_flag:
            if self.first_solution_node_id is None:
                self.first_solution_node_id = new_node_index
                self.first_solution_planning_time = time.time() - planning_start_time
                self.first_solution_path_time = nodes.time_elapsed[new_node_index]
                self.first_solution_path_cost = nodes.cost[new_node_index]
                self.first_solution_per_agent_cost = nodes.per_agent_cost[new_node_index].copy()
            if nodes.cost[new_node_index] < self.path_scalar_cost:
                self.path_found = True
                self.goal_node_id = new_node_index
                self.path_scalar_cost = nodes.cost[new_node_index]
                self.path_cost = nodes.per_agent_cost[new_node_index].copy()
                self.path_time = nodes.time_elapsed[new_node_index]
            nodes.active[new_node_index] = 0
        return new_node_index

    def _run_search_loop(self, planning_start_time):
        curr_num_steps = 0
        while curr_num_steps <= self.max_iter:
            random_point = self.sample_random_point()
            nodes = self._node_matrix
            nearest_node_index = get_best_or_nearest_active_index(
                nodes.matrix_state, nodes.cost, nodes.active, nodes.count,
                random_point, self.best_near_radius,
                self.distance_metric_state_size)
            if nearest_node_index == -1:
                nearest_node_index = get_nearest_active_index(
                    nodes.matrix_state, nodes.active, nodes.count,
                    random_point, self.distance_metric_state_size)
            if nearest_node_index != -1:
                extension_candidate = self.extend_tree(nearest_node_index, random_point)
                if extension_candidate is not None:
                    self._accept_extension_candidate(
                        nearest_node_index, extension_candidate, planning_start_time)

            curr_num_steps += 1
            if time.time() - planning_start_time >= self.planning_time:
                break
        return curr_num_steps

    def _initialize_root(self):
        empty_path_from_parent = np.empty((0, self.joint_state_size), dtype=np.float64)
        root_index = self.add_csst_node(
            self.start_joint_state,
            -1,
            np.zeros(self.joint_action_size, dtype=np.float64),
            0.0,
            empty_path_from_parent,
            0.0,
            0.0,
            np.zeros(self.num_agents, dtype=np.float64),
            np.zeros(self.num_agents, dtype=np.bool_))
        self._node_matrix.active[root_index] = 1
        self.add_csst_witness(self._node_matrix.matrix_state[root_index], 0.0, root_index)

    def plan_path(self):
        self.reset_tree()
        self._initialize_root()
        start_time = time.time()
        curr_num_steps = self._run_search_loop(start_time)
        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        if self.first_solution_node_id is not None:
            self.first_solution_path_time = round(
                self.first_solution_path_time, self.roundoff_digits)
        if self.print_logs or self.debug_flag:
            print("Total Planning Time after", curr_num_steps, "steps:", total_time)
        return total_time

    def replan_path(self):
        # Continue the anytime SST search and keep any incumbent solution.
        if self._node_matrix.count == 0:
            self._initialize_root()
        start_time = time.time()
        curr_num_steps = self._run_search_loop(start_time)
        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        if self.print_logs or self.debug_flag:
            print("Total Replanning Time after", curr_num_steps, "steps:", total_time)
        return total_time

    def get_costs(self):
        if not self.path_found:
            print("Costs can't be found because goal hasn't been reached!")
            return np.array([np.inf for _ in range(self.num_agents)])
        return self.path_cost

    def get_agent_path_times(self, high_resolution_path_numpy_array=None):
        if high_resolution_path_numpy_array is None:
            high_resolution_path_numpy_array = self.get_high_resolution_path_numpy_array()
        path_times = []
        for agent_index in range(self.num_agents):
            agent_path = high_resolution_path_numpy_array[agent_index]
            num_timesteps = agent_path.shape[0]
            last_same = num_timesteps - 1
            while last_same > 0 and np.allclose(agent_path[last_same], agent_path[last_same - 1]):
                last_same -= 1
            total_time = last_same * self.minimum_time_step
            path_times.append(round(total_time, self.roundoff_digits))
        return path_times

    def get_high_resolution_path_numpy_array(self):
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return np.empty((0, self.distance_metric_state_size), dtype=np.float64)

        nodes = self._node_matrix
        max_nodes = nodes.count
        path_node_indices = np.empty(max_nodes, dtype=np.int32)
        path_num_nodes = 0
        num_path_rows = 1

        node_index = self.goal_node_id
        while node_index != -1:
            path_node_indices[path_num_nodes] = node_index
            if nodes.parent[node_index] != -1:
                num_path_rows += nodes.sub_path_length[node_index]
            path_num_nodes += 1
            node_index = nodes.parent[node_index]

        path_states = []
        for agent_index in range(self.num_agents):
            path_states.append(np.empty(
                (num_path_rows, self.agent_state_lengths[agent_index]),
                dtype=np.float64))

        indices = path_node_indices[:path_num_nodes][::-1]
        start_index = indices[0]
        for agent_index in range(self.num_agents):
            path_states[agent_index][0] = self.get_agent_state(
                nodes.state[start_index], agent_index)

        curr_index = 1
        for node_index in indices[1:]:
            len_path_to_node = nodes.sub_path_length[node_index]
            next_index = curr_index + len_path_to_node
            joint_path = nodes.path_from_parent[node_index, :len_path_to_node]
            for agent_index in range(self.num_agents):
                path_states[agent_index][curr_index:next_index] = (
                    self.get_agent_path(joint_path, agent_index))
            curr_index = next_index

        return path_states

    def get_high_resolution_paths(self):
        high_res_dicts = []
        for _ in range(self.num_agents):
            high_res_dicts.append({})
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return high_res_dicts

        nodes = self._node_matrix
        path_node_indices = np.empty(nodes.count, dtype=np.int32)
        path_length = 0
        node_index = self.goal_node_id
        while node_index != -1:
            path_node_indices[path_length] = node_index
            path_length += 1
            node_index = nodes.parent[node_index]

        indices = path_node_indices[:path_length][::-1]
        start_index = indices[0]
        path_time = np.zeros(self.num_agents, dtype=np.float64)
        for agent_index in range(self.num_agents):
            high_res_dicts[agent_index][0.0] = self.get_agent_state(
                nodes.state[start_index], agent_index).copy()

        for node_index in indices[1:]:
            len_path_to_node = nodes.sub_path_length[node_index]
            joint_path = nodes.path_from_parent[node_index, :len_path_to_node]
            for agent_index in range(self.num_agents):
                agent_path = self.get_agent_path(joint_path, agent_index)
                for point_index in range(agent_path.shape[0]):
                    path_time[agent_index] = round(
                        path_time[agent_index] + self.minimum_time_step,
                        self.roundoff_digits)
                    high_res_dicts[agent_index][float(path_time[agent_index])] = (
                        agent_path[point_index].copy())
        return high_res_dicts

    def get_path(self):
        if not self.path_found:
            return [], [], [], [], []

        nodes = self._node_matrix
        max_nodes = nodes.count
        path_node_indices = np.empty(max_nodes, dtype=np.int32)
        path_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        path_states = []
        path_controls = []
        path_costs = []
        for agent_index in range(self.num_agents):
            state_data_type = get_dtype_from_input(self.starts[agent_index])
            path_states.append(np.empty((max_nodes, len(state_data_type)), dtype=np.float64))
            path_controls.append(np.empty((max_nodes, self.agent_action_lengths[agent_index]), dtype=np.float64))
            path_costs.append(np.empty(max_nodes, dtype=np.float64))

        node_index = self.goal_node_id
        path_length = 0
        while node_index != -1:
            path_node_indices[path_length] = node_index
            path_node_ids[path_length] = nodes.id[node_index]
            path_timesteps[path_length] = nodes.action_duration[node_index]
            for agent_index in range(self.num_agents):
                path_states[agent_index][path_length] = self.get_agent_state(
                    nodes.state[node_index], agent_index)
                path_controls[agent_index][path_length] = self.get_agent_action(
                    nodes.parent_action[node_index], agent_index)
                path_costs[agent_index][path_length] = nodes.per_agent_cost[node_index, agent_index]
            path_length += 1
            node_index = nodes.parent[node_index]

        ids = path_node_ids[:path_length][::-1]
        timesteps = path_timesteps[:path_length - 1][::-1]
        states = []
        controls = []
        costs = []
        for agent_index in range(self.num_agents):
            states.append(path_states[agent_index][:path_length][::-1])
            controls.append(path_controls[agent_index][:path_length - 1][::-1])
            costs.append(path_costs[agent_index][:path_length - 1][::-1])
        return ids, states, controls, timesteps, costs
