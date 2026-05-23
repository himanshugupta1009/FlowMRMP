"""
Discontinuity-bounded RRT (Db-RRT) using precomputed motion primitives.

This implements the search component described in:
    iDb-RRT: Sampling-based Kinodynamic Motion Planning with Motion
    Primitives and Trajectory Optimization (arXiv:2403.10745)

This file intentionally covers only the search phase. The trajectory
optimization repair step from iDb-RRT is not implemented here.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numba import njit, types
from numba.typed import List

from rrt import RRT
from utils import euclidean_distance_numba_with_l, check_dynamic_collisions_to_end, \
    check_dynamic_collisions_to_end_3d, has_nearby_state, normalized_state_distance, \
    normalized_state_distance_sq


class DbRRTTreeNode:
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
        self.candidate_edge_ids = None
        self.candidate_edge_mask = None


class DbRRTPlanner(RRT):
    """
    RRT with motion primitives and bounded discontinuity.

    Relative to Db-A*, the main change is the search policy:
        - sample a target
        - pick the nearest tree node
        - expand with one primitive
        - add the new node only if it is not too close to existing nodes
    """

    def __init__(self, *,
                 start,
                 goal,
                 goal_radius,
                 env,
                 agent,
                 motion_primitives,
                 alpha: float = 0.5,
                 delta: float = 0.5,
                 minimum_time_step: float = 0.1,
                 max_iter: int = 10000,
                 planning_time: float = 10.0,
                 isvalid_function: Callable,
                 cost_function: Callable,
                 reached_goal_function: Callable,
                 random_point_function: Callable,
                 translate_function: Callable,
                 sort_edges_function: Callable,
                 transform_trajectory_function: Callable,
                 motion_primitive_kd_tree=None,
                 get_motion_primitive_kd_tree_query: Optional[Callable] = None,
                 max_candidate_motions_per_expand: int = 1000,
                 max_cost: float = math.inf,
                 allow_intermediate_goal: bool = True,
                 num_intermediate_goal_checks: int = 4,
                 cost_delta_factor: float = 0.0,
                 goal_bias: float = 0.1,
                 goal_sampling_probability: Optional[float] = None,
                 goal_expand_mode: str = "focused",
                 random_expand_mode: str = "randomized",
                 udf_seed: int = 77,
                 debug_flag: bool = False,
                 print_logs: bool = False,
                 dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C'))):

        if goal_sampling_probability is None:
            goal_sampling_probability = goal_bias

        super().__init__(
            start=start,
            goal=goal,
            goal_radius=goal_radius,
            env=env,
            agent=agent,
            use_fixed_sampling_time=True,
            sampling_time_step=1.0,
            minimum_time_step=minimum_time_step,
            max_iter=max_iter,
            planning_time=planning_time,
            isvalid_function=isvalid_function,
            cost_function=cost_function,
            reached_goal_function=reached_goal_function,
            random_point_function=random_point_function,
            udf_seed=udf_seed,
            goal_sampling_probability=goal_sampling_probability,
            debug_flag=debug_flag,
            print_logs=print_logs,
            dynamic_obstacles=dynamic_obstacles,
        )

        if not (0.0 <= goal_sampling_probability <= 1.0):
            raise ValueError("goal_sampling_probability must be between 0 and 1")
        if goal_expand_mode not in ("focused", "randomized"):
            raise ValueError("goal_expand_mode must be 'focused' or 'randomized'")
        if random_expand_mode not in ("focused", "randomized"):
            raise ValueError("random_expand_mode must be 'focused' or 'randomized'")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1")
        if delta <= 0.0:
            raise ValueError("delta must be positive")

        self.motion_primitives = motion_primitives
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.connect_radius = self.alpha * self.delta
        self.merge_radius = self.delta
        self.motion_primitive_kd_tree = motion_primitive_kd_tree
        self.get_motion_primitive_kd_tree_query = get_motion_primitive_kd_tree_query
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        self.transform_trajectory_function = transform_trajectory_function
        self.total_motion_primitives = int(motion_primitives.num_edges) #Total number of edges in the set! - doesn't change!
        #DO NOT change self.max_motions without thinking!
        #Why? Because making it less than self.total_motion_primitives will cause allocations in _get_candidate_edges.
        self.max_motions = self.total_motion_primitives
        if max_candidate_motions_per_expand is None:
            self.max_candidate_motions_per_expand = self.max_motions
        else:
            self.max_candidate_motions_per_expand = int(max_candidate_motions_per_expand)
        self.distance_array = np.zeros((self.max_candidate_motions_per_expand,), dtype=np.float64)
        self.max_cost = max_cost
        self.allow_intermediate_goal = allow_intermediate_goal
        self.num_intermediate_goal_checks = num_intermediate_goal_checks
        self.cost_delta_factor = float(cost_delta_factor)
        self.goal_bias = float(goal_sampling_probability)
        self.goal_expand_focused = (goal_expand_mode == "focused") #Basically, True
        self.random_expand_focused = (random_expand_mode == "focused") #Basically, False
        self.node_class = DbRRTTreeNode
        self.metric_indices = self.agent.distance_indices
        self.metric_scales = self.agent.distance_scales
        self.metric_is_angle = self.agent.distance_is_angle

        if self.distance_metric_state_size == 2:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end
        elif self.distance_metric_state_size == 3:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end_3d
        else:
            self.dynamic_col_checker_to_end = None

    def reset_for_iteration(self, *, delta: Optional[float] = None,
                            max_motions: Optional[int] = None,
                            max_cost: Optional[float] = None):
        if delta is not None:
            delta = float(delta)
            if delta <= 0.0:
                raise ValueError("delta must be positive")
            self.delta = delta
            self.connect_radius = self.alpha * self.delta
            self.merge_radius = self.delta

        if max_motions is not None:
            max_motions = int(max_motions)
            if max_motions <= 0:
                raise ValueError("max_motions must be positive")
            self.max_motions = min(max_motions, self.total_motion_primitives)

        if max_cost is not None:
            self.max_cost = float(max_cost)

        self.reset_tree()

    def sample_random_point(self):
        if self.rng.uniform(0.0, 1.0) < self.goal_bias:
            return np.asarray(self.goal, dtype=np.float64), True
        random_point = self.get_random_point(
            self.env,
            self.static_circular_obstacles,
            self.static_rectangular_obstacles,
            self.rng,
        )
        return np.asarray(random_point, dtype=np.float64), False

    def _edge_timestep(self, edge_id: int) -> float:
        return self.motion_primitives.timesteps[edge_id]

    def _edge_actions(self, edge_id: int):
        return self.motion_primitives.actions[edge_id]

    def _edge_start(self, edge_id: int) -> np.ndarray:
        return self.motion_primitives.start_states[edge_id]

    def _edge_final(self, edge_id: int) -> np.ndarray:
        length = int(self.motion_primitives.trajectory_lengths[edge_id])
        return self.motion_primitives.trajectories[edge_id, length - 1]

    def _edge_trajectory(self, edge_id: int) -> np.ndarray:
        length = int(self.motion_primitives.trajectory_lengths[edge_id])
        return self.motion_primitives.trajectories[edge_id, :length]

    def transform_primitive(self, base_state: np.ndarray, edge_id: int) -> Tuple[np.ndarray, np.ndarray]:
        edge_start = self._edge_start(edge_id)
        raw_traj = self._edge_trajectory(edge_id)
        transformed = self.transform_trajectory_function(base_state, edge_start, raw_traj)

        if transformed.shape[0] > 1:
            d0 = euclidean_distance_numba_with_l(transformed[0], base_state, self.distance_metric_state_size)
            if d0 <= 1e-9:
                path = transformed[1:]
            else:
                path = transformed
        else:
            path = transformed

        new_state = path[-1]
        return new_state, path

    def _state_distance(self, a: np.ndarray, b: np.ndarray) -> float:
            return normalized_state_distance(a,b,self.metric_indices,
                self.metric_scales,self.metric_is_angle)

    def lower_bound_time(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        speed = getattr(self.agent, "max_speed", 1.0)
        # speed = max(float(speed), 1e-9)
        return self._state_distance(state_a, state_b) / speed

    def _goal_state_is_safe(self, state: np.ndarray, arrival_time: float) -> bool:
        if self.dynamic_col_checker_to_end is None:
            return True
        if len(self.dynamic_agent_obstacles) == 0:
            return True
        return not self.dynamic_col_checker_to_end(
            state,
            self.agent.radius,
            self.dynamic_agent_obstacles,
            self.dynamic_agent_clearance,
            arrival_time,
            self.minimum_time_step,
        )

    def _get_candidate_edges(self, parent_node) -> Tuple[np.ndarray, np.ndarray]:
        if parent_node.candidate_edge_ids is None:
            query = self.get_motion_primitive_kd_tree_query(parent_node.state)
            edge_ids = self.motion_primitive_kd_tree.radius_query(query, self.connect_radius)
            if self.max_motions < self.total_motion_primitives:
                edge_ids = edge_ids[edge_ids < self.max_motions]
            max_edges = min(len(edge_ids), self.max_candidate_motions_per_expand)
            parent_node.candidate_edge_ids = np.asarray(edge_ids[:max_edges], dtype=np.int64)
            parent_node.candidate_edge_mask = np.full((max_edges,), False, dtype=np.bool_)
        return parent_node.candidate_edge_ids, parent_node.candidate_edge_mask

    def _is_path_valid(self, path: np.ndarray, start_time: float, duration: float) -> bool:
        return self.isvalid(path, self.agent.radius, self.env.size,
                            self.static_circular_obstacles,
                            self.static_rectangular_obstacles,
                            self.dynamic_agent_obstacles,
                            self.agent.dynamic_limit_indices,
                            self.agent.dynamic_limit_values,
                            self.env.obstacle_buffer,
                            self.dynamic_agent_clearance,
                            self.env.boundary_buffer,
                            start_time,
                            duration,
                            self.minimum_time_step)

    def _try_goal_along_path(self, path: np.ndarray) -> Optional[int]:
        if not self.allow_intermediate_goal or path.shape[0] == 0:
            return None
        reached, _ = self.reached_goal(path[-1], self.goal, self.goal_radius, self.agent)
        if reached:
            return path.shape[0] - 1
        if self.num_intermediate_goal_checks <= 0:
            return None
        for nn in range(self.num_intermediate_goal_checks):
            idx = int(float(nn + 1) / (self.num_intermediate_goal_checks + 1) * path.shape[0])
            idx = min(max(idx, 0), path.shape[0] - 1)
            reached, _ = self.reached_goal(path[idx], self.goal, self.goal_radius, self.agent)
            if reached:
                return idx
        return None

    def _new_state_is_far_enough(self, state: np.ndarray) -> bool:
        if self._node_matrix.count == 0:
            return True
        states = self._node_matrix.get_valid_matrix()
        return not has_nearby_state(
            states,
            self._node_matrix.count,
            state,
            self.distance_metric_state_size,
            self.delta,
        )

    def _iter_candidate_edge_slots(self,
                                   parent_node,
                                   target_state: np.ndarray,
                                   is_goal_mode: bool) -> np.ndarray:
        
        edge_ids, edge_mask = self._get_candidate_edges(parent_node)
        if edge_ids.size == 0:
            return np.empty(0, dtype=np.int64)

        #Code in case you want to switch between different expansion modes: focused and randomized.
        #Focused mode sorts the edges based on distance to target, while randomized mode shuffles them.

        # use_focused_mode = self.goal_expand_focused if is_goal_mode else self.random_expand_focused
        # if not use_focused_mode:
        #     return self.rng.permutation(available_slots)
        # available_slots = np.flatnonzero(~edge_mask)
        # if available_slots.size == 0:
        #     return available_slots

        #Forcing focused mode always to be similar to Kite_Extend!
        use_focused_mode = True

        sorted_indices, num_valid_edges = self.sort_edges(
            parent_node.state,
            target_state,
            self.motion_primitives.start_states,
            self.motion_primitives.final_states,
            edge_ids,
            edge_mask,
            self.distance_array,
        )
        return sorted_indices[:num_valid_edges]

    def _expand_once(self, parent_node_id, parent_node, target_state, is_goal_mode: bool):
        candidate_slots = self._iter_candidate_edge_slots(parent_node, target_state, is_goal_mode)
        candidate_edge_ids, candidate_edge_mask = self._get_candidate_edges(parent_node)

        if self.debug_flag:
            print(
                f"Db-RRT: expand node={parent_node_id}, "
                f"mode={int(is_goal_mode)}, candidates={len(candidate_slots)}"
            )

        for slot in candidate_slots:
            edge_id = int(candidate_edge_ids[slot])
            timestep = self._edge_timestep(edge_id)
            new_state, path = self.transform_primitive(parent_node.state, edge_id)

            if path.shape[0] == 0:
                continue

            goal_idx = self._try_goal_along_path(path)
            if goal_idx is not None:
                path_to_check = path[:goal_idx + 1]
                candidate_state = path_to_check[-1]
                candidate_dt = round(
                    (goal_idx + 1) * self.minimum_time_step,
                    self.roundoff_digits,
                )
            else:
                path_to_check = path
                candidate_state = new_state
                candidate_dt = timestep

            if not self._is_path_valid(path_to_check, parent_node.time_elapsed, candidate_dt):
                candidate_edge_mask[slot] = True
                continue

            action_sequence = self._edge_actions(edge_id)
            if hasattr(action_sequence, "ndim") and action_sequence.ndim > 1:
                num_steps = int(round(candidate_dt / self.minimum_time_step))
                executed_action = np.asarray(action_sequence[:num_steps], dtype=np.float64)
            else:
                executed_action = np.asarray(action_sequence, dtype=np.float64)

            discontinuity_cost = self.cost_delta_factor * self.lower_bound_time(
                                    parent_node.state,path_to_check[0])
            
            edge_cost = self.cost(self.env,self.agent,
                parent_node.state,executed_action,candidate_dt,path_to_check)

            total_cost = parent_node.cost_so_far + discontinuity_cost + edge_cost
            if total_cost >= self.max_cost:
                candidate_edge_mask[slot] = True
                continue

            total_elapsed_time = parent_node.time_elapsed + candidate_dt
            reached_goal, goal_dist = self.reached_goal(
                        candidate_state,self.goal,self.goal_radius,self.agent)
            
            if reached_goal and self._goal_state_is_safe(candidate_state, total_elapsed_time):
                new_node_id = self.add_rrt_node(candidate_state,parent_node_id,executed_action,
                    candidate_dt,path_to_check,total_elapsed_time,total_cost)
                self.path_found = True
                self.goal_node_id = new_node_id
                self.path_time = total_elapsed_time
                self.path_cost = total_cost
                candidate_edge_mask[slot] = True
                return new_node_id

            if not self._new_state_is_far_enough(candidate_state):
                candidate_edge_mask[slot] = True
                continue

            new_node_id = self.add_rrt_node(candidate_state,parent_node_id,executed_action,
                candidate_dt,path_to_check,total_elapsed_time,total_cost)
            candidate_edge_mask[slot] = True

            return new_node_id

        return None

    def get_path_to_node_id(self, goal_node_id):
        if goal_node_id is None or goal_node_id < 0:
            return [], [], [], []

        path_node_ids = []
        path_states = []
        path_action_sequences = []
        path_timesteps = []

        node_id = goal_node_id
        while node_id != -1:
            node = self.tree.nodes[node_id]["value"]
            path_node_ids.append(node.id)
            path_states.append(np.asarray(node.state, dtype=np.float64).copy())

            if node.parent_id != -1:
                path_action_sequences.append(np.asarray(node.parent_action).copy())
                path_timesteps.append(float(node.parent_action_duration))

            node_id = node.parent_id

        path_node_ids.reverse()
        path_states.reverse()
        path_action_sequences.reverse()
        path_timesteps.reverse()

        return (
            np.asarray(path_node_ids, dtype=np.int32),
            np.asarray(path_states, dtype=np.float64),
            path_action_sequences,
            np.asarray(path_timesteps, dtype=np.float64),
        )

    def get_high_resolution_path_and_actions(self):
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return (
                np.empty((0, self.agent.state_length), dtype=np.float64),
                np.empty((0, self.agent.action_length), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        node_id = self.goal_node_id
        total_states = 1
        total_controls = 0

        while node_id != -1:
            node = self.tree.nodes[node_id]["value"]
            if node.parent_id != -1:
                total_states += node.path_from_parent.shape[0]
                action_sequence = np.asarray(node.parent_action, dtype=np.float64)
                if action_sequence.ndim > 1:
                    total_controls += action_sequence.shape[0]
                elif action_sequence.size > 0:
                    total_controls += 1
            node_id = node.parent_id

        states = np.empty((total_states, self.agent.state_length), dtype=np.float64)
        controls = np.empty((total_controls, self.agent.action_length), dtype=np.float64)
        timesteps = np.empty(total_controls, dtype=np.float64)

        node_id = self.goal_node_id
        state_write = total_states
        control_write = total_controls

        while node_id != -1:
            node = self.tree.nodes[node_id]["value"]
            if node.parent_id != -1:
                path = np.asarray(node.path_from_parent, dtype=np.float64)
                num_path_states = path.shape[0]
                state_write -= num_path_states
                states[state_write:state_write + num_path_states] = path

                action_sequence = np.asarray(node.parent_action, dtype=np.float64)
                if action_sequence.ndim > 1:
                    num_controls = action_sequence.shape[0]
                    control_write -= num_controls
                    controls[control_write:control_write + num_controls] = action_sequence
                elif action_sequence.size > 0:
                    num_controls = 1
                    control_write -= 1
                    controls[control_write] = action_sequence
                else:
                    num_controls = 0

                if num_controls > 0:
                    dt = float(node.parent_action_duration) / float(num_controls)
                    timesteps[control_write:control_write + num_controls] = dt

            node_id = node.parent_id

        states[0] = np.asarray(self.start, dtype=np.float64)
        return states, controls, timesteps

    def plan_path(self):
        self.path_found = False
        self.goal_node_id = None
        self.path_cost = math.inf
        self.path_time = 0.0
        self.last_added_node_id = -1
        self.reset_tree()
        self.add_rrt_node(self.start, -1, None, 0.0, None, 0.0, 0.0)

        curr_num_steps = 0
        start_time = time.time()
        _, best_goal_dist = self.reached_goal(self.start, self.goal, self.goal_radius, self.agent)

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
                new_node = self.tree.nodes[new_node_id]["value"]
                _, goal_dist = self.reached_goal(new_node.state, self.goal, self.goal_radius, self.agent)
                best_goal_dist = min(best_goal_dist, goal_dist)
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

    def replan_path(self):
        self.plan_path()
