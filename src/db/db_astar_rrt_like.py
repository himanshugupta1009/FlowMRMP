"""
Basic discontinuity-bounded A* (Db-A*) search for the mrmp_with_kite_extend codebase.

This is intentionally search-only: it returns the stitched/transformed primitive path.
The trajectory-optimization repair step from iDb-A* is NOT implemented here.

Expected integration style:
    - Put this file in src/db_astar.py
    - Instantiate DbAStarPlanner similarly to KinoTIEBRRT / ConstrainedKinoTIEBRRT
    - Supply a motion-primitive library with fields:
        start_states, final_states or trajectories, trajectories, trajectory_lengths,
        actions, timesteps

The key difference from your existing KiTE-RRT code:
    KiTE-RRT currently re-propagates agent.get_next_state(parent_state, action, dt).
    Db-A* should NOT do that. It transforms/reuses the stored primitive trajectory.
"""

from __future__ import annotations

import heapq
import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List as PyList, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from numba.typed import List
from numba import njit, types

from rrt import RRT
from utils import euclidean_distance_numba_with_l, check_dynamic_collisions_to_end, \
    check_dynamic_collisions_to_end_3d, wrapped_angle_diff, normalized_state_distance_sq


@dataclass(order=True)
class _OpenItem:
    """Heap item.  heapq is min-heap, so this orders by f, then larger g."""
    f_score: float
    neg_g_score: float
    tie_breaker: int
    open_version: int = field(compare=False)
    node_id: int = field(compare=False)


@njit
def find_nearby_state_ids_normalized(query_state, node_states, n_nodes,
                                     merge_radius, indices, scales, is_angle,
                                     out_ids):
    count = 0
    r2 = merge_radius * merge_radius

    for i in range(n_nodes):
        d2 = normalized_state_distance_sq(
            query_state, node_states[i], indices, scales, is_angle
        )

        if d2 <= r2:
            out_ids[count] = i
            count += 1

    return count

@njit
def _disable_duplicate_motions_numba(start_states,
                                     final_states,
                                     connect_radius,
                                     merge_radius,
                                     connect_metric_indices,
                                     connect_metric_scales,
                                     connect_metric_is_angle,
                                     merge_metric_indices,
                                     merge_metric_scales,
                                     merge_metric_is_angle):
    n = start_states.shape[0]
    disabled = np.zeros(n, dtype=np.bool_)

    connect_radius_sq = connect_radius * connect_radius
    merge_radius_sq = merge_radius * merge_radius

    for i in range(n):
        if disabled[i]:
            continue

        for j in range(i + 1, n):
            if disabled[j]:
                continue

            ds_sq = normalized_state_distance_sq(
                start_states[i],start_states[j],
                connect_metric_indices,connect_metric_scales,connect_metric_is_angle)

            if ds_sq > connect_radius_sq:
                continue

            df_sq = normalized_state_distance_sq(
                final_states[i],final_states[j],
                merge_metric_indices,merge_metric_scales,merge_metric_is_angle)

            if df_sq < merge_radius_sq:
                disabled[j] = True

    return disabled


@njit
def _transform_unicycle_trajectory_numba(base_state, edge_start, raw_traj):
    """
    Fast batch transform for SE(2)-style unicycle primitives.

    base_state: (x_c, y_c, theta_c)
    edge_start: canonical primitive start, typically (0, 0, theta_e)
    raw_traj:   primitive trajectory in canonical frame, shape (T, 3)
    """
    out = np.empty_like(raw_traj)
    x_c = base_state[0]
    y_c = base_state[1]
    theta_c = base_state[2]
    theta_e = edge_start[2]
    delta_theta = theta_c - theta_e
    cos_d = np.cos(delta_theta)
    sin_d = np.sin(delta_theta)

    for k in range(raw_traj.shape[0]):
        x_f = raw_traj[k, 0]
        y_f = raw_traj[k, 1]
        theta_f = raw_traj[k, 2]

        dx_world = x_f * cos_d - y_f * sin_d
        dy_world = x_f * sin_d + y_f * cos_d

        out[k, 0] = x_c + dx_world
        out[k, 1] = y_c + dy_world
        out[k, 2] = theta_c + (theta_f - theta_e)

    return out


# class DbAStarNode(TreeNode):
#     def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
#                  path_from_parent, time_so_far, cost):
#         super().__init__(sid, state, parent_id, parent_action, parent_action_duration,
#                          path_from_parent, time_so_far, cost)
#         self.h_score = 0.0
#         self.f_score = cost
#         self.used_motion = -1
#         self.intermediate_state_index = -1
#         self.closed = False
#         self.open_version = 0


class DbAStarNodeMatrix:
    def __init__(self, initial_capacity, state_dim, max_sub_path_length):
        self.state = np.zeros((initial_capacity, state_dim), dtype=np.float64)
        self.parent = np.full(initial_capacity, -1, dtype=np.int32)

        self.cost = np.full(initial_capacity, np.inf, dtype=np.float64)
        self.h = np.zeros(initial_capacity, dtype=np.float64)
        self.f = np.zeros(initial_capacity, dtype=np.float64)
        self.time_elapsed = np.zeros(initial_capacity, dtype=np.float64)

        self.closed = np.zeros(initial_capacity, dtype=np.bool_)
        self.open_version = np.zeros(initial_capacity, dtype=np.int32)

        # Integer primitive id used to reach this node.
        self.used_motion = np.full(initial_capacity, -1, dtype=np.int32)

        # Actual duration used for the parent primitive.
        # This can be shorter than the full primitive duration if goal is hit midway.
        self.parent_action_duration = np.full(initial_capacity, -1.0, dtype=np.float64)

        # Transformed/truncated state trajectory from parent to this node.
        self.path_from_parent = np.zeros(
            (initial_capacity, max_sub_path_length, state_dim),
            dtype=np.float64,
        )
        self.sub_path_length = np.zeros(initial_capacity, dtype=np.int32)

        self.count = 0
        self.state_dim = state_dim
        self.max_sub_path_length = max_sub_path_length

    def append(
        self,
        state,
        parent,
        cost,
        h,
        time_elapsed,
        used_motion,
        parent_action_duration,
        path_from_parent,
    ):
        if self.count >= self.state.shape[0]:
            self._grow()

        idx = self.count

        self.state[idx] = state
        self.parent[idx] = parent

        self.cost[idx] = cost
        self.h[idx] = h
        self.f[idx] = cost + h
        self.time_elapsed[idx] = time_elapsed

        self.closed[idx] = False
        self.open_version[idx] = 0

        self.used_motion[idx] = used_motion
        self.parent_action_duration[idx] = parent_action_duration

        if path_from_parent is not None:
            path_len = path_from_parent.shape[0]

            if path_len > self.max_sub_path_length:
                raise ValueError(
                    f"path_from_parent length {path_len} exceeds "
                    f"max_sub_path_length {self.max_sub_path_length}"
                )

            self.path_from_parent[idx, :path_len, :] = path_from_parent
            self.sub_path_length[idx] = path_len
        else:
            self.sub_path_length[idx] = 0

        self.count += 1
        return idx

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)

        new_cost = np.full(new_cap, np.inf, dtype=np.float64)
        new_h = np.zeros(new_cap, dtype=np.float64)
        new_f = np.zeros(new_cap, dtype=np.float64)
        new_time_elapsed = np.zeros(new_cap, dtype=np.float64)

        new_closed = np.zeros(new_cap, dtype=np.bool_)
        new_open_version = np.zeros(new_cap, dtype=np.int32)

        new_used_motion = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action_duration = np.full(new_cap, -1.0, dtype=np.float64)

        new_path_from_parent = np.zeros(
            (new_cap, self.max_sub_path_length, self.state_dim),
            dtype=np.float64,
        )
        new_sub_path_length = np.zeros(new_cap, dtype=np.int32)

        new_state[:old_cap] = self.state
        new_parent[:old_cap] = self.parent

        new_cost[:old_cap] = self.cost
        new_h[:old_cap] = self.h
        new_f[:old_cap] = self.f
        new_time_elapsed[:old_cap] = self.time_elapsed

        new_closed[:old_cap] = self.closed
        new_open_version[:old_cap] = self.open_version

        new_used_motion[:old_cap] = self.used_motion
        new_parent_action_duration[:old_cap] = self.parent_action_duration

        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length

        self.state = new_state
        self.parent = new_parent

        self.cost = new_cost
        self.h = new_h
        self.f = new_f
        self.time_elapsed = new_time_elapsed

        self.closed = new_closed
        self.open_version = new_open_version

        self.used_motion = new_used_motion
        self.parent_action_duration = new_parent_action_duration

        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length


class DbAStarPlannerOld(RRT):
    """
    Basic Db-A* planner that reuses/transforms precomputed motion primitives.

    Important:
        This class implements the discontinuity-bounded search only. The returned path
        is a stitched sequence of transformed primitives and may contain small jumps
        at primitive junctions. This is exactly the object that the original iDb-A*
        optimizer later repairs.
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
                 max_iter: int = 100000,
                 planning_time: float = 30.0,
                 isvalid_function: Callable,
                 cost_function: Callable,
                 reached_goal_function: Callable,
                 random_point_function: Callable,
                 translate_function: Callable,
                 transform_trajectory_function: Callable,
                 motion_primitive_kd_tree,
                 get_motion_primitive_kd_tree_query: Callable,
                 max_neighbors_per_expand: Optional[int] = None,
                 max_cost: float = math.inf,
                 allow_intermediate_goal: bool = True,
                 num_intermediate_goal_checks: int = 4,
                 cost_delta_factor: float = 1.0,
                 limit_branching_factor: int = 20,
                 filter_duplicate_motions: bool = True,
                 duplicate_start_radius_scale: float = 0.5,
                 duplicate_final_radius_scale: float = 0.5,
                 merge_metric_state_size: Optional[int] = None,
                 duplicate_policy: str = "hard",  # "hard" or "soft"; hard is simpler/safer first
                 udf_seed: int = 77,
                 debug_flag: bool = False,
                 print_logs: bool = False,
                 dynamic_obstacles=List.empty_list(types.Array(types.float64, 2, 'C'))):

        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1")
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        if duplicate_policy not in ("hard", "soft"):
            raise ValueError("duplicate_policy must be 'hard' or 'soft'")

        super().__init__(start=start, goal=goal, goal_radius=goal_radius,
                         env=env, agent=agent,
                         use_fixed_sampling_time=True,
                         sampling_time_step=1.0,
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter,
                         planning_time=planning_time,
                         isvalid_function=isvalid_function,
                         cost_function=cost_function,
                         random_point_function=random_point_function,
                         reached_goal_function=reached_goal_function,
                         udf_seed=udf_seed,
                         debug_flag=debug_flag,
                         print_logs=print_logs,
                         dynamic_obstacles=dynamic_obstacles)

        self.motion_primitives = motion_primitives
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.connect_radius = self.alpha * self.delta
        self.merge_radius = (1.0 - self.alpha) * self.delta
        self.motion_primitive_kd_tree = motion_primitive_kd_tree
        self.get_motion_primitive_kd_tree_query = get_motion_primitive_kd_tree_query
        self.translate = translate_function
        self.transform_trajectory_function = transform_trajectory_function
        self.max_motions = int(motion_primitives.num_edges) 
        self.max_neighbors_per_expand = max_neighbors_per_expand
        self.max_cost = max_cost
        self.allow_intermediate_goal = allow_intermediate_goal
        self.num_intermediate_goal_checks = num_intermediate_goal_checks
        self.cost_delta_factor = float(cost_delta_factor)
        self.limit_branching_factor = int(limit_branching_factor)
        self.filter_duplicate_motions = bool(filter_duplicate_motions)
        self.duplicate_start_radius = float(duplicate_start_radius_scale) * self.connect_radius
        self.duplicate_final_radius = float(duplicate_final_radius_scale) * self.merge_radius
        if merge_metric_state_size is None:
            self.merge_metric_state_size = int(self.agent.state_length)
        else:
            self.merge_metric_state_size = int(merge_metric_state_size)
        self.duplicate_policy = duplicate_policy
        self.node_class = DbAStarNode
        self._heap_counter = 0
        self._disabled_motions_mask = np.zeros(self._num_edges(), dtype=np.bool_)

        # Reuse your existing dynamic-obstacle-to-end helper.
        if self.distance_metric_state_size == 2:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end
        elif self.distance_metric_state_size == 3:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end_3d
        else:
            self.dynamic_col_checker_to_end = None

        self.merge_metric_indices = self.agent.distance_indices
        self.merge_metric_scales = self.agent.distance_scales
        self.merge_metric_is_angle = self.agent.distance_is_angle

        # self._nearby_buffer = np.empty(self.max_iter + 10, dtype=np.int32)
        self._nearby_buffer = np.empty(10 * self.max_iter + 100, dtype=np.int32)
        self._db_node_matrix = DbAStarNodeMatrix(
            initial_capacity=self.max_iter + 10,
            state_dim=int(self.agent.state_length),
        )

        if self.filter_duplicate_motions:
            self._build_duplicate_motion_mask()


    # -------------------------------------------------------------------------
    # Primitive-library access
    # -------------------------------------------------------------------------
    def _num_edges(self) -> int:
        return min(int(self.max_motions),
                   int(getattr(self.motion_primitives, "num_edges", len(self.motion_primitives.timesteps))))

    def _edge_timestep(self, edge_id: int) -> float:
        return self.motion_primitives.timesteps[edge_id]

    def _edge_actions(self, edge_id: int):
        """Return whatever the primitive stores as action/action sequence."""
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
        """
        Transform/reuse the stored primitive trajectory from the current node.

        Returns
        -------
        new_state : np.ndarray
            Transformed final state of the primitive.
        path_from_parent : np.ndarray
            Transformed primitive trajectory used for collision checking and output.
            This is intended to exclude the parent state and include new_state.
        """
        edge_start = self._edge_start(edge_id)
        raw_traj = self._edge_trajectory(edge_id)
        transformed = self.transform_trajectory_function(base_state, edge_start, raw_traj)

        # Many primitive files store trajectories excluding the start state. If the
        # transformed first state is the current state, remove it from collision/output.
        if transformed.shape[0] > 1:
            d0 = euclidean_distance_numba_with_l(transformed[0], base_state, self.distance_metric_state_size)
            if d0 <= 1e-9:
                path = transformed[1:]
            else:
                path = transformed
        else:
            path = transformed

        new_state = np.asarray(path[-1], dtype=np.float64)
        return new_state, path

    # -------------------------------------------------------------------------
    # Search utilities
    # -------------------------------------------------------------------------
    def heuristic(self, state: np.ndarray) -> float:
        """Default admissibility is not guaranteed; good enough for first prototype."""
        return euclidean_distance_numba_with_l(state, self.goal, self.distance_metric_state_size)

    def lower_bound_time(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """
        Cheap lower bound on time between two states.
        """
        speed = getattr(self.agent, "max_speed", 1.0)
        speed = max(float(speed), 1e-9)
        return self._state_distance(state_a, state_b, self.distance_metric_state_size) / speed

    @staticmethod
    def _wrapped_angle_difference(a: float, b: float) -> float:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def _state_distance(self, a: np.ndarray, b: np.ndarray, size: int) -> float:
        """
        Distance used for merge / fallback lookup.

        For unicycle-like 3D states, include wrapped heading difference when the
        full state is being compared. Otherwise fall back to the existing Euclidean
        helper over the requested leading dimensions.
        """
        if size == 3 and a.shape[0] >= 3 and b.shape[0] >= 3:
            dx = float(a[0] - b[0])
            dy = float(a[1] - b[1])
            dtheta = self._wrapped_angle_difference(float(a[2]), float(b[2]))
            return math.sqrt(dx * dx + dy * dy + dtheta * dtheta)
        return euclidean_distance_numba_with_l(a, b, size)

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

    def _build_duplicate_motion_mask(self) -> None:
        n = self._num_edges()
        if n <= 1:
            self._disabled_motions_mask = np.zeros(n, dtype=np.bool_)
            return
        starts = np.asarray(self.motion_primitives.start_states[:n], dtype=np.float64)
        finals = np.empty((n, starts.shape[1]), dtype=np.float64)
        for i in range(n):
            finals[i] = self._edge_final(i)
        merge_metric = min(self.merge_metric_state_size, finals.shape[1])
        self._disabled_motions_mask = _disable_duplicate_motions_numba(
            starts,
            finals,
            self.duplicate_start_radius,
            self.duplicate_final_radius,
            min(self.distance_metric_state_size, starts.shape[1]),
            merge_metric,
        )
        if self.print_logs or self.debug_flag:
            num_disabled = int(np.sum(self._disabled_motions_mask))
            print(f"Db-A*: disabled {num_disabled}/{n} duplicate motions")

    def _push_open(self, open_heap: PyList[_OpenItem], node_id: int, node: DbAStarNode) -> None:
        self._heap_counter += 1
        node.open_version += 1
        heapq.heappush(open_heap, _OpenItem(node.f_score, -node.cost_so_far,
                                            self._heap_counter, node.open_version, node_id))

    def _get_candidate_edges(self, node_state: np.ndarray) -> Sequence[int]:
        """
        Return primitives whose canonical start is within alpha*delta.

        Preferred: use the same KD-tree interface as your KiTE-RRT code.
        Fallback: brute-force over edge starts using the agent distance metric.
        """
        query = self.get_motion_primitive_kd_tree_query(node_state)
        edge_ids = self.motion_primitive_kd_tree.radius_query(query, self.connect_radius)
        edge_ids = [int(i) for i in edge_ids if not self._disabled_motions_mask[int(i)]]

        if self.max_neighbors_per_expand is not None:
            edge_ids = edge_ids[:self.max_neighbors_per_expand]
        return edge_ids

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

    def _find_nearby_state_ids(self, state: np.ndarray) -> int:
        return find_nearby_state_ids_normalized(
            state,
            self._db_node_matrix.state,
            self._db_node_matrix.count,
            self.merge_radius,
            self.merge_metric_indices,
            self.merge_metric_scales,
            self.merge_metric_is_angle,
            self._nearby_buffer,
        )

    def _add_astar_node(self, state, parent_node_id, parent_action, parent_action_duration,
                        path_from_parent, time_elapsed, g_score, h_score, used_motion,
                        intermediate_state_index=-1) -> int:
        
        node_id = self._db_node_matrix.append(
                state=state,
                parent=parent_node_id,
                cost=g_score,
                h=h_score,
                time_elapsed=time_elapsed,
                used_motion=used_motion,
        )
        nx_id = self.add_rrt_node(state, parent_node_id, parent_action,
                                    parent_action_duration, path_from_parent,
                                    time_elapsed, g_score)
        assert nx_id == node_id, "Node ID mismatch between matrix and graph"
        node = self.tree.nodes[node_id]["value"]
        node.h_score = h_score
        node.f_score = g_score + h_score
        node.used_motion = used_motion
        node.intermediate_state_index = intermediate_state_index
        return node_id

    def _update_existing_node(self, node_id: int, node: DbAStarNode, *,
                              state, parent_node_id, parent_action, parent_action_duration,
                              path_from_parent, time_elapsed, g_score, h_score,
                              used_motion, intermediate_state_index=-1) -> None:
        """
        Lower the g-value of an existing representative node.

        We keep the same node id to avoid rebuilding the graph, but replace the
        representative state and parent pointer. This is the simplest Python analogue
        of keeping the best path to a merged region.
        """
        node.state = np.asarray(state, dtype=np.float64)
        node.parent_id = parent_node_id
        node.parent_action = parent_action
        node.parent_action_duration = parent_action_duration
        node.path_from_parent = path_from_parent
        node.time_elapsed = round(time_elapsed, self.roundoff_digits)
        node.cost_so_far = g_score
        node.h_score = h_score
        node.f_score = g_score + h_score
        node.used_motion = used_motion
        node.intermediate_state_index = intermediate_state_index
        node.closed = False if self.duplicate_policy == "soft" else node.closed

        # NOTE: self._db_node_matrix is not updated here. That matrix is used by RRT
        # nearest-neighbor queries, not by Db-A* search. Path reconstruction uses
        # the graph node values, so this is okay for the prototype.

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

    # -------------------------------------------------------------------------
    # Main search
    # -------------------------------------------------------------------------         
    def plan_path(self):
        self.path_found = False
        self.goal_node_id = None
        self.path_cost = math.inf
        self.path_time = 0.0
        self.last_added_node_id = -1

        self.reset_tree()
        self._heap_counter = 0

        # Reset the fast Db-A* node matrix.
        self._db_node_matrix = DbAStarNodeMatrix(
            initial_capacity=self.max_iter + 10,
            state_dim=int(self.agent.state_length),
        )

        h0 = self.heuristic(self.start)

        start_id = self._add_astar_node(
            self.start,
            -1,
            None,
            None,
            None,
            0.0,
            0.0,
            h0,
            used_motion=-1,
        )

        open_heap: PyList[_OpenItem] = []
        self._push_open(open_heap, start_id, self.tree.nodes[start_id]["value"])

        start_wall = time.time()
        num_expands = 0
        best_goal_dist = h0

        while open_heap:
            if num_expands >= self.max_iter:
                break

            if time.time() - start_wall >= self.planning_time:
                break

            item = heapq.heappop(open_heap)
            curr_id = item.node_id
            curr_node = self.tree.nodes[curr_id]["value"]

            # Skip stale heap entries.
            if item.open_version != curr_node.open_version:
                continue

            # Use the matrix for fast search state.
            if self._db_node_matrix.closed[curr_id]:
                continue

            if self._db_node_matrix.cost[curr_id] >= self.max_cost:
                continue

            curr_node.closed = True
            self._db_node_matrix.closed[curr_id] = True
            num_expands += 1

            curr_state = self._db_node_matrix.state[curr_id]
            curr_cost = self._db_node_matrix.cost[curr_id]
            curr_time = self._db_node_matrix.time_elapsed[curr_id]

            reached, goal_dist = self.reached_goal(
                curr_state,
                self.goal,
                self.goal_radius,
                self.agent,
            )

            best_goal_dist = min(best_goal_dist, goal_dist)

            if reached and self._goal_state_is_safe(curr_state, curr_time):
                self.path_found = True
                self.goal_node_id = curr_id
                self.path_cost = curr_cost
                self.path_time = curr_time
                break

            candidate_edges = self._get_candidate_edges(curr_state)

            if self.debug_flag:
                print(
                    f"Db-A*: expand node={curr_id}, "
                    f"g={curr_cost:.3f}, "
                    f"h={self._node_matrix.h[curr_id]:.3f}, "
                    f"neighbors={len(candidate_edges)}"
                )

            accepted_expansions = 0

            for edge_id in candidate_edges:
                timestep = self._edge_timestep(edge_id)
                parent_time = curr_time

                new_state, path = self.transform_primitive(curr_state, edge_id)

                if path.shape[0] == 0:
                    continue

                # If the path reaches the goal in the middle, truncate it.
                goal_idx = self._try_goal_along_path(path)

                if goal_idx is not None:
                    path_to_check = path[:goal_idx + 1]
                    candidate_state = path_to_check[-1]
                    candidate_dt = round(
                        (goal_idx + 1) * self.minimum_time_step,
                        self.roundoff_digits,
                    )
                    intermediate_state_index = goal_idx
                else:
                    path_to_check = path
                    candidate_state = new_state
                    candidate_dt = timestep
                    intermediate_state_index = -1

                if not self._is_path_valid(path_to_check, parent_time, candidate_dt):
                    continue

                action = self._edge_actions(edge_id)

                discontinuity_cost = self.cost_delta_factor * self.lower_bound_time(
                    curr_state,
                    path_to_check[0],
                )

                edge_cost = self.cost(
                    self.env,
                    self.agent,
                    curr_state,
                    action,
                    candidate_dt,
                    path_to_check,
                )

                g_new = curr_cost + discontinuity_cost + edge_cost

                if g_new >= self.max_cost:
                    continue

                h_new = self.heuristic(candidate_state)
                total_time = parent_time + candidate_dt

                # ------------------------------------------------------------
                # Fast nearby/merge logic using DbAStarNodeMatrix + Numba.
                # ------------------------------------------------------------
                if intermediate_state_index != -1:
                    near_count = 0
                else:
                    near_count = self._find_nearby_state_ids(candidate_state)

                new_node_id = None

                if near_count == 0:
                    new_node_id = self._add_astar_node(
                        candidate_state,
                        curr_id,
                        action,
                        candidate_dt,
                        path_to_check,
                        total_time,
                        g_new,
                        h_new,
                        used_motion=edge_id,
                        intermediate_state_index=intermediate_state_index,
                    )

                    new_node = self.tree.nodes[new_node_id]["value"]
                    self._push_open(open_heap, new_node_id, new_node)
                    accepted_expansions += 1

                else:
                    for kk in range(near_count):
                        near_id = int(self._nearby_buffer[kk])
                        near_node = self.tree.nodes[near_id]["value"]

                        tentative_g = g_new + self.lower_bound_time(
                            candidate_state,
                            self._db_node_matrix.state[near_id],
                        )

                        if tentative_g + 1e-12 >= self._db_node_matrix.cost[near_id]:
                            continue

                        near_h = self._db_node_matrix.h[near_id]
                        if near_h == 0.0:
                            near_h = self.heuristic(self._db_node_matrix.state[near_id])

                        # Update fast matrix representation.
                        self._db_node_matrix.parent[near_id] = curr_id
                        self._db_node_matrix.cost[near_id] = tentative_g
                        self._db_node_matrix.h[near_id] = near_h
                        self._db_node_matrix.f[near_id] = tentative_g + near_h
                        self._db_node_matrix.time_elapsed[near_id] = total_time
                        self._db_node_matrix.closed[near_id] = False
                        self._db_node_matrix.used_motion[near_id] = edge_id

                        # Keep NetworkX node consistent for now.
                        self._update_existing_node(
                            near_id,
                            near_node,
                            state=near_node.state,
                            parent_node_id=curr_id,
                            parent_action=action,
                            parent_action_duration=candidate_dt,
                            path_from_parent=path_to_check,
                            time_elapsed=total_time,
                            g_score=tentative_g,
                            h_score=near_h,
                            used_motion=edge_id,
                            intermediate_state_index=-1,
                        )

                        near_node.closed = False
                        self._push_open(open_heap, near_id, near_node)

                        accepted_expansions += 1

                        if new_node_id is None:
                            new_node_id = near_id

                # Goal check after accepting/merging.
                if new_node_id is not None:
                    reached, _ = self.reached_goal(
                        candidate_state,
                        self.goal,
                        self.goal_radius,
                        self.agent,
                    )
                else:
                    reached = False

                if reached and self._goal_state_is_safe(candidate_state, total_time):
                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = g_new
                    self.path_time = total_time
                    open_heap.clear()
                    break

                if accepted_expansions >= self.limit_branching_factor:
                    break

        total_wall = time.time() - start_wall
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            print(
                f"Db-A*: found={self.path_found}, "
                f"expands={num_expands}, "
                f"nodes={self._db_node_matrix.count}, "
                f"best_goal_dist={best_goal_dist:.3f}, "
                f"wall_time={total_wall:.3f}s"
            )

    def replan_path(self):
        # For now, replan from scratch. KCBS can still call this safely.
        self.plan_path()


    def get_dbastar_path_to_node_id(self, goal_node_id):
        """
        Return the Db-A* path as node ids, states, action sequences, and primitive durations.

        Unlike RRT.get_path_to_node_id(), this supports parent_action being a control
        sequence with shape (T, action_dim), not just a single action.
        """
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
            path_states.append(node.state)

            # Root has no parent action.
            if node.parent_id != -1:
                path_action_sequences.append(node.parent_action)
                path_timesteps.append(node.parent_action_duration)

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


    def get_dbastar_high_resolution_path_and_actions(self):
        """
        Return the stitched high-resolution state trajectory and the corresponding
        primitive action sequence list.

        states:
            Full state sequence along the stitched primitive path.

        action_sequences:
            List of action arrays, one per primitive edge.
            Each element may have shape (T, action_dim).
        """
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return (
                np.empty((0, self.agent.state_length), dtype=np.float64),
                [],
                np.empty(0, dtype=np.float64),
            )

        edge_paths = []
        action_sequences = []
        timesteps = []

        node_id = self.goal_node_id

        while node_id != -1:
            node = self.tree.nodes[node_id]["value"]

            if node.parent_id != -1:
                if node.path_from_parent is not None and node.path_from_parent.shape[0] > 0:
                    edge_paths.append(node.path_from_parent)

                action_sequences.append(node.parent_action)
                timesteps.append(node.parent_action_duration)

            node_id = node.parent_id

        edge_paths.reverse()
        action_sequences.reverse()
        timesteps.reverse()

        total_len = 1
        for path in edge_paths:
            total_len += path.shape[0]

        states = np.empty((total_len, self.agent.state_length), dtype=np.float64)
        states[0] = self.start

        write_idx = 1
        for path in edge_paths:
            n = path.shape[0]
            states[write_idx:write_idx + n] = path
            write_idx += n

        return states, action_sequences, np.asarray(timesteps, dtype=np.float64)
