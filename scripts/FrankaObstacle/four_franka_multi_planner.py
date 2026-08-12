#!/usr/bin/env python3
"""Shared exact-collision planner adapters for four-Franka benchmarks.

This module keeps all articulated geometry in the existing 295-sphere
MorphIt/cuRobo adapter.  It supplies current VanillaRRT and FlowEBRRT as low
level planners for Prioritized Planning and KCBS, plus centralized CRRT and a
dynamic Flow-generated CRRT-EB variant.  PyBullet is intentionally absent.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
OBSTACLE_SCRIPTS = Path(__file__).resolve().parent
MRMP_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
FLOW_SOURCE = REPOSITORY_ROOT / "src"
for module_path in (MAIN_SCRIPTS, OBSTACLE_SCRIPTS, MRMP_SOURCE, FLOW_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import EmptyFrankaEnvironment  # noqa: E402
from cRRT import CRRT  # noqa: E402
from flow_eb_rrt import FlowEBRRT, FrankaFlowEdgeGenerator  # noqa: E402
from kcbs import KCBS  # noqa: E402
from prioritized_planning import PrioritizedPlanning  # noqa: E402
from rrt import RRT  # noqa: E402


def joint_metric(agents, joint_state: np.ndarray) -> np.ndarray:
    """Concatenate each arm's normalized 14D search state."""
    state = np.asarray(joint_state, dtype=np.float64)
    return np.concatenate(
        [agent.get_distance_metric_state(state[index * 14 : (index + 1) * 14])
         for index, agent in enumerate(agents)]
    )


class FlowCRRT(CRRT):
    """Centralized CRRT-EB whose bundles come from the current Flow model.

    One model batch generates 32 variable-duration acceleration sequences for
    every active arm at a selected joint-tree node.  Candidates are ranked by
    normalized 7D joint-position distance, physically propagated, checked per
    arm against cuRobo, then checked jointly with exact world-frame spheres.
    """

    def __init__(
        self,
        *,
        flow_edge_generator: FrankaFlowEdgeGenerator,
        max_joint_edge_trials: int = 20,
        epsilon_random: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flow_edge_generator = flow_edge_generator
        self.max_joint_edge_trials = int(max_joint_edge_trials)
        self.epsilon_random = float(epsilon_random)
        self.profile = {
            "flow_generation_calls": 0,
            "flow_generated_bundles": 0,
            "joint_flow_trials": 0,
            "joint_flow_valid": 0,
            "joint_flow_static_rejections": 0,
            "joint_flow_interarm_rejections": 0,
            "random_fallbacks": 0,
        }

    def _bundle_candidates(self, parent_node, active_indices):
        cache = getattr(parent_node, "flow_bundles", None)
        if cache is None:
            cache = {}
            parent_node.flow_bundles = cache
        missing = [index for index in active_indices if index not in cache]
        if missing:
            states = np.stack(
                [self.get_agent_state(parent_node.state, index) for index in missing]
            )
            bundles = self.flow_edge_generator.sample_batch(
                states, num_edges=self.flow_edge_generator.set_size
            )
            self.profile["flow_generation_calls"] += 1
            self.profile["flow_generated_bundles"] += len(bundles)
            cache.update(zip(missing, bundles))
        orders = {}
        for index in active_indices:
            bundle = cache[index]
            target = self._last_agent_raw_samples[index]
            scores = np.asarray(
                [self.agents[index].get_flow_edge_ranking_distance(
                    predicted, target,
                    bool(self._last_agent_sample_was_goal[index]))
                 for predicted in bundle.final_states],
                dtype=np.float64,
            )
            orders[index] = np.argsort(scores, kind="stable")
        return cache, orders

    def _select_best_joint_extension_candidate(self, parent_node, random_point):
        if self.epsilon_random > 0.0 and self.rng.random() < self.epsilon_random:
            self.profile["random_fallbacks"] += 1
            return super()._select_best_joint_extension_candidate(
                parent_node, random_point)

        active = [index for index in range(self.num_agents)
                  if not self._agent_is_parked(parent_node, index)]
        if not active:
            return None
        cache, orders = self._bundle_candidates(parent_node, active)
        cursors = {index: 0 for index in active}

        for trial in range(self.max_joint_edge_trials):
            self.profile["joint_flow_trials"] += 1
            paths = {}
            actions = {}
            invalid_agent = None
            for index in active:
                order = orders[index]
                if cursors[index] >= len(order):
                    invalid_agent = index
                    break
                edge_index = int(order[cursors[index]])
                action = np.asarray(
                    cache[index].action_sequences[edge_index], dtype=np.float64)
                violation = self.agents[index].action_sequence_violation(
                    action, cache[index].action_dt)
                if violation is not None:
                    cursors[index] += 1
                    invalid_agent = index
                    break
                _, path = self.agents[index].get_next_state_sequence(
                    self.get_agent_state(parent_node.state, index),
                    action,
                    cache[index].action_dt,
                )
                paths[index] = path
                actions[index] = action
            if invalid_agent is not None:
                continue

            common_steps = min(len(paths[index]) for index in active)
            if common_steps <= 0:
                break
            edge_dt = float(cache[active[0]].action_dt)
            edge_time = common_steps * edge_dt
            joint_path = np.empty((common_steps, self.joint_state_size), dtype=np.float64)
            new_joint_state = np.empty(self.joint_state_size, dtype=np.float64)
            joint_action = np.zeros(self.joint_action_size, dtype=np.float64)

            for index in range(self.num_agents):
                if self._agent_is_parked(parent_node, index):
                    held = self.get_agent_state(parent_node.state, index)
                    agent_path = np.repeat(held[None, :], common_steps, axis=0)
                else:
                    agent_path = paths[index][:common_steps]
                    agent = self.agents[index]
                    if not self.isvalid[index](
                        agent_path, agent.radius, self.env.size,
                        self.static_circular_obstacles,
                        self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles,
                        agent.dynamic_limit_indices, agent.dynamic_limit_values,
                        self.env.obstacle_buffer, self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        edge_time, edge_dt,
                    ):
                        cursors[index] += 1
                        invalid_agent = index
                        self.profile["joint_flow_static_rejections"] += 1
                        break
                self.set_agent_path(joint_path, index, agent_path)
                self.set_agent_state(new_joint_state, index, agent_path[-1])
            if invalid_agent is not None:
                continue

            collision, first, second, _ = self.first_joint_path_collision(joint_path)
            if collision:
                # Alternate which colliding arm changes so the repair sequence
                # explores both sides instead of deterministically starving one.
                selected = first if trial % 2 == 0 else second
                if selected not in cursors:
                    selected = second if first not in cursors else first
                if selected in cursors:
                    cursors[selected] += 1
                self.profile["joint_flow_interarm_rejections"] += 1
                continue

            self.profile["joint_flow_valid"] += 1
            return new_joint_state, joint_path, joint_action, edge_time

        self.profile["random_fallbacks"] += 1
        return super()._select_best_joint_extension_candidate(parent_node, random_point)


class _FrankaConstraintMixin:
    """Install exact articulated KCBS constraints on an existing RRT class."""

    def enable_franka_constraints(self, adapter, robot_index: int) -> None:
        self.franka_constraint_adapter = adapter
        self.franka_constraint_robot_index = int(robot_index)
        self.franka_constraints = []
        self._unconstrained_isvalid = self.isvalid
        self.isvalid = self._isvalid_with_constraints

    def set_constraints(self, constraints) -> None:
        self.franka_constraints = list(constraints)

    def _isvalid_with_constraints(self, path, *args) -> bool:
        if not self._unconstrained_isvalid(path, *args):
            return False
        return self.franka_constraint_adapter.constraint_path_valid(
            self.franka_constraint_robot_index,
            path,
            self.franka_constraints,
            start_time=float(args[-3]),
            step_time=float(args[-1]),
        )

    def parking_blocked(self, state, arrival_time):
        if super().parking_blocked(state, arrival_time):
            return True
        return self.franka_constraint_adapter.constraint_parking_blocked(
            self.franka_constraint_robot_index,
            state,
            self.franka_constraints,
            arrival_time=float(arrival_time),
            step_time=float(self.minimum_time_step),
        )

    def plan_path_with_constraints(self, curr_tree_structure, constraints):
        # Exact sphere constraints cannot be safely pruned using the legacy
        # point-radius tree-reuse predicate.  Scratch replanning is deliberate.
        del curr_tree_structure
        self.set_constraints(constraints)
        return self.plan_path()


class FrankaConstrainedRRT(_FrankaConstraintMixin, RRT):
    """Current VanillaRRT with exact KCBS sphere constraints."""


class FrankaConstrainedFlowEBRRT(_FrankaConstraintMixin, FlowEBRRT):
    """Current FlowEBRRT with exact KCBS sphere constraints."""


def make_single_arm_planner(
    *,
    base: str,
    agent,
    start: np.ndarray,
    goal: np.ndarray,
    seed: int,
    planning_time: float,
    goal_radius: float,
    generator: FrankaFlowEdgeGenerator | None = None,
    constrained: bool = False,
):
    """Build the latest single-arm Vanilla or Flow planner configuration."""
    common = dict(
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        env=EmptyFrankaEnvironment(),
        agent=agent,
        use_fixed_sampling_time=False,
        max_iter=10_000_000,
        planning_time=planning_time,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=seed,
        goal_sampling_probability=0.30,
        dynamic_agent_clearance=0.0,
        debug_flag=False,
        print_logs=False,
    )
    if base == "vanilla":
        planner_class = FrankaConstrainedRRT if constrained else RRT
        return planner_class(
            **common,
            sampling_time_step=1.0,
            minimum_time_step=0.02,
            num_extension_trials=16,
        )
    if base != "flow" or generator is None:
        raise ValueError("Flow planning requires a FrankaFlowEdgeGenerator")
    planner_class = FrankaConstrainedFlowEBRRT if constrained else FlowEBRRT
    planner = planner_class(
        **common,
        flow_edge_generator=generator,
        sampling_time_step=0.30,
        minimum_time_step=generator.dt,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=generator.set_size * 4,
        flow_prefetch_batch_size=16,
        minimum_sequence_prefix_steps=5,
        truncate_sequence_to_target=False,
        num_skip_edges=32,
        num_random_edges=1,
        epsilon_random=0.05,
        goal_edge_pool_multiplier=4,
        goal_edge_pool_radius_multiplier=1.25,
    )
    planner.set_profile_enabled(True)
    return planner


def make_crrt(
    *,
    base: str,
    adapter,
    starts: np.ndarray,
    goals: np.ndarray,
    seed: int,
    planning_time: float,
    goal_radius: float,
    generator: FrankaFlowEdgeGenerator | None = None,
):
    """Build centralized Vanilla CRRT or dynamic Flow CRRT-EB."""
    agents = adapter.agents
    common = dict(
        agents=agents,
        starts=list(starts),
        goals=list(goals),
        goal_radii=[goal_radius] * len(agents),
        env=EmptyFrankaEnvironment(),
        use_fixed_sampling_time=False,
        sampling_time_step=1.0,
        minimum_time_step=0.02,
        max_iter=10_000_000,
        planning_time=planning_time,
        num_extension_trials=16,
        isvalid_function=[agent.is_new_node_valid for agent in agents],
        cost_function=[agent.get_cost for agent in agents],
        reached_goal_function=[agent.agent_reached_goal for agent in agents],
        random_point_function=[agent.get_random_point for agent in agents],
        udf_seed=seed,
        goal_sampling_probability=0.30,
        branch_goal_parking=True,
        joint_state_metric_function=lambda state: joint_metric(agents, state),
        joint_path_collision_function=adapter.joint_path_collides,
        first_joint_path_collision_function=adapter.joint_path_first_collision,
        print_logs=False,
        debug_flag=False,
    )
    if base == "vanilla":
        return CRRT(**common)
    if base != "flow" or generator is None:
        raise ValueError("Flow CRRT requires a FrankaFlowEdgeGenerator")
    common["num_extension_trials"] = 4
    planner = FlowCRRT(
        **common,
        flow_edge_generator=generator,
        max_joint_edge_trials=20,
        epsilon_random=0.05,
    )
    generator.set_profile_enabled(True)
    return planner


def make_kcbs(
    *,
    base: str,
    adapter,
    starts: np.ndarray,
    goals: np.ndarray,
    seed: int,
    planning_time: float,
    goal_radius: float,
    generator: FrankaFlowEdgeGenerator | None = None,
):
    """Build KCBS with exact 295-sphere conflicts and constraints."""
    planners = []
    for index, agent in enumerate(adapter.agents):
        planner = make_single_arm_planner(
            base=base,
            agent=agent,
            start=starts[index],
            goal=goals[index],
            seed=seed + index,
            planning_time=planning_time,
            goal_radius=goal_radius,
            generator=generator,
            constrained=True,
        )
        planner.enable_franka_constraints(adapter, index)
        planners.append(planner)
    return KCBS(
        env=EmptyFrankaEnvironment(),
        agents=adapter.agents,
        low_level_planners=planners,
        max_trials=10_000,
        planning_time=planning_time,
        minimum_time_step=0.02,
        clearance_threshold=0.0,
        rng_seed=seed,
        reuse_tree=False,
        store_cbs_nodes=False,
        conflict_detector=adapter.first_synchronized_path_conflict,
        constraint_builder=lambda replanned, obstacle, keys, states: (
            np.asarray(keys, dtype=np.float64),
            np.asarray(states, dtype=np.float64),
            float(obstacle),
        ),
        print_logs=False,
        debug_flag=False,
    )


def path_motion_time(path: np.ndarray, dt: float) -> float:
    """Return task execution time, excluding a terminal stationary tail."""
    states = np.asarray(path, dtype=np.float64)
    if len(states) <= 1:
        return 0.0
    last = len(states) - 1
    while last > 0 and np.allclose(states[last], states[last - 1]):
        last -= 1
    return float(last * dt)


def audit_synchronized_paths(adapter, paths: Iterable[np.ndarray], dt: float):
    """Audit saved four-arm paths with static/self and exact inter-arm geometry."""
    arrays = [np.asarray(path, dtype=np.float64) for path in paths]
    if len(arrays) != 4 or any(path.ndim != 2 or path.shape[1] != 14 or len(path) == 0
                               for path in arrays):
        return False, {"reason": "missing_or_malformed_path"}
    per_robot = []
    for index, path in enumerate(arrays):
        static, _ = adapter.validate_path(
            index, path[:, :7], start_time=0.0, step_time=dt)
        limits = np.asarray(
            [adapter.agents[index].is_state_within_limits(state) for state in path]
        )
        per_robot.append(bool(np.all(static & limits)))
    conflict = adapter.first_synchronized_path_conflict(arrays, dt)
    interarm_free = len(conflict[0]) == 0
    return bool(all(per_robot) and interarm_free), {
        "per_robot_static_self_limits_valid": per_robot,
        "interarm_collision_free": interarm_free,
        "first_conflict_time": None if interarm_free else float(conflict[0][0]),
        "first_conflict_pair": None if interarm_free else [int(conflict[1]), int(conflict[3])],
    }
