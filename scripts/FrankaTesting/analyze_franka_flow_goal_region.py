#!/usr/bin/env python3
"""Audit FlowEBRRT's 32 candidates at single-Franka near-goal nodes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from franka_paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RAW_TRAJECTORY_DATASET,
    DEFAULT_URDF,
    FLOW_EB_RRT_SRC,
    FLOWMRMP_ROOT,
)
from franka_solution_quality import path_motion_time_from_states

MAIN_SCRIPTS = FLOWMRMP_ROOT / "scripts"
CORE_RRT_SRC = FLOWMRMP_ROOT / "mrmp_with_kite_extend" / "src"
for module_path in (MAIN_SCRIPTS, FLOW_EB_RRT_SRC, CORE_RRT_SRC):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import (  # noqa: E402
    EmptyFrankaEnvironment,
    FrankaPanda,
    FrankaSelfCollisionChecker,
)
from benchmark_franka_flow_eb_rrt import (  # noqa: E402
    dense_path_to_node,
    load_problems,
)
from benchmark_franka_vanilla_rrt import sample_dataset_state_bank  # noqa: E402
from flow_eb_rrt import FlowEBRRT, FrankaFlowEdgeGenerator  # noqa: E402


CURRENT_PROBLEMS = (
    FLOWMRMP_ROOT
    / "data"
    / "benchmarks"
    / "franka_reachable_morphit295_ab_100_goal_r040.npz"
)
CURRENT_DATASET = (
    DEFAULT_RAW_TRAJECTORY_DATASET
    if DEFAULT_RAW_TRAJECTORY_DATASET.is_file()
    else Path.home() / "Downloads" / "dataset200k.h5"
)


class RetainedDiagnosticGenerator(FrankaFlowEdgeGenerator):
    """Keep an immutable copy of controls that the planner normally releases."""

    def sample_batch(self, states, num_edges=None):
        bundles = super().sample_batch(states, num_edges=num_edges)
        for bundle in bundles:
            bundle.diagnostic_action_sequences = tuple(
                np.array(actions, dtype=np.float32, copy=True)
                for actions in bundle.action_sequences
            )
        return bundles


class DiagnosticFlowEBRRT(FlowEBRRT):
    """Count node selections and random fallbacks after bundle exhaustion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_selection_counts: dict[int, int] = {}
        self.node_random_fallback_counts: dict[int, int] = {}
        self.node_exhausted_fallback_counts: dict[int, int] = {}

    def extend_tree(self, parent_node_id, parent_node, random_point):
        node_id = int(parent_node_id)
        self.node_selection_counts[node_id] = (
            self.node_selection_counts.get(node_id, 0) + 1
        )
        return super().extend_tree(parent_node_id, parent_node, random_point)

    def _try_random_control_profiled(self, parent_node, parent_node_id, random_point):
        node_id = int(parent_node_id)
        self.node_random_fallback_counts[node_id] = (
            self.node_random_fallback_counts.get(node_id, 0) + 1
        )
        mask = parent_node.edge_bundle_mask
        if mask is not None and len(mask) and bool(np.all(mask)):
            self.node_exhausted_fallback_counts[node_id] = (
                self.node_exhausted_fallback_counts.get(node_id, 0) + 1
            )
        return super()._try_random_control_profiled(
            parent_node, parent_node_id, random_point
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", type=int, default=61)
    parser.add_argument("--dataset", type=Path, default=CURRENT_DATASET)
    parser.add_argument("--problems", type=Path, default=CURRENT_PROBLEMS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--planning-time", type=float, default=30.0)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument(
        "--near-goal-radius",
        type=float,
        default=None,
        help="Diagnostic radius; defaults to twice --goal-radius.",
    )
    parser.add_argument("--max-near-goal-nodes", type=int, default=50)
    parser.add_argument("--state-bank-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--goal-sampling-probability", type=float, default=0.30)
    parser.add_argument("--max-random-edge-time", type=float, default=0.30)
    parser.add_argument("--acceleration-scale", type=float, default=0.50)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--flow-prefetch-batch-size", type=int, default=16)
    parser.add_argument("--num-sorted-edge-trials", type=int, default=32)
    parser.add_argument("--num-random-edges", type=int, default=1)
    parser.add_argument("--epsilon-random", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def rejection_reason(agent: FrankaPanda, before: tuple[int, int]) -> str:
    if agent.limit_rejections > before[0]:
        return "state_limit_invalid"
    if agent.self_collision_rejections > before[1]:
        return "self_collision_invalid"
    return "path_invalid_other"


def audit_edge(
    *,
    agent: FrankaPanda,
    parent_state: np.ndarray,
    actions: np.ndarray,
    action_dt: float,
    goal: np.ndarray,
    goal_radius: float,
) -> dict[str, object]:
    violation = agent.action_sequence_violation(actions, action_dt)
    if violation is not None:
        return {
            "classification": f"{violation}_invalid",
            "actual_endpoint_goal_distance": np.nan,
            "minimum_waypoint_goal_distance": np.nan,
            "first_goal_step": -1,
        }

    _, path = agent.get_next_state_sequence(parent_state, actions, action_dt)
    distances = np.asarray(
        [agent.get_goal_distance(state, goal) for state in path], dtype=np.float64
    )
    goal_steps = np.flatnonzero(distances <= goal_radius)
    first_goal_step = int(goal_steps[0]) if goal_steps.size else -1
    validation_path = path if first_goal_step < 0 else path[: first_goal_step + 1]
    before = (agent.limit_rejections, agent.self_collision_rejections)
    valid = agent.is_new_node_valid(validation_path)
    if not valid:
        classification = rejection_reason(agent, before)
    elif first_goal_step >= 0:
        classification = "valid_goal_reaching"
    else:
        classification = "valid_non_goal"
    return {
        "classification": classification,
        "actual_endpoint_goal_distance": float(distances[-1]),
        "minimum_waypoint_goal_distance": float(distances.min()),
        "first_goal_step": first_goal_step,
    }


def summarize_counts(values: list[dict[str, object]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        label = str(value[key])
        result[label] = result.get(label, 0) + 1
    return dict(sorted(result.items()))


def main() -> None:
    args = parse_args()
    if args.planning_time <= 0.0 or args.max_near_goal_nodes <= 0:
        raise ValueError("planning-time and max-near-goal-nodes must be positive")
    near_goal_radius = (
        2.0 * args.goal_radius
        if args.near_goal_radius is None
        else args.near_goal_radius
    )
    if near_goal_radius < args.goal_radius:
        raise ValueError("near-goal-radius must be at least the goal radius")
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        or FLOWMRMP_ROOT / "results" / "franka_flow_goal_region" / run_name
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problems(args.problems, [args.problem_id])[0]
    rng = np.random.default_rng(args.seed)
    state_bank, _, _ = sample_dataset_state_bank(
        args.dataset, args.state_bank_size, rng
    )
    checker = FrankaSelfCollisionChecker(args.urdf)
    try:
        agent = FrankaPanda(
            state_bank=state_bank,
            collision_checker=checker,
            agent_id=args.problem_id,
            acceleration_scale=args.acceleration_scale,
        )
        start = np.asarray(problem["start"], dtype=np.float64)
        goal = np.asarray(problem["goal"], dtype=np.float64)
        seed = args.seed + args.problem_id
        generator = RetainedDiagnosticGenerator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            sample_steps=args.sample_steps,
            clamp_outputs=True,
            seed=42 + seed,
        )
        planner = DiagnosticFlowEBRRT(
            start=start,
            goal=goal,
            goal_radius=args.goal_radius,
            env=EmptyFrankaEnvironment(),
            agent=agent,
            flow_edge_generator=generator,
            use_fixed_sampling_time=False,
            sampling_time_step=args.max_random_edge_time,
            minimum_time_step=generator.dt,
            max_iter=10_000_000,
            planning_time=args.planning_time,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function=agent.agent_reached_goal,
            translate_function=agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            max_num_edges_per_node=generator.set_size,
            flow_prefetch_batch_size=args.flow_prefetch_batch_size,
            num_skip_edges=args.num_sorted_edge_trials,
            num_random_edges=args.num_random_edges,
            epsilon_random=args.epsilon_random,
            udf_seed=seed,
            goal_sampling_probability=args.goal_sampling_probability,
            debug_flag=False,
            print_logs=False,
        )
        planner.set_profile_enabled(True)
        started = time.perf_counter()
        planner.plan_path()
        diagnostic_wall_time = time.perf_counter() - started

        node_records = []
        for node_id, item in planner.tree.nodes(data=True):
            node = item["value"]
            distance = agent.get_goal_distance(node.state, goal)
            if distance > near_goal_radius:
                continue
            mask = node.edge_bundle_mask
            bundle = node.flow_edge_bundle
            node_records.append(
                {
                    "node_id": int(node_id),
                    "parent_goal_distance": float(distance),
                    "selection_count": planner.node_selection_counts.get(
                        int(node_id), 0
                    ),
                    "bundle_generated": bundle is not None,
                    "bundle_size": 0 if mask is None else int(len(mask)),
                    "attempted_edges": 0 if mask is None else int(np.count_nonzero(mask)),
                    "remaining_edges": 0 if mask is None else int(np.count_nonzero(~mask)),
                    "all_edges_expanded": bool(
                        mask is not None and len(mask) and np.all(mask)
                    ),
                    "random_fallbacks": planner.node_random_fallback_counts.get(
                        int(node_id), 0
                    ),
                    "fallbacks_after_all_edges_expanded": (
                        planner.node_exhausted_fallback_counts.get(int(node_id), 0)
                    ),
                    "node": node,
                }
            )
        node_records.sort(key=lambda record: record["parent_goal_distance"])
        audited_nodes = [
            record for record in node_records if record["bundle_generated"]
        ][: args.max_near_goal_nodes]

        edge_records: list[dict[str, object]] = []
        for node_record in audited_nodes:
            node = node_record.pop("node")
            bundle = node.flow_edge_bundle
            mask = node.edge_bundle_mask
            for edge_index, actions in enumerate(bundle.diagnostic_action_sequences):
                result = audit_edge(
                    agent=agent,
                    parent_state=node.state,
                    actions=actions,
                    action_dt=bundle.action_dt,
                    goal=goal,
                    goal_radius=args.goal_radius,
                )
                edge_records.append(
                    {
                        "node_id": node_record["node_id"],
                        "parent_goal_distance": node_record["parent_goal_distance"],
                        "edge_index": edge_index,
                        "attempted_during_planning": bool(mask[edge_index]),
                        "predicted_endpoint_goal_distance": agent.get_goal_distance(
                            bundle.final_states[edge_index], goal
                        ),
                        "duration_seconds": float(len(actions) * bundle.action_dt),
                        **result,
                    }
                )
        for record in node_records:
            record.pop("node", None)

        edge_fields = list(edge_records[0]) if edge_records else []
        if edge_records:
            with (output_dir / "edge_audit.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=edge_fields)
                writer.writeheader()
                writer.writerows(edge_records)
        node_fields = list(node_records[0]) if node_records else []
        if node_records:
            with (output_dir / "near_goal_nodes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=node_fields)
                writer.writeheader()
                writer.writerows(node_records)

        distances = np.asarray(
            [
                agent.get_goal_distance(item["value"].state, goal)
                for _, item in planner.tree.nodes(data=True)
            ],
            dtype=np.float64,
        )
        best_node_id = int(np.argmin(distances))
        path_node_id = int(planner.goal_node_id) if planner.path_found else best_node_id
        dense_path = dense_path_to_node(planner, path_node_id)
        path_motion_time = (
            path_motion_time_from_states(dense_path, generator.dt)
            if planner.path_found
            else None
        )
        summary = {
            "problem_id": args.problem_id,
            "success": bool(planner.path_found),
            "planning_time_seconds": float(planner.last_plan_wall_time),
            "diagnostic_wall_time_seconds": float(diagnostic_wall_time),
            "path_motion_time_seconds": path_motion_time,
            "path_quality_definition": (
                "physical execution time through the first goal-reaching waypoint"
            ),
            "goal_radius": args.goal_radius,
            "near_goal_radius": near_goal_radius,
            "best_tree_goal_distance": float(distances.min()),
            "tree_nodes": int(planner.num_rrt_nodes()),
            "flow_bundle_size": int(generator.set_size),
            "near_goal_tree_nodes": len(node_records),
            "near_goal_nodes_with_generated_bundle": int(
                sum(bool(record["bundle_generated"]) for record in node_records)
            ),
            "near_goal_nodes_selected_for_expansion": int(
                sum(int(record["selection_count"]) > 0 for record in node_records)
            ),
            "near_goal_nodes_with_all_edges_expanded": int(
                sum(bool(record["all_edges_expanded"]) for record in node_records)
            ),
            "near_goal_fallbacks_after_all_edges_expanded": int(
                sum(
                    int(record["fallbacks_after_all_edges_expanded"])
                    for record in node_records
                )
            ),
            "audited_near_goal_nodes": len(audited_nodes),
            "audited_edges": len(edge_records),
            "audited_edge_classifications": summarize_counts(
                edge_records, "classification"
            ),
            "audited_predicted_goal_endpoints": int(
                sum(
                    float(record["predicted_endpoint_goal_distance"])
                    <= args.goal_radius
                    for record in edge_records
                )
            ),
            "audited_actual_goal_crossings": int(
                sum(int(record["first_goal_step"]) >= 0 for record in edge_records)
            ),
            "configuration": {
                "checkpoint": str(args.checkpoint.resolve()),
                "dataset": str(args.dataset.resolve()),
                "problems": str(args.problems.resolve()),
                "urdf": str(args.urdf.resolve()),
                "device": str(generator.device),
                "seed": args.seed,
                "planner_seed": seed,
                "goal_sampling_probability": args.goal_sampling_probability,
                "epsilon_random": args.epsilon_random,
                "num_sorted_edge_trials": args.num_sorted_edge_trials,
                "num_random_edges": args.num_random_edges,
                "flow_prefetch_batch_size": args.flow_prefetch_batch_size,
                "integration_dt": generator.dt,
            },
            "interpretation": {
                "attempted_edges": (
                    "mask entries set true by real planner trials; generated but "
                    "unselected candidates remain false"
                ),
                "all_edges_expanded": (
                    "all 32 mask entries are true; later selections cannot use a "
                    "Flow edge and go directly to random-control fallback"
                ),
                "offline_edge_audit": (
                    "replays retained controls from the exact generated bundles; "
                    "it does not alter the completed planner run"
                ),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "best_or_solution_path.npz",
            states=dense_path,
            start=start,
            goal=goal,
            success=np.asarray(planner.path_found),
            integration_dt=np.asarray(generator.dt),
            path_motion_time_seconds=np.asarray(
                np.nan if path_motion_time is None else path_motion_time
            ),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"diagnostics saved to {output_dir}")
    finally:
        checker.close()


if __name__ == "__main__":
    main()
