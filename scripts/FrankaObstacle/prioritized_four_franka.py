#!/usr/bin/env python3
"""Run prioritized VanillaRRT and FlowEBRRT on four-Franka scenarios."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
OBSTACLE_SCRIPTS = Path(__file__).resolve().parent
FRANKA_TESTING = MAIN_SCRIPTS / "FrankaTesting"
MRMP_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
FLOW_SOURCE = REPOSITORY_ROOT / "src"
for module_path in (
    MAIN_SCRIPTS,
    OBSTACLE_SCRIPTS,
    FRANKA_TESTING,
    MRMP_SOURCE,
    FLOW_SOURCE,
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import EmptyFrankaEnvironment  # noqa: E402
from benchmark_franka_vanilla_rrt import (  # noqa: E402
    file_sha256,
    sample_dataset_state_bank,
)
from flow_eb_rrt import FlowEBRRT, FrankaFlowEdgeGenerator  # noqa: E402
from franka_prioritized_adapter import (  # noqa: E402
    FrankaPrioritizedCollisionAdapter,
    ROBOT_NAMES,
)
from prioritized_planning import PrioritizedPlanning  # noqa: E402
from rrt import RRT  # noqa: E402


DEFAULT_DATASET = Path("C:/Users/sodan/Downloads/dataset200k.h5")
DEFAULT_PROBLEMS = (
    REPOSITORY_ROOT
    / "results"
    / "franka_obstacle"
    / "four_franka_difficult_problems"
    / "four_franka_difficult_problems.npz"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "franka_edge_flow"
    / "franka_edge_flow_k32_200k_pool128_n350000_max50_fullvalid_mps_v1"
    / "best_inference.pt"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" / "four_franka_prioritized"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--problem-ids", default="0,1,2")
    parser.add_argument("--algorithms", default="vanilla,flow")
    parser.add_argument("--priority-order", default="0,1,2,3")
    parser.add_argument("--planning-time", type=float, default=120.0)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument("--state-bank-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dynamic-clearance", type=float, default=0.0)
    parser.add_argument("--run-name")
    return parser.parse_args()


def parse_int_list(value: str, *, expected_unique: int | None = None) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if expected_unique is not None and sorted(result) != list(range(expected_unique)):
        raise ValueError(
            f"Expected a permutation of 0..{expected_unique - 1}, got {result}"
        )
    return result


def load_scenarios(path: Path, selected_ids: list[int]) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        problem_ids = data["problem_ids"].astype(int)
        selected = []
        for problem_id in selected_ids:
            matches = np.flatnonzero(problem_ids == problem_id)
            if matches.size != 1:
                raise ValueError(f"Problem ID {problem_id} not found exactly once")
            index = int(matches[0])
            selected.append(
                {
                    "problem_id": problem_id,
                    "name": str(data["names"][index]),
                    "starts": data["starts"][index].astype(np.float64),
                    "goals": data["goals"][index].astype(np.float64),
                }
            )
    return selected


def dense_path_to_node(planner, node_id: int) -> np.ndarray:
    reverse_ids = []
    current_id = int(node_id)
    while current_id != 0:
        reverse_ids.append(current_id)
        current_id = int(planner.tree.nodes[current_id]["value"].parent_id)
    pieces = [np.asarray(planner.start, dtype=np.float64)[None, :]]
    for current_id in reversed(reverse_ids):
        pieces.append(
            np.asarray(
                planner.tree.nodes[current_id]["value"].path_from_parent,
                dtype=np.float64,
            )
        )
    return np.concatenate(pieces, axis=0)


def best_available_path(planner) -> tuple[np.ndarray, bool, bool]:
    if planner.path_found:
        return planner.get_high_resolution_path_numpy_array(), True, True
    if len(planner.tree.nodes) == 0:
        return np.asarray(planner.start, dtype=np.float64)[None, :], False, False
    distances = [
        planner.agent.get_goal_distance(node["value"].state, planner.goal)
        for _, node in planner.tree.nodes(data=True)
    ]
    best_node = int(np.argmin(np.asarray(distances, dtype=np.float64)))
    return dense_path_to_node(planner, best_node), False, True


def make_vanilla_planner(
    *, agent, start: np.ndarray, goal: np.ndarray, seed: int, args: argparse.Namespace
):
    return RRT(
        start=start,
        goal=goal,
        goal_radius=args.goal_radius,
        env=EmptyFrankaEnvironment(),
        agent=agent,
        use_fixed_sampling_time=False,
        sampling_time_step=1.0,
        minimum_time_step=0.02,
        max_iter=10_000_000,
        planning_time=args.planning_time,
        num_extension_trials=16,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=seed,
        goal_sampling_probability=0.30,
        dynamic_agent_clearance=args.dynamic_clearance,
        debug_flag=False,
        print_logs=False,
    )


def make_flow_planner(
    *,
    agent,
    start: np.ndarray,
    goal: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    generator: FrankaFlowEdgeGenerator,
):
    planner = FlowEBRRT(
        start=start,
        goal=goal,
        goal_radius=args.goal_radius,
        env=EmptyFrankaEnvironment(),
        agent=agent,
        flow_edge_generator=generator,
        use_fixed_sampling_time=False,
        sampling_time_step=0.30,
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
        flow_prefetch_batch_size=16,
        minimum_sequence_prefix_steps=5,
        truncate_sequence_to_target=False,
        num_skip_edges=32,
        num_random_edges=1,
        epsilon_random=0.05,
        udf_seed=seed,
        goal_sampling_probability=0.30,
        dynamic_agent_clearance=args.dynamic_clearance,
        debug_flag=False,
        print_logs=False,
    )
    planner.set_profile_enabled(True)
    return planner


def write_stage_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "generation",
        "robot_index",
        "robot_name",
        "status",
        "success",
        "planning_time_seconds",
        "iterations",
        "nodes",
        "path_motion_time_seconds",
        "path_cost",
        "checked_waypoints",
        "limit_rejections",
        "self_or_static_collision_rejections",
        "inter_robot_collision_rejections",
        "collision_profile",
        "planner_profile",
        "flow_generator_profile",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for nested in (
                "collision_profile",
                "planner_profile",
                "flow_generator_profile",
            ):
                row[nested] = json.dumps(row[nested] or {}, sort_keys=True)
            writer.writerow(row)


def save_paths(
    path: Path,
    planners_by_robot: dict[int, object],
    *,
    integration_dt: float,
) -> list[dict[str, object]]:
    arrays: dict[str, np.ndarray] = {
        "integration_dt": np.asarray(integration_dt, dtype=np.float64),
        "robot_names": np.asarray(ROBOT_NAMES),
    }
    path_records = []
    for robot_index in range(len(ROBOT_NAMES)):
        planner = planners_by_robot[robot_index]
        dense, success, attempted = best_available_path(planner)
        arrays[f"robot_{robot_index}_states"] = dense
        arrays[f"robot_{robot_index}_success"] = np.asarray(success)
        arrays[f"robot_{robot_index}_attempted"] = np.asarray(attempted)
        path_records.append(
            {
                "robot_index": robot_index,
                "robot_name": ROBOT_NAMES[robot_index],
                "success": success,
                "attempted": attempted,
                "saved_states": int(len(dense)),
            }
        )
    np.savez_compressed(path, **arrays)
    return path_records


def complete_generation_records(
    stage_records: list[dict[str, object]], priority_order: list[int]
) -> list[dict[str, object]]:
    complete = [dict(record, status="success" if record["success"] else "failed") for record in stage_records]
    for generation in range(len(complete), len(priority_order)):
        robot_index = priority_order[generation]
        complete.append(
            {
                "generation": generation,
                "robot_index": robot_index,
                "robot_name": ROBOT_NAMES[robot_index],
                "status": "not_attempted_after_prior_failure",
                "success": False,
            }
        )
    return complete


def run_scenario(
    scenario: dict[str, object],
    *,
    algorithm: str,
    state_bank: np.ndarray,
    priority_order: list[int],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = FrankaPrioritizedCollisionAdapter(
        state_bank=state_bank,
        device=args.device,
        dynamic_clearance=args.dynamic_clearance,
        acceleration_scale=0.5,
    )
    generator = None
    try:
        start_valid, start_audit = adapter.configuration_set_valid(scenario["starts"])
        goal_valid, goal_audit = adapter.configuration_set_valid(scenario["goals"])
        if not start_valid or not goal_valid:
            raise RuntimeError(
                "Scenario endpoint is invalid under the active 295-sphere model: "
                f"start={start_audit}, goal={goal_audit}"
            )

        if algorithm == "flow":
            generator = FrankaFlowEdgeGenerator(
                checkpoint_path=args.checkpoint,
                device="cuda",
                sample_steps=16,
                clamp_outputs=True,
                seed=args.seed + 10_000 * int(scenario["problem_id"]),
            )

        planners = []
        planners_by_robot = {}
        for generation, robot_index in enumerate(priority_order):
            seed = (
                args.seed
                + 100_000 * int(scenario["problem_id"])
                + 1_000 * (0 if algorithm == "vanilla" else 1)
                + robot_index
            )
            kwargs = {
                "agent": adapter.agents[robot_index],
                "start": np.asarray(scenario["starts"])[robot_index],
                "goal": np.asarray(scenario["goals"])[robot_index],
                "seed": seed,
                "args": args,
            }
            planner = (
                make_vanilla_planner(**kwargs)
                if algorithm == "vanilla"
                else make_flow_planner(**kwargs, generator=generator)
            )
            planners.append(planner)
            planners_by_robot[robot_index] = planner

        started = time.perf_counter()
        solved, reported_time, total_cost = PrioritizedPlanning.plan_multi(
            planners=planners,
            planning_time=args.planning_time,
            print_logs=True,
            dynamic_obstacle_adapter=adapter,
        )
        wall_time = time.perf_counter() - started
        integration_dt = float(planners[0].minimum_time_step)
        path_records = save_paths(
            output_dir / "trajectories.npz",
            planners_by_robot,
            integration_dt=integration_dt,
        )
        generations = complete_generation_records(
            adapter.stage_records, priority_order
        )
        write_stage_csv(output_dir / "generations.csv", generations)
        result = {
            "problem_id": int(scenario["problem_id"]),
            "scenario_name": str(scenario["name"]),
            "algorithm": "VanillaRRT" if algorithm == "vanilla" else "FlowEBRRT",
            "solved": bool(solved),
            "priority_order": priority_order,
            "priority_names": [ROBOT_NAMES[index] for index in priority_order],
            "planning_budget_seconds": float(args.planning_time),
            "reported_planning_time_seconds": float(reported_time),
            "wall_time_seconds": float(wall_time),
            "total_path_cost": float(total_cost) if np.isfinite(total_cost) else None,
            "goal_radius": float(args.goal_radius),
            "integration_dt": integration_dt,
            "dynamic_clearance": float(args.dynamic_clearance),
            "collision_model": {
                "backend": "cuRobo API plus batched PyTorch CUDA sphere-pair tests",
                "sphere_count_per_robot": 295,
                "world_from_base": True,
                "pybullet_used_for_planning_or_audit": False,
            },
            "endpoint_audit": {"start": start_audit, "goal": goal_audit},
            "generations": generations,
            "paths": path_records,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": file_sha256(args.dataset),
            "problems": str(args.problems.resolve()),
            "problems_sha256": file_sha256(args.problems),
            "checkpoint": str(args.checkpoint.resolve()) if algorithm == "flow" else None,
            "checkpoint_sha256": file_sha256(args.checkpoint) if algorithm == "flow" else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
    finally:
        adapter.close()


def write_run_summary(path: Path, results: list[dict[str, object]], args) -> None:
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "configuration": {
            "planning_time_seconds": args.planning_time,
            "goal_radius": args.goal_radius,
            "state_bank_size": args.state_bank_size,
            "seed": args.seed,
            "device": args.device,
            "dynamic_clearance": args.dynamic_clearance,
        },
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.problems = args.problems.resolve()
    args.checkpoint = args.checkpoint.resolve()
    for required in (args.dataset, args.problems, args.checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.planning_time <= 0.0:
        raise ValueError("planning-time must be positive")
    selected_ids = parse_int_list(args.problem_ids)
    priority_order = parse_int_list(args.priority_order, expected_unique=4)
    algorithms = [value.strip().lower() for value in args.algorithms.split(",")]
    if not algorithms or any(value not in {"vanilla", "flow"} for value in algorithms):
        raise ValueError("algorithms must contain vanilla and/or flow")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"sampling {args.state_bank_size:,} states from {args.dataset}", flush=True)
    state_bank, _, _ = sample_dataset_state_bank(
        args.dataset,
        args.state_bank_size,
        np.random.default_rng(args.seed),
    )
    scenarios = load_scenarios(args.problems, selected_ids)
    results = []
    for scenario in scenarios:
        for algorithm in algorithms:
            output_dir = (
                run_dir
                / f"problem_{int(scenario['problem_id']):02d}_{scenario['name']}"
                / algorithm
            )
            print(
                f"running problem={scenario['problem_id']} algorithm={algorithm} "
                f"priority={priority_order}",
                flush=True,
            )
            result = run_scenario(
                scenario,
                algorithm=algorithm,
                state_bank=state_bank,
                priority_order=priority_order,
                output_dir=output_dir,
                args=args,
            )
            results.append(result)
            write_run_summary(run_dir / "run_summary.partial.json", results, args)
            print(
                f"completed problem={scenario['problem_id']} algorithm={algorithm} "
                f"solved={result['solved']} wall={result['wall_time_seconds']:.3f}s",
                flush=True,
            )
    write_run_summary(run_dir / "run_summary.json", results, args)
    (run_dir / "run_summary.partial.json").unlink(missing_ok=True)
    print(f"saved prioritized run: {run_dir}")


if __name__ == "__main__":
    main()
