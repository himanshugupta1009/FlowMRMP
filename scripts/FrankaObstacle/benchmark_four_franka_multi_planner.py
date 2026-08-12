#!/usr/bin/env python3
"""Run the resumable 360-trial four-Franka coordination benchmark.

The matrix is three fresh difficulty tiers x 20 evaluation seeds x
{Prioritized, CRRT, KCBS} x {VanillaRRT, FlowEBRRT}.  Every four-arm trial has
one 30-second total wall-clock planning budget.  Planning and auditing use only
cuRobo plus the exact 295-sphere world model; PyBullet is not imported.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
OBSTACLE_SCRIPTS = Path(__file__).resolve().parent
FRANKA_TESTING = MAIN_SCRIPTS / "FrankaTesting"
FLOW_SOURCE = REPOSITORY_ROOT / "src"
for module_path in (MAIN_SCRIPTS, OBSTACLE_SCRIPTS, FRANKA_TESTING, FLOW_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from benchmark_franka_vanilla_rrt import sample_dataset_state_bank  # noqa: E402
from flow_eb_rrt import FrankaFlowEdgeGenerator  # noqa: E402
from four_franka_multi_planner import (  # noqa: E402
    audit_synchronized_paths,
    make_crrt,
    make_kcbs,
    make_single_arm_planner,
    path_motion_time,
)
from franka_prioritized_adapter import (  # noqa: E402
    FrankaPrioritizedCollisionAdapter,
    ROBOT_NAMES,
)
from prioritized_planning import PrioritizedPlanning  # noqa: E402

DEFAULT_DATASET = Path("C:/Users/sodan/Downloads/dataset200k.h5")
DEFAULT_PROBLEMS = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" /
    "four_franka_multi_planner_problems_final_v3_20260812" /
    "four_franka_easy_medium_hard.npz"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "checkpoints" / "franka_edge_flow" /
    "franka_edge_flow_k32_200k_pool128_n350000_max50_fullvalid_mps_v1" /
    "best_inference.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" /
    "four_franka_multi_planner_20seeds_final_v3_20260812"
)
COORDINATIONS = ("prioritized", "crrt", "kcbs")
BASES = ("vanilla", "flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--planning-time", type=float, default=30.0)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument("--state-bank-size", type=int, default=100_000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026082200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coordinations", default=",".join(COORDINATIONS))
    parser.add_argument("--bases", default=",".join(BASES))
    parser.add_argument("--problem-ids", default="0,1,2")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_values(value, allowed):
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not values or any(item not in allowed for item in values):
        raise ValueError(f"Expected a subset of {allowed}, got {values}")
    return values


def load_problems(path: Path, ids):
    with np.load(path, allow_pickle=False) as data:
        all_ids = data["problem_ids"].astype(int)
        result = []
        for problem_id in ids:
            matches = np.flatnonzero(all_ids == problem_id)
            if len(matches) != 1:
                raise ValueError(f"Problem {problem_id} missing or duplicated")
            index = int(matches[0])
            result.append({
                "problem_id": int(problem_id),
                "difficulty": str(data["names"][index]),
                "starts": data["starts"][index].astype(np.float64),
                "goals": data["goals"][index].astype(np.float64),
            })
    return result


def reset_flow_generator(generator, seed):
    if generator is None:
        return
    generator.seed = int(seed)
    generator.torch_generator.manual_seed(int(seed))
    for key in list(generator.profile):
        generator.profile[key] = 0.0 if key.endswith("_s") else 0


def safe_paths(planners):
    paths = []
    for planner in planners:
        if not planner.path_found:
            return []
        paths.append(planner.get_high_resolution_path_numpy_array())
    return paths


def run_trial(*, coordination, base, scenario, trial_seed, adapter, generator, args):
    adapter.reset()
    reset_flow_generator(generator, trial_seed + 991)
    starts = scenario["starts"]
    goals = scenario["goals"]
    start_valid, start_audit = adapter.configuration_set_valid(starts)
    goal_valid, goal_audit = adapter.configuration_set_valid(goals)
    if not start_valid or not goal_valid:
        raise RuntimeError(f"Invalid endpoint: start={start_audit}, goal={goal_audit}")

    started = time.perf_counter()
    generations = []
    planner_profile = {}
    high_level_nodes = 0
    low_level_planners = []

    if coordination == "prioritized":
        low_level_planners = [
            make_single_arm_planner(
                base=base,
                agent=adapter.agents[index],
                start=starts[index],
                goal=goals[index],
                seed=trial_seed + index,
                planning_time=args.planning_time,
                goal_radius=args.goal_radius,
                generator=generator,
            )
            for index in range(4)
        ]
        solved, _, path_cost = PrioritizedPlanning.plan_multi(
            planners=low_level_planners,
            planning_time=args.planning_time,
            print_logs=False,
            dynamic_obstacle_adapter=adapter,
        )
        paths = safe_paths(low_level_planners) if solved else []
        generations = adapter.stage_records
    elif coordination == "crrt":
        planner = make_crrt(
            base=base,
            adapter=adapter,
            starts=starts,
            goals=goals,
            seed=trial_seed,
            planning_time=args.planning_time,
            goal_radius=args.goal_radius,
            generator=generator,
        )
        planner.plan_path()
        solved = bool(planner.path_found)
        path_cost = float(np.sum(planner.path_cost)) if solved else np.inf
        paths = planner.get_high_resolution_path_numpy_array() if solved else []
        high_level_nodes = len(planner.tree.nodes)
        low_level_planners = [planner]
        planner_profile = getattr(planner, "profile", {})
    elif coordination == "kcbs":
        planner = make_kcbs(
            base=base,
            adapter=adapter,
            starts=starts,
            goals=goals,
            seed=trial_seed,
            planning_time=args.planning_time,
            goal_radius=args.goal_radius,
            generator=generator,
        )
        solved, returned_paths, path_cost, _ = planner.plan_multi_agent_paths()
        paths = list(returned_paths) if solved else []
        high_level_nodes = int(planner.cbs_node_count)
        low_level_planners = list(planner.low_level_planners)
        planner_profile = {
            "cbs_nodes": int(planner.cbs_node_count),
            "conflict_counts": planner.collision_count.tolist(),
        }
    else:
        raise ValueError(coordination)
    wall_time = time.perf_counter() - started

    audit_passed = False
    audit = {"reason": "planner_failed"}
    if solved and len(paths) == 4:
        audit_passed, audit = audit_synchronized_paths(adapter, paths, 0.02)
    success = bool(solved and audit_passed)
    if solved and not audit_passed:
        audit["planner_claimed_success"] = True

    tree_nodes = 0
    iterations = 0
    for planner in low_level_planners:
        tree_nodes += len(getattr(planner, "tree", {}).nodes) if hasattr(
            getattr(planner, "tree", None), "nodes") else 0
        iterations += int(getattr(planner, "last_plan_iterations", 0))
    checked_waypoints = int(sum(agent.checked_waypoints for agent in adapter.agents))
    path_times = [path_motion_time(path, 0.02) for path in paths] if success else []
    return {
        "coordination": coordination,
        "base_planner": base,
        "problem_id": scenario["problem_id"],
        "difficulty": scenario["difficulty"],
        "trial_seed": trial_seed,
        "success": success,
        "planner_solved": bool(solved),
        "audit_passed": bool(audit_passed),
        "planning_time_seconds": float(wall_time),
        "path_makespan_seconds": max(path_times) if path_times else None,
        "path_mean_agent_time_seconds": float(np.mean(path_times)) if path_times else None,
        "path_cost": float(path_cost) if np.isfinite(path_cost) else None,
        "tree_nodes": int(tree_nodes),
        "high_level_nodes": int(high_level_nodes),
        "iterations": int(iterations),
        "checked_waypoints": checked_waypoints,
        "per_agent_path_times_seconds": path_times,
        "audit": audit,
        "generations": generations,
        "planner_profile": planner_profile,
        "flow_generator_profile": dict(generator.profile) if base == "flow" else {},
    }, paths


def trial_stem(record):
    return (
        f"{record['coordination']}_{record['base_planner']}_"
        f"p{record['problem_id']}_{record['difficulty']}_s{record['trial_seed']}"
    )


def load_records(trial_dir):
    records = []
    for path in sorted(trial_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return records


def mean_median(values):
    if not values:
        return None, None
    return float(statistics.fmean(values)), float(statistics.median(values))


def aggregate(records):
    rows = []
    for coordination in COORDINATIONS:
        for base in BASES:
            for difficulty in ("easy", "medium", "hard", "all"):
                selected = [record for record in records
                            if record.get("coordination") == coordination
                            and record.get("base_planner") == base
                            and (difficulty == "all" or record.get("difficulty") == difficulty)
                            and "error" not in record]
                if not selected:
                    continue
                successful = [record for record in selected if record.get("success")]
                planning_mean, planning_median = mean_median([
                    record["planning_time_seconds"] for record in successful])
                path_mean, path_median = mean_median([
                    record["path_makespan_seconds"] for record in successful])
                rows.append({
                    "coordination": coordination,
                    "base_planner": base,
                    "difficulty": difficulty,
                    "trials": len(selected),
                    "successes": len(successful),
                    "success_percent": 100.0 * len(successful) / len(selected),
                    "successful_planning_mean_seconds": planning_mean,
                    "successful_planning_median_seconds": planning_median,
                    "successful_path_makespan_mean_seconds": path_mean,
                    "successful_path_makespan_median_seconds": path_median,
                    "all_trial_tree_nodes_total": sum(r["tree_nodes"] for r in selected),
                    "all_trial_checked_waypoints_total": sum(r["checked_waypoints"] for r in selected),
                    "successful_tree_nodes_mean": mean_median([
                        r["tree_nodes"] for r in successful])[0],
                    "successful_checked_waypoints_mean": mean_median([
                        r["checked_waypoints"] for r in successful])[0],
                })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.problems = args.problems.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    for path in (args.dataset, args.problems, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    coordinations = parse_values(args.coordinations, COORDINATIONS)
    bases = parse_values(args.bases, BASES)
    problem_ids = tuple(int(value) for value in args.problem_ids.split(","))
    scenarios = load_problems(args.problems, problem_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = args.output_dir / "trials"
    trajectory_dir = args.output_dir / "trajectories"
    trial_dir.mkdir(exist_ok=True)
    trajectory_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed_base - 1)
    state_bank, _, _ = sample_dataset_state_bank(
        args.dataset, args.state_bank_size, rng)
    adapter = FrankaPrioritizedCollisionAdapter(
        state_bank=state_bank, device=args.device, acceleration_scale=0.5)
    generator = None
    if "flow" in bases:
        generator = FrankaFlowEdgeGenerator(
            checkpoint_path=args.checkpoint,
            device="cuda",
            sample_steps=16,
            clamp_outputs=True,
            seed=args.seed_base,
        )
        generator.set_profile_enabled(True)

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planning_time_seconds_total_per_four_arm_trial": args.planning_time,
        "goal_radius": args.goal_radius,
        "num_evaluation_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "coordinations": coordinations,
        "base_planners": bases,
        "problem_ids": problem_ids,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "problems": str(args.problems),
        "problems_sha256": sha256(args.problems),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "flow_near_goal_policy": {
            "activation": "goal-biased query with parent distance <= 1.25 * goal radius",
            "independent_bundles": 4,
            "edges_per_bundle": 32,
            "total_candidate_edges": 128,
            "ranking": "physical intermediate 7D q distance",
            "goal_candidate_order": "earliest goal-region entry first",
            "fallback": "original 32-edge ranking, then short random control",
        },
        "collision_backend": "cuRobo API + exact batched 295-sphere world overlap",
        "pybullet_used": False,
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8")

    try:
        total = len(coordinations) * len(bases) * len(scenarios) * args.num_seeds
        completed = 0
        for coordination in coordinations:
            for base in bases:
                for scenario in scenarios:
                    for seed_index in range(args.num_seeds):
                        trial_seed = args.seed_base + seed_index
                        identity = {
                            "coordination": coordination,
                            "base_planner": base,
                            "problem_id": scenario["problem_id"],
                            "difficulty": scenario["difficulty"],
                            "trial_seed": trial_seed,
                        }
                        stem = trial_stem(identity)
                        record_path = trial_dir / f"{stem}.json"
                        completed += 1
                        if record_path.is_file():
                            print(f"[{completed}/{total}] resume-skip {stem}", flush=True)
                            continue
                        print(f"[{completed}/{total}] start {stem}", flush=True)
                        try:
                            record, paths = run_trial(
                                coordination=coordination,
                                base=base,
                                scenario=scenario,
                                trial_seed=trial_seed,
                                adapter=adapter,
                                generator=generator if base == "flow" else None,
                                args=args,
                            )
                            if record["success"]:
                                np.savez_compressed(
                                    trajectory_dir / f"{stem}.npz",
                                    integration_dt=np.asarray(0.02),
                                    **{f"robot_{index}_states": path
                                       for index, path in enumerate(paths)},
                                )
                        except Exception as error:
                            record = {
                                **identity,
                                "success": False,
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(),
                            }
                            if args.fail_fast:
                                record_path.write_text(
                                    json.dumps(record, indent=2), encoding="utf-8")
                                raise
                        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
                        record_path.write_text(
                            json.dumps(record, indent=2, default=str), encoding="utf-8")
                        records = load_records(trial_dir)
                        rows = aggregate(records)
                        write_csv(args.output_dir / "aggregate.partial.csv", rows)
                        print(
                            f"[{completed}/{total}] done {stem} success="
                            f"{record.get('success')} wall="
                            f"{record.get('planning_time_seconds', 'error')}", flush=True)
    finally:
        adapter.close()

    records = load_records(trial_dir)
    rows = aggregate(records)
    write_csv(args.output_dir / "aggregate.csv", rows)
    write_csv(args.output_dir / "trials.csv", [
        {key: record.get(key) for key in (
            "coordination", "base_planner", "problem_id", "difficulty",
            "trial_seed", "success", "planner_solved", "audit_passed",
            "planning_time_seconds", "path_makespan_seconds",
            "path_mean_agent_time_seconds", "path_cost", "tree_nodes",
            "high_level_nodes", "iterations", "checked_waypoints", "error")}
        for record in records
    ])
    (args.output_dir / "summary.json").write_text(json.dumps({
        "configuration": configuration,
        "trial_count": len(records),
        "aggregate": rows,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "aggregate.partial.csv").unlink(missing_ok=True)
    print(args.output_dir / "aggregate.csv")


if __name__ == "__main__":
    main()
