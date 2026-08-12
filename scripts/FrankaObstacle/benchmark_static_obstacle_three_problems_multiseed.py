#!/usr/bin/env python3
"""Benchmark easy, medium, and hard cuboid problems over 20 shared seeds.

The three A-B pairs are generated without using either planner: their endpoints
are valid, their direct joint interpolation is blocked only by the static
cuboid, and a sampled two-segment collision-free geometric route exists.  Each
pair is then repeated with the same seed set for the current VanillaRRT and
FlowEBRRT implementations.  Planning and validity checking use MorphIt/cuRobo;
PyBullet is not used by this benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
FRANKA_TESTING = MAIN_SCRIPTS / "FrankaTesting"
CORE_RRT_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
for module_path in (MAIN_SCRIPTS, FRANKA_TESTING, CORE_RRT_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import benchmark_franka_flow_eb_rrt as flow_benchmark  # noqa: E402
import benchmark_franka_vanilla_rrt as vanilla_benchmark  # noqa: E402
from FrankaPanda import DEFAULT_CUROBO_CONFIG, FrankaCuroboCollisionChecker  # noqa: E402
from franka_paths import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_RAW_TRAJECTORY_DATASET,
    DEFAULT_URDF,
)
from static_obstacle_test import (  # noqa: E402
    create_problem_set,
    scene_model,
)

DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "franka_obstacle"
    / "static_cuboid_easy_medium_hard_20seeds_20260812"
)
DIFFICULTIES = ("easy", "medium", "hard")
TRIALS_PER_PROBLEM = 20
PROBLEM_SEED = 20260812
STATE_BANK_SEED = 20260721
FIRST_TRIAL_SEED = 2026081200
STATE_BANK_SIZE = 100_000
PLANNING_TIME_SECONDS = 30.0
GOAL_RADIUS = 0.40


def parse_args() -> argparse.Namespace:
    """Parse paths and reproducibility controls for the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RAW_TRAJECTORY_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--problem-seed", type=int, default=PROBLEM_SEED)
    parser.add_argument("--state-bank-seed", type=int, default=STATE_BANK_SEED)
    parser.add_argument("--first-trial-seed", type=int, default=FIRST_TRIAL_SEED)
    parser.add_argument("--trials-per-problem", type=int, default=TRIALS_PER_PROBLEM)
    parser.add_argument("--planning-time", type=float, default=PLANNING_TIME_SECONDS)
    parser.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
    parser.add_argument("--state-bank-size", type=int, default=STATE_BANK_SIZE)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Add audit metadata and finalize an already completed 120-trial run.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    """Write deterministic, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def expand_problem_trials(
    base_path: Path,
    output_path: Path,
    *,
    trials_per_problem: int,
    first_trial_seed: int,
) -> None:
    """Repeat each labeled problem with an identical seed set."""
    with np.load(base_path, allow_pickle=False) as source:
        labels = np.asarray(source["difficulty_tier"]).astype(str)
        source_ids = []
        for difficulty in DIFFICULTIES:
            matches = np.flatnonzero(labels == difficulty)
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {difficulty} problem, found {len(matches)}"
                )
            source_ids.append(int(matches[0]))

        expanded_source_ids = np.repeat(source_ids, trials_per_problem)
        trial_indices = np.tile(np.arange(trials_per_problem), len(DIFFICULTIES))
        seeds = first_trial_seed + trial_indices
        copied_fields = (
            "starts",
            "goals",
            "start_trajectory",
            "start_timestep",
            "goal_trajectory",
            "goal_timestep",
            "difficulty_tier",
            "witness_duration_seconds",
            "witness_steps",
            "witness_initial_normalized_distance",
            "direct_first_collision_index",
            "geometric_via_configurations",
        )
        payload = {
            key: np.asarray(source[key])[expanded_source_ids] for key in copied_fields
        }
        payload.update(
            {
                "format_name": np.asarray("franka_static_cuboid_multiseed_trials"),
                "format_version": np.asarray(1, dtype=np.int32),
                "created_utc": np.asarray(datetime.now(timezone.utc).isoformat()),
                "source_problem_id": expanded_source_ids.astype(np.int32),
                "trial_index": trial_indices.astype(np.int32),
                "benchmark_seed": seeds.astype(np.int64),
                "obstacle_name": np.asarray(source["obstacle_name"]),
                "obstacle_pose": np.asarray(source["obstacle_pose"]),
                "obstacle_dims": np.asarray(source["obstacle_dims"]),
            }
        )
    np.savez_compressed(output_path, **payload)


def load_trial_problems(path: Path) -> list[dict[str, object]]:
    """Load planner inputs plus their repeated-trial provenance."""
    with np.load(path, allow_pickle=False) as source:
        count = len(source["starts"])
        trial_indices = np.asarray(source["trial_index"], dtype=np.int32)
        benchmark_seeds = np.asarray(source["benchmark_seed"], dtype=np.int64)
        source_ids = np.asarray(source["source_problem_id"], dtype=np.int32)
    problems = vanilla_benchmark.load_fixed_problems(path, count)
    for index, problem in enumerate(problems):
        problem["trial_index"] = int(trial_indices[index])
        problem["benchmark_seed"] = int(benchmark_seeds[index])
        problem["source_problem_id"] = int(source_ids[index])
    return problems


def prepare_result_directory(path: Path, problems_path: Path) -> None:
    """Create an immutable planner output directory."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "paths").mkdir()
    shutil.copy2(problems_path, path / "problems.npz")


def write_trials(path: Path, results: list[dict[str, object]]) -> None:
    """Persist every scalar trial metric, including seed and difficulty."""
    excluded = {"start", "goal", "dense_path"}
    fields = [key for key in results[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def numeric_stats(values: list[float]) -> dict[str, float] | None:
    """Return common descriptive statistics for finite values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_rows(results: list[dict[str, object]]) -> dict[str, object]:
    """Summarize success, successful timing, solution quality, and workload."""
    successes = [row for row in results if bool(row["success"])]
    return {
        "trials": len(results),
        "successes": len(successes),
        # Compatibility names used by the shared Franka result auditor.
        "num_problems": len(results),
        "num_successes": len(successes),
        "success_rate": len(successes) / len(results),
        "planning_time_seconds_all": numeric_stats(
            [float(row["planning_time_seconds"]) for row in results]
        ),
        "planning_time_seconds_successes": numeric_stats(
            [float(row["planning_time_seconds"]) for row in successes]
        ),
        "path_motion_time_seconds_successes": numeric_stats(
            [float(row["path_motion_time_seconds"]) for row in successes]
        ),
        "final_normalized_distance_failures": numeric_stats(
            [
                float(row["final_normalized_distance"])
                for row in results
                if not bool(row["success"])
            ]
        ),
        "total_tree_nodes": int(sum(int(row["nodes"]) for row in results)),
        "tree_nodes_per_trial": numeric_stats(
            [float(row["nodes"]) for row in results]
        ),
        "total_checked_waypoints": int(
            sum(int(row["checked_waypoints"]) for row in results)
        ),
        "checked_waypoints_per_trial": numeric_stats(
            [float(row["checked_waypoints"]) for row in results]
        ),
        "total_limit_rejections": int(
            sum(int(row["limit_rejections"]) for row in results)
        ),
        "total_acceleration_rejections": int(
            sum(int(row["acceleration_rejections"]) for row in results)
        ),
        "total_jerk_rejections": int(
            sum(int(row["jerk_rejections"]) for row in results)
        ),
        "total_collision_rejections": int(
            sum(int(row["self_collision_rejections"]) for row in results)
        ),
    }


def summarize_planner(
    results: list[dict[str, object]], configuration: dict[str, object]
) -> dict[str, object]:
    """Build overall and difficulty-stratified planner summaries."""
    summary = summarize_rows(results)
    summary["by_difficulty"] = {
        difficulty: summarize_rows(
            [row for row in results if row["difficulty_tier"] == difficulty]
        )
        for difficulty in DIFFICULTIES
    }
    summary["configuration"] = configuration
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    return summary


def save_vanilla_path(output_dir: Path, result: dict[str, object]) -> None:
    """Save one VanillaRRT trajectory using its unique expanded trial ID."""
    np.savez_compressed(
        output_dir / "paths" / f"problem_{int(result['problem_id']):03d}.npz",
        states=result["dense_path"],
        start=result["start"],
        goal=result["goal"],
        success=np.asarray(result["success"]),
        integration_dt=np.asarray(0.02),
        goal_radius=np.asarray(result["goal_radius"]),
        planner=np.asarray("VanillaRRT"),
        path_motion_time_seconds=np.asarray(result["path_motion_time_seconds"]),
        path_waypoints=np.asarray(result["path_waypoints"]),
    )


def run_vanilla(
    output_dir: Path,
    problems_path: Path,
    problems: list[dict[str, object]],
    state_bank: np.ndarray,
    scene: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run current VanillaRRT for all repeated problems."""
    prepare_result_directory(output_dir, problems_path)
    checker = FrankaCuroboCollisionChecker(DEFAULT_CUROBO_CONFIG, scene_model=scene)
    config: dict[str, object] = {
        "seed": 0,
        "planning_time": args.planning_time,
        "goal_radius": args.goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": 0.30,
        "max_edge_time": 1.0,
        "integration_dt": 0.02,
        "extension_trials": 16,
        "acceleration_scale": 0.50,
        "max_iterations": 10_000_000,
    }
    vanilla_benchmark._WORKER_STATE_BANK = state_bank
    vanilla_benchmark._WORKER_CHECKER = checker
    vanilla_benchmark._WORKER_CONFIG = config
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for problem in problems:
            # run_problem adds problem_id; offset the base to preserve the exact
            # same benchmark seed for corresponding Vanilla and Flow trials.
            config["seed"] = int(problem["benchmark_seed"]) - int(problem["problem_id"])
            result = vanilla_benchmark.run_problem(problem)
            results.append(result)
            save_vanilla_path(output_dir, result)
            write_trials(output_dir / "trials.partial.csv", results)
            print(
                f"VANILLA {result['difficulty_tier']} "
                f"trial={int(result['trial_index']):02d} seed={int(result['seed'])} "
                f"success={bool(result['success'])} "
                f"time={float(result['planning_time_seconds']):.3f}s "
                f"nodes={int(result['nodes'])}",
                flush=True,
            )
    finally:
        checker.close()
        vanilla_benchmark._WORKER_CHECKER = None
    configuration = {
        **config,
        "seed": "benchmark_seed stored per trial",
        "planner": "VanillaRRT",
        "goal_parent_metric": (
            "FrankaPanda: q-only for explicit goal samples; full q,dq otherwise"
        ),
        "extension_candidate_metric": (
            "FrankaPanda: q-only for explicit goal samples; full q,dq otherwise"
        ),
        "environment": "MorphIt/cuRobo self- and cuboid-collision checking",
        "robot_collision_config": str(DEFAULT_CUROBO_CONFIG.resolve()),
        "problems": "problems.npz",
        "problems_sha256": vanilla_benchmark.file_sha256(problems_path),
        "problem_ids": [int(problem["problem_id"]) for problem in problems],
        "scene_model": scene,
        "urdf_sha256": vanilla_benchmark.file_sha256(args.urdf.resolve()),
    }
    summary = summarize_planner(results, configuration)
    summary["benchmark_wall_time_seconds"] = time.perf_counter() - started
    write_trials(output_dir / "trials.csv", results)
    (output_dir / "trials.partial.csv").unlink(missing_ok=True)
    write_json(output_dir / "summary.json", summary)
    return results, summary


def run_flow(
    output_dir: Path,
    problems_path: Path,
    problems: list[dict[str, object]],
    state_bank: np.ndarray,
    scene: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run current FlowEBRRT for all repeated problems."""
    prepare_result_directory(output_dir, problems_path)
    checker = FrankaCuroboCollisionChecker(DEFAULT_CUROBO_CONFIG, scene_model=scene)
    generator = flow_benchmark.FrankaFlowEdgeGenerator(
        checkpoint_path=args.checkpoint.resolve(),
        device="cuda",
        sample_steps=16,
        clamp_outputs=True,
        seed=args.first_trial_seed,
    )
    planner_args = argparse.Namespace(
        seed=0,
        acceleration_scale=0.50,
        goal_radius=args.goal_radius,
        max_random_edge_time=0.30,
        max_iterations=10_000_000,
        planning_time=args.planning_time,
        flow_prefetch_batch_size=16,
        minimum_flow_prefix_steps=5,
        num_sorted_edge_trials=32,
        num_random_edges=1,
        epsilon_random=0.05,
        goal_sampling_probability=0.30,
    )
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for problem in problems:
            planner_args.seed = int(problem["benchmark_seed"]) - int(
                problem["problem_id"]
            )
            result = flow_benchmark.run_problem(
                problem,
                state_bank=state_bank,
                checker=checker,
                generator=generator,
                args=planner_args,
            )
            results.append(result)
            flow_benchmark.save_path_result(
                output_dir,
                result,
                integration_dt=generator.dt,
                goal_radius=args.goal_radius,
            )
            write_trials(output_dir / "trials.partial.csv", results)
            print(
                f"FLOW {result['difficulty_tier']} "
                f"trial={int(result['trial_index']):02d} seed={int(result['seed'])} "
                f"success={bool(result['success'])} "
                f"time={float(result['planning_time_seconds']):.3f}s "
                f"nodes={int(result['nodes'])}",
                flush=True,
            )
            gc.collect()
    finally:
        checker.close()
    configuration = {
        "planner": "FlowEBRRT",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": flow_benchmark.file_sha256(args.checkpoint.resolve()),
        "checkpoint_epoch": generator.checkpoint_epoch,
        "seed": "benchmark_seed stored per trial",
        "planning_time": args.planning_time,
        "goal_radius": args.goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": 0.30,
        "goal_parent_metric": (
            "FrankaPanda: q-only for explicit goal samples; full q,dq otherwise"
        ),
        "flow_edge_ranking_metric": "normalized_7d_joint_position_l2",
        "max_random_edge_time": 0.30,
        "flow_prefetch_batch_size": 16,
        "minimum_flow_prefix_steps": 5,
        "num_sorted_edge_trials": 32,
        "num_random_edges": 1,
        "epsilon_random": 0.05,
        "environment": "MorphIt/cuRobo self- and cuboid-collision checking",
        "robot_collision_config": str(DEFAULT_CUROBO_CONFIG.resolve()),
        "problems": "problems.npz",
        "problems_sha256": flow_benchmark.file_sha256(problems_path),
        "problem_ids": [int(problem["problem_id"]) for problem in problems],
        "scene_model": scene,
        "urdf_sha256": flow_benchmark.file_sha256(args.urdf.resolve()),
        "collision_asset_sha256": flow_benchmark.collision_asset_hashes(
            args.urdf.resolve()
        ),
    }
    summary = summarize_planner(results, configuration)
    summary["benchmark_wall_time_seconds"] = time.perf_counter() - started
    summary["flow_profile_totals"] = {
        key: float(sum(float(row[key]) for row in results))
        for key in (
            "flow_generation_seconds",
            "flow_model_seconds",
            "flow_postprocess_seconds",
            "flow_generated_bundles",
            "flow_generation_calls",
            "flow_cache_hits",
            "flow_cache_misses",
            "flow_edge_trials",
            "random_control_trials",
            "sequence_edges_executed",
            "sequence_executed_steps",
            "sequence_available_steps",
        )
    }
    write_trials(output_dir / "trials.csv", results)
    (output_dir / "trials.partial.csv").unlink(missing_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_config.json", configuration)
    return results, summary


def fmt_stats(stats: dict[str, float] | None, key: str) -> str:
    """Format one optional timing statistic for Markdown."""
    return "--" if stats is None else f"{stats[key]:.3f}"


def make_result_tables(
    vanilla: dict[str, object], flow: dict[str, object]
) -> str:
    """Create overall and per-difficulty Markdown comparison tables."""
    lines = [
        "# Static cuboid: three problems x 20 seeds",
        "",
        "All planning times below use only successful trials unless marked `all`. ",
        "Path time is physical trajectory execution time, not planner wall time.",
        "",
        "| Metric | VanillaRRT | FlowEBRRT |",
        "|---|---:|---:|",
    ]
    lines.extend(
        [
            f"| Success | {vanilla['successes']}/{vanilla['trials']} ({100*vanilla['success_rate']:.1f}%) | {flow['successes']}/{flow['trials']} ({100*flow['success_rate']:.1f}%) |",
            *[
                f"| {difficulty.capitalize()} success | "
                f"{vanilla['by_difficulty'][difficulty]['successes']}/20 "
                f"({100*vanilla['by_difficulty'][difficulty]['success_rate']:.1f}%) | "
                f"{flow['by_difficulty'][difficulty]['successes']}/20 "
                f"({100*flow['by_difficulty'][difficulty]['success_rate']:.1f}%) |"
                for difficulty in DIFFICULTIES
            ],
            f"| Successful planning mean (s) | {fmt_stats(vanilla['planning_time_seconds_successes'], 'mean')} | {fmt_stats(flow['planning_time_seconds_successes'], 'mean')} |",
            f"| Successful planning median (s) | {fmt_stats(vanilla['planning_time_seconds_successes'], 'median')} | {fmt_stats(flow['planning_time_seconds_successes'], 'median')} |",
            f"| Path-duration mean (s) | {fmt_stats(vanilla['path_motion_time_seconds_successes'], 'mean')} | {fmt_stats(flow['path_motion_time_seconds_successes'], 'mean')} |",
            f"| Path-duration median (s) | {fmt_stats(vanilla['path_motion_time_seconds_successes'], 'median')} | {fmt_stats(flow['path_motion_time_seconds_successes'], 'median')} |",
            f"| Total tree nodes | {vanilla['total_tree_nodes']:,} | {flow['total_tree_nodes']:,} |",
            f"| Checked waypoints | {vanilla['total_checked_waypoints']:,} | {flow['total_checked_waypoints']:,} |",
            "",
            "| Problem | Planner | Success | Successful planning mean / median (s) | Path duration mean / median (s) | Total nodes | Checked waypoints |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        for planner, summary in (("VanillaRRT", vanilla), ("FlowEBRRT", flow)):
            row = summary["by_difficulty"][difficulty]
            lines.append(
                f"| {difficulty.capitalize()} | {planner} | "
                f"{row['successes']}/{row['trials']} ({100*row['success_rate']:.1f}%) | "
                f"{fmt_stats(row['planning_time_seconds_successes'], 'mean')} / "
                f"{fmt_stats(row['planning_time_seconds_successes'], 'median')} | "
                f"{fmt_stats(row['path_motion_time_seconds_successes'], 'mean')} / "
                f"{fmt_stats(row['path_motion_time_seconds_successes'], 'median')} | "
                f"{row['total_tree_nodes']:,} | {row['total_checked_waypoints']:,} |"
            )
    return "\n".join(lines) + "\n"


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a required audit and preserve its full output."""
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def finalize_existing(output_root: Path, args: argparse.Namespace) -> None:
    """Finalize complete saved trials without rerunning either planner."""
    scene_path = output_root / "static_obstacle_scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    problem_hashes: dict[str, str] = {}
    for label in ("vanilla", "flow"):
        result_dir = output_root / label
        summary_path = result_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["num_problems"] = int(summary["trials"])
        summary["num_successes"] = int(summary["successes"])
        configuration = summary["configuration"]
        problems_path = result_dir / "problems.npz"
        problems_sha256 = vanilla_benchmark.file_sha256(problems_path)
        problem_hashes[label] = problems_sha256
        configuration.update(
            {
                "problems": "problems.npz",
                "problems_sha256": problems_sha256,
                "problem_ids": list(range(int(summary["trials"]))),
                "scene_model": scene,
                "urdf": str(args.urdf.resolve()),
                "urdf_sha256": vanilla_benchmark.file_sha256(args.urdf.resolve()),
            }
        )
        if label == "flow":
            configuration["collision_asset_sha256"] = (
                flow_benchmark.collision_asset_hashes(args.urdf.resolve())
            )
            write_json(result_dir / "run_config.json", configuration)
        write_json(summary_path, summary)

    if len(set(problem_hashes.values())) != 1:
        raise RuntimeError("Vanilla and Flow trial problem files differ")
    audit_script = FRANKA_TESTING / "audit_franka_rrt_results.py"
    for label in ("vanilla", "flow"):
        run_logged(
            [
                sys.executable,
                str(audit_script),
                "--results-dir",
                str(output_root / label),
                "--urdf",
                str(args.urdf.resolve()),
                "--scene-json",
                str(scene_path),
                "--expected-count",
                "60",
            ],
            output_root / f"{label}_audit.log",
        )

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    run_logged(
        [
            sys.executable,
            str(FRANKA_TESTING / "compare_franka_rrt_results.py"),
            "--vanilla-dir",
            str(output_root / "vanilla"),
            "--flow-dir",
            str(output_root / "flow"),
            "--output-dir",
            str(comparison_dir),
        ],
        output_root / "comparison.log",
    )
    vanilla_summary = json.loads(
        (output_root / "vanilla" / "summary.json").read_text(encoding="utf-8")
    )
    flow_summary = json.loads(
        (output_root / "flow" / "summary.json").read_text(encoding="utf-8")
    )
    comparison = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vanilla": vanilla_summary,
        "flow": flow_summary,
        "seed_pairing_verified": True,
        "path_audits_passed": True,
        "results_table": "results_table.md",
    }
    write_json(output_root / "comparison_summary.json", comparison)
    table = make_result_tables(vanilla_summary, flow_summary)
    (output_root / "results_table.md").write_text(table, encoding="utf-8")
    print(table, flush=True)


def main() -> None:
    """Generate, benchmark, audit, compare, and summarize all trials."""
    args = parse_args()
    if args.trials_per_problem != TRIALS_PER_PROBLEM:
        raise ValueError("This requested benchmark requires exactly 20 trials per problem")
    if args.planning_time != PLANNING_TIME_SECONDS:
        raise ValueError("This requested benchmark requires a 30-second time limit")
    output_root = args.output_root.resolve()
    if args.finalize_existing:
        finalize_existing(output_root, args)
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    scene = scene_model()
    scene_path = output_root / "static_obstacle_scene.json"
    write_json(scene_path, scene)
    base_problems_path = output_root / "static_obstacle_problems_easy_medium_hard.npz"
    generation_report_path = output_root / "problem_generation.json"
    create_problem_set(
        dataset=args.dataset.resolve(),
        output_path=base_problems_path,
        report_path=generation_report_path,
        num_problems=3,
        state_bank_size=args.state_bank_size,
        seed=args.problem_seed,
        scene=scene,
        tier_targets={"easy": 1, "medium": 1, "hard": 1},
    )
    trial_problems_path = output_root / "trial_problems_60.npz"
    expand_problem_trials(
        base_problems_path,
        trial_problems_path,
        trials_per_problem=args.trials_per_problem,
        first_trial_seed=args.first_trial_seed,
    )
    problems = load_trial_problems(trial_problems_path)

    state_rng = np.random.default_rng(args.state_bank_seed)
    state_bank, _, _ = vanilla_benchmark.sample_dataset_state_bank(
        args.dataset.resolve(), args.state_bank_size, state_rng
    )
    write_json(
        output_root / "benchmark_contract.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "difficulty_order": list(DIFFICULTIES),
            "trials_per_problem": args.trials_per_problem,
            "shared_trial_seeds": [
                args.first_trial_seed + index
                for index in range(args.trials_per_problem)
            ],
            "planning_time_seconds": args.planning_time,
            "goal_radius": args.goal_radius,
            "state_bank_seed": args.state_bank_seed,
            "state_bank_size": args.state_bank_size,
            "scene_model": scene,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": vanilla_benchmark.file_sha256(args.dataset.resolve()),
            "base_problems_sha256": vanilla_benchmark.file_sha256(base_problems_path),
            "trial_problems_sha256": vanilla_benchmark.file_sha256(trial_problems_path),
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": vanilla_benchmark.file_sha256(args.urdf.resolve()),
        },
    )

    vanilla_results, vanilla_summary = run_vanilla(
        output_root / "vanilla",
        trial_problems_path,
        problems,
        state_bank,
        scene,
        args,
    )
    flow_results, flow_summary = run_flow(
        output_root / "flow",
        trial_problems_path,
        problems,
        state_bank,
        scene,
        args,
    )

    audit_script = FRANKA_TESTING / "audit_franka_rrt_results.py"
    for label in ("vanilla", "flow"):
        run_logged(
            [
                sys.executable,
                str(audit_script),
                "--results-dir",
                str(output_root / label),
                "--urdf",
                str(args.urdf.resolve()),
                "--scene-json",
                str(scene_path),
                "--expected-count",
                str(len(problems)),
            ],
            output_root / f"{label}_audit.log",
        )

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir()
    run_logged(
        [
            sys.executable,
            str(FRANKA_TESTING / "compare_franka_rrt_results.py"),
            "--vanilla-dir",
            str(output_root / "vanilla"),
            "--flow-dir",
            str(output_root / "flow"),
            "--output-dir",
            str(comparison_dir),
        ],
        output_root / "comparison.log",
    )

    comparison = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vanilla": vanilla_summary,
        "flow": flow_summary,
        "seed_pairing_verified": [row["seed"] for row in vanilla_results]
        == [row["seed"] for row in flow_results],
        "results_table": "results_table.md",
    }
    write_json(output_root / "comparison_summary.json", comparison)
    table = make_result_tables(vanilla_summary, flow_summary)
    (output_root / "results_table.md").write_text(table, encoding="utf-8")
    print(table, flush=True)
    print(f"RESULT_ROOT={output_root}", flush=True)


if __name__ == "__main__":
    main()
