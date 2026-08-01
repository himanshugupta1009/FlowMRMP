#!/usr/bin/env python3
"""Benchmark Franka FlowEBRRT on the Vanilla-RRT A-B problem set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_BENCHMARK_PROBLEMS as DEFAULT_PROBLEMS,
    DEFAULT_CHECKPOINT,
    DEFAULT_FLOW_RESULTS_ROOT as DEFAULT_RESULTS_ROOT,
    DEFAULT_RAW_TRAJECTORY_DATASET as DEFAULT_DATASET,
    DEFAULT_URDF,
    FLOW_EB_RRT_SRC,
    FLOWMRMP_ROOT,
    path_label,
)

TUNING_PROBLEM_IDS = (4, 10, 17, 37, 41, 57, 64, 81)
ENVIRONMENT_DESCRIPTION = (
    "empty; joint/velocity limits and self-collision checked per waypoint; "
    "acceleration and intra-edge jerk limits checked per control sequence"
)

for module_path in (FLOW_EB_RRT_SRC, FLOWMRMP_SRC):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from Agents.FrankaPanda import (  # noqa: E402
    EmptyFrankaEnvironment,
    FrankaPanda,
    FrankaSelfCollisionChecker,
)
from flow_eb_rrt import FlowEBRRT, FrankaFlowEdgeGenerator  # noqa: E402
from benchmark_franka_vanilla_rrt import sample_dataset_state_bank  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(path: Path) -> dict[str, object]:
    """Record the exact checked-out revision and whether local edits exist."""
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def source_file_hashes() -> dict[str, str]:
    """Fingerprint every repository source file imported by this benchmark."""
    paths = (
        Path(__file__).resolve(),
        FLOWMRMP_ROOT
        / "scripts"
        / "FrankaTesting"
        / "benchmark_franka_vanilla_rrt.py",
        FLOWMRMP_ROOT / "src" / "flow_eb_rrt.py",
        FLOWMRMP_ROOT / "scripts" / "train_franka_edge_flow_matching.py",
        FLOWMRMP_ROOT / "scripts" / "train_soc_edge_flow_matching.py",
        FLOWMRMP_ROOT
        / "mrmp_with_kite_extend"
        / "src"
        / "kinodynamic_TI_eb_rrt.py",
        FLOWMRMP_ROOT / "mrmp_with_kite_extend" / "src" / "rrt.py",
        FLOWMRMP_ROOT / "mrmp_with_kite_extend" / "src" / "utils.py",
        FLOWMRMP_ROOT
        / "mrmp_with_kite_extend"
        / "src"
        / "Agents"
        / "FrankaPanda.py",
    )
    return {
        path_label(path): file_sha256(path)
        for path in paths
    }


def collision_asset_hashes(urdf_path: Path) -> dict[str, str]:
    """Fingerprint the URDF, adjacent SRDF, and all collision geometry files."""
    robot_dir = urdf_path.resolve().parent
    paths = [urdf_path.resolve(), *sorted(robot_dir.glob("*.srdf"))]
    collision_dir = robot_dir / "meshes" / "collision"
    paths.extend(sorted(path for path in collision_dir.rglob("*") if path.is_file()))
    return {
        path_label(path): file_sha256(path)
        for path in paths
    }


def dependency_versions() -> dict[str, str | None]:
    names = ("numpy", "torch", "h5py", "numba", "networkx", "pybullet", "matplotlib")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tuning-selection",
        type=Path,
        default=None,
        help="Verified tuning selection.json to bind to a final benchmark run.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument(
        "--problem-ids",
        default=None,
        help="Optional comma-separated IDs; overrides --num-problems.",
    )
    parser.add_argument("--planning-time", type=float, default=30.0)
    parser.add_argument("--state-bank-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.25,
        help="Normalized 7D joint-position goal radius.",
    )
    parser.add_argument("--goal-sampling-probability", type=float, default=0.30)
    parser.add_argument("--max-random-edge-time", type=float, default=0.30)
    parser.add_argument("--acceleration-scale", type=float, default=0.50)
    parser.add_argument("--max-iterations", type=int, default=10_000_000)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--flow-prefetch-batch-size", type=int, default=16)
    parser.add_argument(
        "--minimum-flow-prefix-steps",
        type=int,
        default=5,
        help=(
            "Deprecated compatibility field; learned-duration edges now execute "
            "their full predicted N except when the goal is reached earlier."
        ),
    )
    parser.add_argument(
        "--no-flow-prefix-truncation",
        action="store_true",
        help=(
            "Deprecated no-op retained for command compatibility; full learned "
            "duration is now always used unless the goal is reached earlier."
        ),
    )
    parser.add_argument("--num-sorted-edge-trials", type=int, default=32)
    parser.add_argument("--num-random-edges", type=int, default=1)
    parser.add_argument("--epsilon-random", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    return parser.parse_args()


def load_problems(path: Path, problem_ids: list[int]) -> list[dict[str, object]]:
    with np.load(path) as data:
        available = len(data["starts"])
        if not problem_ids or min(problem_ids) < 0 or max(problem_ids) >= available:
            raise ValueError(
                f"Problem IDs must be within [0, {available - 1}]: {problem_ids}"
            )
        starts = data["starts"]
        goals = data["goals"]
        start_trajectory = data["start_trajectory"]
        start_timestep = data["start_timestep"]
        goal_trajectory = data["goal_trajectory"]
        goal_timestep = data["goal_timestep"]
        optional = set(data.files)
        tiers = (
            np.asarray(data["difficulty_tier"]).astype(str)
            if "difficulty_tier" in optional
            else np.full(available, "unlabeled")
        )
        durations = (
            np.asarray(data["witness_duration_seconds"], dtype=np.float64)
            if "witness_duration_seconds" in optional
            else np.full(available, np.nan)
        )
        steps = (
            np.asarray(data["witness_steps"], dtype=np.int32)
            if "witness_steps" in optional
            else np.full(available, -1, dtype=np.int32)
        )
        witness_distances = (
            np.asarray(data["witness_initial_normalized_distance"], dtype=np.float64)
            if "witness_initial_normalized_distance" in optional
            else np.full(available, np.nan)
        )
    return [
        {
            "problem_id": problem_id,
            "start": starts[problem_id].astype(np.float64),
            "goal": goals[problem_id].astype(np.float64),
            "start_trajectory": int(start_trajectory[problem_id]),
            "start_timestep": int(start_timestep[problem_id]),
            "goal_trajectory": int(goal_trajectory[problem_id]),
            "goal_timestep": int(goal_timestep[problem_id]),
            "difficulty_tier": str(tiers[problem_id]),
            "witness_duration_seconds": float(durations[problem_id]),
            "witness_steps": int(steps[problem_id]),
            "witness_initial_normalized_distance": float(
                witness_distances[problem_id]
            ),
        }
        for problem_id in problem_ids
    ]


def dense_path_to_node(rrt: FlowEBRRT, node_id: int) -> np.ndarray:
    reverse_ids = []
    current = int(node_id)
    while current != -1:
        reverse_ids.append(current)
        current = int(rrt.tree.nodes[current]["value"].parent_id)
    node_ids = reverse_ids[::-1]
    states = [rrt.start.copy()]
    for current in node_ids[1:]:
        edge = np.asarray(rrt.tree.nodes[current]["value"].path_from_parent)
        states.extend(edge)
    return np.asarray(states, dtype=np.float64)


def profile_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before.get(key, 0.0)) for key in after}


def run_problem(
    problem: dict[str, object],
    *,
    state_bank: np.ndarray,
    checker: FrankaSelfCollisionChecker,
    generator: FrankaFlowEdgeGenerator,
    args: argparse.Namespace,
) -> dict[str, object]:
    problem_id = int(problem["problem_id"])
    seed = int(args.seed) + problem_id
    generator.torch_generator.manual_seed(42 + seed)
    before_planner = dict(generator.profile)

    agent = FrankaPanda(
        state_bank=state_bank,
        collision_checker=checker,
        agent_id=problem_id,
        acceleration_scale=args.acceleration_scale,
    )
    start = np.asarray(problem["start"], dtype=np.float64)
    goal = np.asarray(problem["goal"], dtype=np.float64)
    initial_distance = agent.get_goal_distance(start, goal)
    planner = FlowEBRRT(
        start=start,
        goal=goal,
        goal_radius=args.goal_radius,
        env=EmptyFrankaEnvironment(),
        agent=agent,
        flow_edge_generator=generator,
        use_fixed_sampling_time=False,
        sampling_time_step=args.max_random_edge_time,
        minimum_time_step=generator.dt,
        max_iter=args.max_iterations,
        planning_time=args.planning_time,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=generator.set_size,
        flow_prefetch_batch_size=args.flow_prefetch_batch_size,
        minimum_sequence_prefix_steps=args.minimum_flow_prefix_steps,
        truncate_sequence_to_target=False,
        num_skip_edges=args.num_sorted_edge_trials,
        num_random_edges=args.num_random_edges,
        epsilon_random=args.epsilon_random,
        udf_seed=seed,
        goal_sampling_probability=args.goal_sampling_probability,
        debug_flag=False,
        print_logs=False,
    )
    planner.set_profile_enabled(True)
    planner.plan_path()

    distances = np.asarray(
        [
            agent.get_goal_distance(node["value"].state, goal)
            for _, node in planner.tree.nodes(data=True)
        ],
        dtype=np.float64,
    )
    best_node_id = int(np.argmin(distances))
    path_node_id = int(planner.goal_node_id) if planner.path_found else best_node_id
    dense_path = dense_path_to_node(planner, path_node_id)
    final_distance = float(agent.get_goal_distance(dense_path[-1], goal))
    generator_delta = profile_delta(generator.profile, before_planner)
    return {
        **problem,
        "seed": seed,
        "success": bool(planner.path_found),
        "planning_time_seconds": float(planner.last_plan_wall_time),
        "iterations": int(planner.last_plan_iterations),
        "nodes": int(planner.num_rrt_nodes()),
        "initial_normalized_distance": float(initial_distance),
        "final_normalized_distance": final_distance,
        "goal_radius": float(args.goal_radius),
        "path_motion_time_seconds": (
            float(planner.path_time) if planner.path_found else np.nan
        ),
        "path_cost": float(planner.path_cost) if planner.path_found else np.nan,
        "checked_waypoints": int(agent.checked_waypoints),
        "limit_rejections": int(agent.limit_rejections),
        "acceleration_rejections": int(agent.acceleration_rejections),
        "jerk_rejections": int(agent.jerk_rejections),
        "self_collision_rejections": int(agent.self_collision_rejections),
        "flow_generation_seconds": float(planner.profile["flow_generation_s"]),
        "flow_model_seconds": generator_delta["model_s"],
        "flow_postprocess_seconds": generator_delta["postprocess_s"],
        "flow_generated_bundles": int(planner.profile["flow_generated_bundles"]),
        "flow_generation_calls": int(planner.profile["flow_generation_calls"]),
        "flow_cache_hits": int(planner.profile["flow_cache_hits"]),
        "flow_cache_misses": int(planner.profile["flow_cache_misses"]),
        "flow_edge_trials": int(planner.profile["try_edge_calls"]),
        "random_control_trials": int(planner.profile["random_control_calls"]),
        "sequence_edges_executed": int(planner.profile["sequence_edges_executed"]),
        "sequence_executed_steps": int(planner.profile["sequence_executed_steps"]),
        "sequence_available_steps": int(planner.profile["sequence_available_steps"]),
        "sequence_acceleration_rejections": int(
            planner.profile["sequence_acceleration_rejections"]
        ),
        "sequence_jerk_rejections": int(
            planner.profile["sequence_jerk_rejections"]
        ),
        "dense_path": dense_path,
    }


def write_trials(path: Path, results: list[dict[str, object]]) -> None:
    excluded = {"start", "goal", "dense_path"}
    fields = [key for key in results[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def save_path_result(
    output_dir: Path,
    result: dict[str, object],
    *,
    integration_dt: float,
    goal_radius: float,
) -> None:
    """Persist each problem immediately so a long benchmark loses no paths."""
    np.savez_compressed(
        output_dir / "paths" / f"problem_{int(result['problem_id']):03d}.npz",
        states=result["dense_path"],
        start=result["start"],
        goal=result["goal"],
        success=np.asarray(result["success"]),
        integration_dt=np.asarray(integration_dt),
        goal_radius=np.asarray(goal_radius),
        planner=np.asarray("FlowEBRRT"),
    )


def make_summary(results: list[dict[str, object]], elapsed: float) -> dict[str, object]:
    successes = np.asarray([bool(result["success"]) for result in results])
    times = np.asarray([float(result["planning_time_seconds"]) for result in results])
    successful_times = times[successes]
    profile_keys = (
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
        "sequence_acceleration_rejections",
        "sequence_jerk_rejections",
    )
    return {
        "num_problems": len(results),
        "num_successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "planning_time_seconds_all": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "min": float(times.min()),
            "max": float(times.max()),
            "p95": float(np.percentile(times, 95)),
        },
        "planning_time_seconds_successes": (
            None
            if successful_times.size == 0
            else {
                "mean": float(successful_times.mean()),
                "median": float(np.median(successful_times)),
                "min": float(successful_times.min()),
                "max": float(successful_times.max()),
            }
        ),
        "benchmark_wall_time_seconds": float(elapsed),
        "total_checked_waypoints": int(sum(int(x["checked_waypoints"]) for x in results)),
        "total_limit_rejections": int(sum(int(x["limit_rejections"]) for x in results)),
        "total_acceleration_rejections": int(
            sum(int(x["acceleration_rejections"]) for x in results)
        ),
        "total_jerk_rejections": int(
            sum(int(x["jerk_rejections"]) for x in results)
        ),
        "total_self_collision_rejections": int(
            sum(int(x["self_collision_rejections"]) for x in results)
        ),
        "flow_profile_totals": {
            key: float(sum(float(result[key]) for result in results))
            for key in profile_keys
        },
    }


def subset_summary(results: list[dict[str, object]]) -> dict[str, object]:
    success = np.asarray([bool(result["success"]) for result in results])
    planning = np.asarray([float(result["planning_time_seconds"]) for result in results])
    final = np.asarray([float(result["final_normalized_distance"]) for result in results])
    return {
        "num_problems": len(results),
        "num_successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "planning_time_seconds_mean": float(planning.mean()),
        "planning_time_seconds_median": float(np.median(planning)),
        "final_normalized_distance_median": float(np.median(final)),
    }


def plot_summary(path: Path, results: list[dict[str, object]]) -> None:
    success = np.asarray([bool(result["success"]) for result in results])
    planning = np.asarray([float(result["planning_time_seconds"]) for result in results])
    initial = np.asarray([float(result["initial_normalized_distance"]) for result in results])
    final = np.asarray([float(result["final_normalized_distance"]) for result in results])
    flow = np.asarray([float(result["flow_generation_seconds"]) for result in results])
    colors = np.where(success, "#2ca02c", "#d62728")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(np.arange(len(results)), planning, color=colors, width=0.9)
    axes[0, 0].set(title="FlowEBRRT planning time", xlabel="Problem", ylabel="Seconds")
    axes[0, 1].bar(
        ["success", "failure"],
        [success.sum(), (~success).sum()],
        color=["#2ca02c", "#d62728"],
    )
    axes[0, 1].set(title=f"Success rate: {100.0 * success.mean():.1f}%")
    axes[1, 0].scatter(initial, final, c=colors, alpha=0.8)
    limit = max(float(initial.max()), float(final.max()))
    axes[1, 0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1, 0].set(
        title="Normalized 14D goal distance",
        xlabel="Initial distance",
        ylabel="Final/best distance",
    )
    axes[1, 1].bar(np.arange(len(results)), flow, color="#4c78a8", width=0.9)
    axes[1, 1].set(
        title="Flow bundle generation time",
        xlabel="Problem",
        ylabel="Seconds",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_problems <= 0 or args.planning_time <= 0.0:
        raise ValueError("num-problems and planning-time must be positive")
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (DEFAULT_RESULTS_ROOT / run_name)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paths").mkdir()
    local_problems_path = output_dir / "problems.npz"
    shutil.copy2(args.problems.resolve(), local_problems_path)
    problems_sha256 = file_sha256(local_problems_path)
    if problems_sha256 != file_sha256(args.problems.resolve()):
        raise IOError("Copied A-B problem file does not match its source")

    problem_ids = (
        [int(value) for value in args.problem_ids.split(",")]
        if args.problem_ids
        else list(range(args.num_problems))
    )
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("problem IDs must be unique")
    problems = load_problems(local_problems_path, problem_ids)
    rng = np.random.default_rng(args.seed)
    state_bank, _, _ = sample_dataset_state_bank(args.dataset, args.state_bank_size, rng)
    checker = FrankaSelfCollisionChecker(args.urdf)
    validation_agent = FrankaPanda(state_bank=state_bank, collision_checker=checker)
    for problem in problems:
        for label in ("start", "goal"):
            if not validation_agent.is_state_valid(np.asarray(problem[label])):
                raise ValueError(
                    f"Problem {problem['problem_id']} has an invalid {label} state"
                )

    generator = FrankaFlowEdgeGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        sample_steps=args.sample_steps,
        clamp_outputs=False,
        seed=args.seed,
    )
    print(f"checkpoint: {generator.checkpoint_path}", flush=True)
    print(f"flow device: {generator.device}", flush=True)
    print(f"problems: {len(problems)} from {args.problems}", flush=True)
    checkpoint_sha256 = file_sha256(generator.checkpoint_path)
    dataset_sha256 = file_sha256(args.dataset)
    urdf_sha256 = file_sha256(args.urdf)
    collision_hashes = collision_asset_hashes(args.urdf)
    source_hashes = source_file_hashes()
    current_fixed_configuration = {
        "checkpoint_sha256": checkpoint_sha256,
        "problems_sha256": problems_sha256,
        "dataset_sha256": dataset_sha256,
        "urdf_sha256": urdf_sha256,
        "collision_asset_sha256": collision_hashes,
        "source_file_sha256": source_hashes,
        "device": str(generator.device),
        "state_bank_size": args.state_bank_size,
        "seed": args.seed,
        "goal_radius": args.goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": args.goal_sampling_probability,
        "max_random_edge_time": args.max_random_edge_time,
        "acceleration_scale": args.acceleration_scale,
        "max_iterations": args.max_iterations,
        "flow_prefetch_batch_size": args.flow_prefetch_batch_size,
        "num_sorted_edge_trials": args.num_sorted_edge_trials,
        "num_random_edges": args.num_random_edges,
        "epsilon_random": args.epsilon_random,
        "integration_dt": generator.dt,
        "flow_training_dataset_sha256": generator.metadata.get("dataset_sha256"),
        "environment": ENVIRONMENT_DESCRIPTION,
    }
    tuning_selection_info = None
    if args.tuning_selection is not None:
        tuning_source = args.tuning_selection.resolve()
        tuning_report = json.loads(tuning_source.read_text(encoding="utf-8"))
        selected = tuning_report.get("selected", {})
        expected_tuning = {
            "checkpoint_sha256": checkpoint_sha256,
            "problems_sha256": problems_sha256,
            "sample_steps": args.sample_steps,
            "minimum_prefix_steps": args.minimum_flow_prefix_steps,
            "truncate": False,
            "resolved_device": str(generator.device),
            "source_file_sha256": source_hashes,
            "problem_ids": list(TUNING_PROBLEM_IDS),
            "fixed_benchmark_configuration": current_fixed_configuration,
        }
        actual_tuning = {
            "checkpoint_sha256": tuning_report.get("checkpoint_sha256"),
            "problems_sha256": tuning_report.get("problems_sha256"),
            "sample_steps": selected.get("sample_steps"),
            "minimum_prefix_steps": selected.get("minimum_prefix_steps"),
            "truncate": selected.get("truncate"),
            "resolved_device": tuning_report.get("resolved_device"),
            "source_file_sha256": tuning_report.get("source_file_sha256"),
            "problem_ids": tuning_report.get("problem_ids"),
            "fixed_benchmark_configuration": tuning_report.get(
                "fixed_benchmark_configuration"
            ),
        }
        if actual_tuning != expected_tuning:
            raise ValueError(
                "Final benchmark configuration does not match tuning selection: "
                f"expected={expected_tuning}, actual={actual_tuning}"
            )
        local_tuning_path = output_dir / "tuning_selection.json"
        shutil.copy2(tuning_source, local_tuning_path)
        tuning_selection_info = {
            "file": "tuning_selection.json",
            "sha256": file_sha256(local_tuning_path),
            "selected": selected,
            "problem_ids": tuning_report.get("problem_ids"),
        }
    run_configuration = {
        "planner": "FlowEBRRT",
        "checkpoint": str(generator.checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": generator.checkpoint_epoch,
        "flow_training_dataset": generator.config.get("dataset"),
        "flow_training_dataset_sha256": generator.metadata.get("dataset_sha256"),
        "dataset": str(args.dataset.resolve()),
        "dataset_bytes": args.dataset.stat().st_size,
        "dataset_sha256": dataset_sha256,
        "problems": "problems.npz",
        "problems_source": str(args.problems.resolve()),
        "problems_sha256": problems_sha256,
        "num_problems": len(problems),
        "problem_ids": problem_ids,
        "problem_sampling": (
            "known-feasible same-trajectory acceleration witnesses; selection "
            "uses physics and diversity criteria only"
            if problems and problems[0]["difficulty_tier"] != "unlabeled"
            else "same distinct A-B pairs as Vanilla RRT benchmark"
        ),
        "tuning_selection": tuning_selection_info,
        "seed": args.seed,
        "planner_seed_rule": "seed + problem_id",
        "flow_noise_seed_rule": "42 + seed + problem_id",
        "state_bank_seed": args.seed,
        "state_bank_size": args.state_bank_size,
        "device": str(generator.device),
        "sample_steps": args.sample_steps,
        "planning_time": args.planning_time,
        "max_iterations": args.max_iterations,
        "goal_radius": args.goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": args.goal_sampling_probability,
        "max_random_edge_time": args.max_random_edge_time,
        "acceleration_scale": args.acceleration_scale,
        "flow_prefetch_batch_size": args.flow_prefetch_batch_size,
        "minimum_flow_prefix_steps": args.minimum_flow_prefix_steps,
        "flow_prefix_truncation": False,
        "flow_edge_ranking": (
            "FM-predicted query-relative terminal 14D state; candidates are "
            "traversed nearest-first and propagated for their full predicted N"
        ),
        "num_sorted_edge_trials": args.num_sorted_edge_trials,
        "num_random_edges": args.num_random_edges,
        "epsilon_random": args.epsilon_random,
        "integration_dt": generator.dt,
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": urdf_sha256,
        "collision_asset_sha256": collision_hashes,
        "environment": ENVIRONMENT_DESCRIPTION,
        "collision_geometry_note": (
            "Paired Vanilla/Flow benchmark proxy: the shared PyBullet URDF fixes "
            "the gripper origins at y=+/-0.065 m, whereas the CuRobo source robot "
            "uses +/-0.04 m. Both planners use the same proxy geometry."
        ),
        "command": [sys.executable, *sys.argv],
        "source_file_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "git": {"flowmrmp_repository": git_snapshot(FLOWMRMP_ROOT)},
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_configuration, stream, indent=2, sort_keys=True)

    started = time.perf_counter()
    results = []
    try:
        for problem in problems:
            result = run_problem(
                problem,
                state_bank=state_bank,
                checker=checker,
                generator=generator,
                args=args,
            )
            results.append(result)
            save_path_result(
                output_dir,
                result,
                integration_dt=generator.dt,
                goal_radius=args.goal_radius,
            )
            write_trials(output_dir / "trials.partial.csv", results)
            print(
                f"problem={int(result['problem_id']):03d} "
                f"success={result['success']} "
                f"time={result['planning_time_seconds']:.3f}s "
                f"nodes={result['nodes']} "
                f"flow={result['flow_generation_seconds']:.3f}s",
                flush=True,
            )
            gc.collect()
    finally:
        checker.close()
    elapsed = time.perf_counter() - started

    summary = make_summary(results, elapsed)
    result_ids = {int(result["problem_id"]) for result in results}
    tuning_ids = set(TUNING_PROBLEM_IDS)
    if (
        run_configuration["tuning_selection"] is not None
        and tuning_ids.issubset(result_ids)
        and len(result_ids) > len(tuning_ids)
    ):
        summary["tuning_problem_ids"] = list(TUNING_PROBLEM_IDS)
        summary["tuning_subset"] = subset_summary(
            [result for result in results if int(result["problem_id"]) in tuning_ids]
        )
        summary["untuned_held_out_subset"] = subset_summary(
            [result for result in results if int(result["problem_id"]) not in tuning_ids]
        )
    summary["configuration"] = run_configuration
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_trials(output_dir / "trials.csv", results)
    (output_dir / "trials.partial.csv").unlink(missing_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    plot_summary(output_dir / "benchmark_summary.png", results)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
