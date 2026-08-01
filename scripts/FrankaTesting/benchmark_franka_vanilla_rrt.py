#!/usr/bin/env python3
"""Benchmark FlowMRMP's vanilla RRT on reachable Franka 14D A-B problems.

By default, this uses the curated, witness-backed 100-problem set. The planner
runs in an empty world but checks joint limits, velocity limits, and Franka
self-collision at every rollout waypoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_BENCHMARK_PROBLEMS as DEFAULT_PROBLEMS,
    DEFAULT_RAW_TRAJECTORY_DATASET as DEFAULT_DATASET,
    DEFAULT_URDF,
    DEFAULT_VANILLA_RESULTS_ROOT as DEFAULT_RESULTS_ROOT,
)

if str(FLOWMRMP_SRC) not in sys.path:
    sys.path.insert(0, str(FLOWMRMP_SRC))

from Agents.FrankaPanda import (  # noqa: E402
    EmptyFrankaEnvironment,
    FrankaPanda,
    FrankaSelfCollisionChecker,
)
from rrt import RRT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--problems",
        type=Path,
        default=DEFAULT_PROBLEMS,
        help=(
            "Fixed reachable A-B problem set. Defaults to the canonical "
            "100-problem Franka benchmark."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--planning-time", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--state-bank-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.25,
        help="Normalized 7D joint-position goal radius.",
    )
    parser.add_argument("--goal-sampling-probability", type=float, default=0.30)
    parser.add_argument(
        "--max-edge-time",
        type=float,
        default=1.0,
        help="Maximum constant-control edge duration (50 steps at dt=0.02).",
    )
    parser.add_argument("--integration-dt", type=float, default=0.02)
    parser.add_argument("--extension-trials", type=int, default=16)
    parser.add_argument("--acceleration-scale", type=float, default=0.50)
    parser.add_argument("--max-iterations", type=int, default=10_000_000)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_problems(path: Path, num_problems: int) -> list[dict[str, object]]:
    with np.load(path) as data:
        if len(data["starts"]) < num_problems:
            raise ValueError(
                f"Requested {num_problems} problems but {path} contains {len(data['starts'])}"
            )
        optional = set(data.files)
        starts = np.asarray(data["starts"][:num_problems], dtype=np.float64)
        goals = np.asarray(data["goals"][:num_problems], dtype=np.float64)
        start_trajectory = np.asarray(data["start_trajectory"][:num_problems])
        start_timestep = np.asarray(data["start_timestep"][:num_problems])
        goal_trajectory = np.asarray(data["goal_trajectory"][:num_problems])
        goal_timestep = np.asarray(data["goal_timestep"][:num_problems])
        tiers = (
            np.asarray(data["difficulty_tier"][:num_problems]).astype(str)
            if "difficulty_tier" in optional
            else np.full(num_problems, "unlabeled")
        )
        durations = (
            np.asarray(data["witness_duration_seconds"][:num_problems], dtype=np.float64)
            if "witness_duration_seconds" in optional
            else np.full(num_problems, np.nan)
        )
        steps = (
            np.asarray(data["witness_steps"][:num_problems], dtype=np.int32)
            if "witness_steps" in optional
            else np.full(num_problems, -1, dtype=np.int32)
        )
        distances = (
            np.asarray(
                data["witness_initial_normalized_distance"][:num_problems],
                dtype=np.float64,
            )
            if "witness_initial_normalized_distance" in optional
            else np.full(num_problems, np.nan)
        )
    return [
        {
            "problem_id": problem_id,
            "start": starts[problem_id].copy(),
            "goal": goals[problem_id].copy(),
            "start_trajectory": int(start_trajectory[problem_id]),
            "start_timestep": int(start_timestep[problem_id]),
            "goal_trajectory": int(goal_trajectory[problem_id]),
            "goal_timestep": int(goal_timestep[problem_id]),
            "difficulty_tier": str(tiers[problem_id]),
            "witness_duration_seconds": float(durations[problem_id]),
            "witness_steps": int(steps[problem_id]),
            "witness_initial_normalized_distance": float(distances[problem_id]),
        }
        for problem_id in range(num_problems)
    ]


def sample_dataset_state_bank(
    dataset_path: Path, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample states plus trajectory/timestep provenance from many trajectories."""
    if size < 2:
        raise ValueError("state-bank-size must be at least 2")
    state_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    timestep_parts: list[np.ndarray] = []
    collected = 0

    with h5py.File(dataset_path, "r") as source:
        names = np.asarray(sorted(source.keys()))
        if names.size == 0:
            raise ValueError(f"No trajectories found in {dataset_path}")
        for name in names[rng.permutation(names.size)]:
            group = source[str(name)]
            q = group["positions"][:]
            dq = group["velocities"][:]
            if q.shape != dq.shape or q.ndim != 2 or q.shape[1] != 7:
                raise ValueError(f"Malformed trajectory {name}: q={q.shape}, dq={dq.shape}")
            states = np.concatenate((q, dq), axis=1).astype(np.float64, copy=False)
            state_parts.append(states)
            trajectory_parts.append(
                np.full(states.shape[0], int(name), dtype=np.int32)
            )
            timestep_parts.append(np.arange(states.shape[0], dtype=np.int32))
            collected += states.shape[0]
            if collected >= size:
                break

    if collected < size:
        raise ValueError(f"Requested {size:,} states but dataset contains {collected:,}")
    states = np.concatenate(state_parts, axis=0)
    trajectory_ids = np.concatenate(trajectory_parts)
    timestep_ids = np.concatenate(timestep_parts)
    chosen = rng.choice(states.shape[0], size=size, replace=False)
    return states[chosen], trajectory_ids[chosen], timestep_ids[chosen]


def choose_valid_problems(
    state_bank: np.ndarray,
    trajectory_ids: np.ndarray,
    timestep_ids: np.ndarray,
    *,
    num_problems: int,
    checker: FrankaSelfCollisionChecker,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Choose unique, independent valid start/goal state pairs."""
    agent = FrankaPanda(state_bank=state_bank, collision_checker=checker)
    order = rng.permutation(state_bank.shape[0])
    valid_ids: list[int] = []
    for index in order:
        if agent.is_state_valid(state_bank[index]):
            valid_ids.append(int(index))
            if len(valid_ids) == 2 * num_problems:
                break
    if len(valid_ids) < 2 * num_problems:
        raise RuntimeError(
            f"Only {len(valid_ids)} valid sampled states for {num_problems} problems"
        )

    problems = []
    for problem_id in range(num_problems):
        start_index = valid_ids[2 * problem_id]
        goal_index = valid_ids[2 * problem_id + 1]
        problems.append(
            {
                "problem_id": problem_id,
                "start": state_bank[start_index].copy(),
                "goal": state_bank[goal_index].copy(),
                "start_trajectory": int(trajectory_ids[start_index]),
                "start_timestep": int(timestep_ids[start_index]),
                "goal_trajectory": int(trajectory_ids[goal_index]),
                "goal_timestep": int(timestep_ids[goal_index]),
                "difficulty_tier": "unconstrained",
                "witness_duration_seconds": np.nan,
                "witness_steps": -1,
                "witness_initial_normalized_distance": np.nan,
            }
        )
    return problems


_WORKER_STATE_BANK: np.ndarray | None = None
_WORKER_CHECKER: FrankaSelfCollisionChecker | None = None
_WORKER_CONFIG: dict[str, object] | None = None


def initialize_worker(
    state_bank: np.ndarray, urdf: str, config: dict[str, object]
) -> None:
    global _WORKER_STATE_BANK, _WORKER_CHECKER, _WORKER_CONFIG
    _WORKER_STATE_BANK = np.asarray(state_bank, dtype=np.float64)
    _WORKER_CHECKER = FrankaSelfCollisionChecker(urdf)
    _WORKER_CONFIG = config


def dense_path_to_node(rrt: RRT, node_id: int) -> np.ndarray:
    ids, _, _, _ = rrt.get_path_to_node_id(node_id)
    states = [rrt.start.copy()]
    for current_id in ids[1:]:
        edge = rrt.tree.nodes[int(current_id)]["value"].path_from_parent
        states.extend(np.asarray(edge))
    return np.asarray(states, dtype=np.float64)


def run_problem(problem: dict[str, object]) -> dict[str, object]:
    if _WORKER_STATE_BANK is None or _WORKER_CHECKER is None or _WORKER_CONFIG is None:
        raise RuntimeError("Benchmark worker was not initialized")
    config = _WORKER_CONFIG
    seed = int(config["seed"]) + int(problem["problem_id"])
    agent = FrankaPanda(
        state_bank=_WORKER_STATE_BANK,
        collision_checker=_WORKER_CHECKER,
        agent_id=int(problem["problem_id"]),
        acceleration_scale=float(config["acceleration_scale"]),
    )
    start = np.asarray(problem["start"], dtype=np.float64)
    goal = np.asarray(problem["goal"], dtype=np.float64)
    initial_distance = agent.get_goal_distance(start, goal)
    rrt = RRT(
        start=start,
        goal=goal,
        goal_radius=float(config["goal_radius"]),
        env=EmptyFrankaEnvironment(),
        agent=agent,
        use_fixed_sampling_time=False,
        sampling_time_step=float(config["max_edge_time"]),
        minimum_time_step=float(config["integration_dt"]),
        max_iter=int(config["max_iterations"]),
        planning_time=float(config["planning_time"]),
        num_extension_trials=int(config["extension_trials"]),
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        reached_goal_function=agent.agent_reached_goal,
        random_point_function=agent.get_random_point,
        udf_seed=seed,
        goal_sampling_probability=float(config["goal_sampling_probability"]),
        print_logs=False,
    )
    rrt.plan_path()

    distances = np.array(
        [
            agent.get_goal_distance(node["value"].state, goal)
            for _, node in rrt.tree.nodes(data=True)
        ],
        dtype=np.float64,
    )
    best_node_id = int(np.argmin(distances))
    # Reconstruct from the stored edge waypoint arrays.  The legacy helper in
    # rrt.py sizes its output from rounded motion time and can under-allocate
    # when the goal is reached partway through a variable-duration edge.
    path_node_id = int(rrt.goal_node_id) if rrt.path_found else best_node_id
    dense_path = dense_path_to_node(rrt, path_node_id)
    final_distance = float(agent.get_goal_distance(dense_path[-1], goal))
    return {
        **problem,
        "seed": seed,
        "success": bool(rrt.path_found),
        "planning_time_seconds": float(rrt.last_plan_wall_time),
        "iterations": int(rrt.last_plan_iterations),
        "nodes": int(rrt.num_rrt_nodes()),
        "initial_normalized_distance": float(initial_distance),
        "final_normalized_distance": final_distance,
        "goal_radius": float(config["goal_radius"]),
        "path_motion_time_seconds": float(rrt.path_time) if rrt.path_found else np.nan,
        "path_cost": float(rrt.path_cost) if rrt.path_found else np.nan,
        "checked_waypoints": int(agent.checked_waypoints),
        "limit_rejections": int(agent.limit_rejections),
        "acceleration_rejections": int(agent.acceleration_rejections),
        "jerk_rejections": int(agent.jerk_rejections),
        "self_collision_rejections": int(agent.self_collision_rejections),
        "dense_path": dense_path,
    }


def write_trial_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "problem_id",
        "success",
        "planning_time_seconds",
        "iterations",
        "nodes",
        "initial_normalized_distance",
        "final_normalized_distance",
        "goal_radius",
        "path_motion_time_seconds",
        "path_cost",
        "checked_waypoints",
        "limit_rejections",
        "acceleration_rejections",
        "jerk_rejections",
        "self_collision_rejections",
        "start_trajectory",
        "start_timestep",
        "goal_trajectory",
        "goal_timestep",
        "difficulty_tier",
        "witness_duration_seconds",
        "witness_steps",
        "witness_initial_normalized_distance",
        "seed",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def make_summary(results: list[dict[str, object]], elapsed: float) -> dict[str, object]:
    success = np.array([bool(result["success"]) for result in results])
    times = np.array([float(result["planning_time_seconds"]) for result in results])
    successful_times = times[success]
    return {
        "num_problems": len(results),
        "num_successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "planning_time_seconds_all": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "min": float(times.min()),
            "max": float(times.max()),
            "p95": float(np.percentile(times, 95)),
        },
        "planning_time_seconds_successes": None
        if successful_times.size == 0
        else {
            "mean": float(successful_times.mean()),
            "median": float(np.median(successful_times)),
            "min": float(successful_times.min()),
            "max": float(successful_times.max()),
        },
        "benchmark_wall_time_seconds": float(elapsed),
        "total_checked_waypoints": int(sum(int(r["checked_waypoints"]) for r in results)),
        "total_limit_rejections": int(sum(int(r["limit_rejections"]) for r in results)),
        "total_acceleration_rejections": int(
            sum(int(r["acceleration_rejections"]) for r in results)
        ),
        "total_jerk_rejections": int(
            sum(int(r["jerk_rejections"]) for r in results)
        ),
        "total_self_collision_rejections": int(
            sum(int(r["self_collision_rejections"]) for r in results)
        ),
    }


def plot_summary(output_path: Path, results: list[dict[str, object]]) -> None:
    success = np.array([bool(result["success"]) for result in results])
    times = np.array([float(result["planning_time_seconds"]) for result in results])
    initial = np.array([float(result["initial_normalized_distance"]) for result in results])
    final = np.array([float(result["final_normalized_distance"]) for result in results])
    x = np.arange(len(results))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = np.where(success, "#2ca02c", "#d62728")
    axes[0].bar(x, times, color=colors, width=0.9)
    axes[0].set(title="Planning time per A-B problem", xlabel="Problem", ylabel="Seconds")
    axes[1].bar(["success", "failure"], [success.sum(), (~success).sum()], color=["#2ca02c", "#d62728"])
    axes[1].set(title=f"Success rate: {100.0 * success.mean():.1f}%", ylabel="Problems")
    axes[2].scatter(initial, final, c=colors, alpha=0.8)
    limit = max(float(initial.max()), float(final.max()))
    axes[2].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[2].set(
        title="Normalized 14D goal distance",
        xlabel="Initial distance",
        ylabel="Final/best distance",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_problems <= 0 or args.planning_time <= 0 or args.workers <= 0:
        raise ValueError("num-problems, planning-time, and workers must be positive")
    rng = np.random.default_rng(args.seed)
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (DEFAULT_RESULTS_ROOT / run_name)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = output_dir / "paths"
    paths_dir.mkdir()

    print(f"sampling {args.state_bank_size:,} states from {args.dataset}", flush=True)
    state_bank, trajectory_ids, timestep_ids = sample_dataset_state_bank(
        args.dataset, args.state_bank_size, rng
    )
    problem_checker = FrankaSelfCollisionChecker(args.urdf)
    local_problems_path = output_dir / "problems.npz"
    if args.problems is None:
        problems = choose_valid_problems(
            state_bank,
            trajectory_ids,
            timestep_ids,
            num_problems=args.num_problems,
            checker=problem_checker,
            rng=rng,
        )
        starts = np.stack([problem["start"] for problem in problems])
        goals = np.stack([problem["goal"] for problem in problems])
        np.savez_compressed(
            local_problems_path,
            starts=starts,
            goals=goals,
            start_trajectory=np.array([p["start_trajectory"] for p in problems]),
            start_timestep=np.array([p["start_timestep"] for p in problems]),
            goal_trajectory=np.array([p["goal_trajectory"] for p in problems]),
            goal_timestep=np.array([p["goal_timestep"] for p in problems]),
        )
        problem_sampling = "independent random (q, dq) states from dataset200k support"
        problems_source = None
    else:
        source_problems = args.problems.resolve()
        problems = load_fixed_problems(source_problems, args.num_problems)
        shutil.copy2(source_problems, local_problems_path)
        if file_sha256(source_problems) != file_sha256(local_problems_path):
            raise IOError("Copied A-B problem file does not match its source")
        problem_sampling = (
            "known-feasible same-trajectory acceleration witnesses; selection "
            "uses physics and diversity criteria only"
        )
        problems_source = str(source_problems)

    for problem in problems:
        for label in ("start", "goal"):
            state = np.asarray(problem[label], dtype=np.float64)
            if not (
                np.isfinite(state).all()
                and FrankaPanda(
                    state_bank=state_bank,
                    collision_checker=problem_checker,
                ).is_state_valid(state)
            ):
                raise ValueError(
                    f"Problem {problem['problem_id']} has an invalid {label} state"
                )
    problem_checker.close()

    config = {
        "seed": args.seed,
        "planning_time": args.planning_time,
        "goal_radius": args.goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": args.goal_sampling_probability,
        "max_edge_time": args.max_edge_time,
        "integration_dt": args.integration_dt,
        "extension_trials": args.extension_trials,
        "acceleration_scale": args.acceleration_scale,
        "max_iterations": args.max_iterations,
        "problems": "problems.npz",
        "problems_source": problems_source,
        "problems_sha256": file_sha256(local_problems_path),
    }
    started = time.perf_counter()
    if args.workers == 1:
        initialize_worker(state_bank, str(args.urdf), config)
        results = []
        for problem in problems:
            result = run_problem(problem)
            results.append(result)
            print(
                f"problem={int(result['problem_id']):03d} "
                f"success={result['success']} time={result['planning_time_seconds']:.3f}s "
                f"nodes={result['nodes']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=initialize_worker,
            initargs=(state_bank, str(args.urdf), config),
        ) as pool:
            results = []
            for result in pool.imap_unordered(run_problem, problems):
                results.append(result)
                print(
                    f"problem={int(result['problem_id']):03d} "
                    f"success={result['success']} time={result['planning_time_seconds']:.3f}s "
                    f"nodes={result['nodes']}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: int(item["problem_id"]))

    for result in results:
        np.savez_compressed(
            paths_dir / f"problem_{int(result['problem_id']):03d}.npz",
            states=result.pop("dense_path"),
            start=result["start"],
            goal=result["goal"],
            success=np.array(result["success"]),
            integration_dt=np.array(args.integration_dt),
            goal_radius=np.array(args.goal_radius),
        )
        result.pop("start")
        result.pop("goal")

    summary = make_summary(results, elapsed)
    summary["configuration"] = {
        **config,
        "dataset": str(args.dataset.resolve()),
        "urdf": str(args.urdf.resolve()),
        "workers": args.workers,
        "state_bank_size": args.state_bank_size,
        "problem_sampling": problem_sampling,
        "environment": (
            "empty; self-collision and joint/velocity limits checked per "
            "waypoint; acceleration limits enforced on sampled controls"
        ),
    }
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_trial_csv(output_dir / "trials.csv", results)
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    plot_summary(output_dir / "benchmark_summary.png", results)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
