#!/usr/bin/env python3
"""Create and compare 25 Franka planning problems around one static cuboid.

Problem selection is planner-independent. Every start and goal is valid in the
shared MorphIt/cuRobo scene, the direct joint-space interpolation is valid in an
empty scene but blocked by the cuboid, and a collision-free two-segment
geometric route exists through a sampled via configuration.
"""

from __future__ import annotations

import argparse
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
FRANKA_TESTING = REPOSITORY_ROOT / "scripts" / "FrankaTesting"
CORE_RRT_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
for module_path in (MAIN_SCRIPTS, FRANKA_TESTING, CORE_RRT_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import benchmark_franka_flow_eb_rrt as flow_benchmark  # noqa: E402
import benchmark_franka_vanilla_rrt as vanilla_benchmark  # noqa: E402
from FrankaPanda import (  # noqa: E402
    DEFAULT_CUROBO_CONFIG,
    DQ_MAX,
    FrankaCuroboCollisionChecker,
    FrankaPanda,
    Q_LOWER,
    Q_UPPER,
)
from franka_paths import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_RAW_TRAJECTORY_DATASET,
    DEFAULT_URDF,
)

DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" / "static_cuboid_25_20260805"
)
DEFAULT_PROBLEM_SEED = 20260808
DEFAULT_PLANNER_SEED = 20260721
DEFAULT_NUM_PROBLEMS = 25
DEFAULT_STATE_BANK_SIZE = 100_000
DIRECT_INTERPOLATION_STEPS = 65
VIA_INTERPOLATION_STEPS = 33
TIER_TARGETS = {"easy": 8, "medium": 12, "hard": 5}

# Axis-aligned cuboid directly in front of the Panda base. cuRobo poses use
# [x, y, z, qw, qx, qy, qz] and dimensions are full side lengths in metres.
OBSTACLE_NAME = "front_static_cuboid"
OBSTACLE_POSE = [0.50, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0]
OBSTACLE_DIMS = [0.20, 0.25, 0.30]


def parse_args() -> argparse.Namespace:
    """Parse the combined problem-generation and comparison command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RAW_TRAJECTORY_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-problems", type=int, default=DEFAULT_NUM_PROBLEMS)
    parser.add_argument("--state-bank-size", type=int, default=DEFAULT_STATE_BANK_SIZE)
    parser.add_argument("--problem-seed", type=int, default=DEFAULT_PROBLEM_SEED)
    parser.add_argument("--planner-seed", type=int, default=DEFAULT_PLANNER_SEED)
    parser.add_argument("--planning-time", type=float, default=30.0)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument("--problems-only", action="store_true")
    return parser.parse_args()


def scene_model() -> dict[str, object]:
    """Return the serializable cuRobo scene containing the benchmark cuboid."""
    return {
        "cuboid": {
            OBSTACLE_NAME: {
                "pose": list(OBSTACLE_POSE),
                "dims": list(OBSTACLE_DIMS),
            }
        }
    }


def tier_name(distance: float) -> str | None:
    """Map normalized configuration distance to the benchmark difficulty tier."""
    if 0.65 <= distance < 1.0:
        return "easy"
    if 1.0 <= distance < 1.4:
        return "medium"
    if 1.4 <= distance < 1.8:
        return "hard"
    return None


def collision_free_in_chunks(
    checker: FrankaCuroboCollisionChecker,
    configurations: np.ndarray,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Check ``(N, 7)`` configurations in large CUDA batches."""
    configurations = np.asarray(configurations, dtype=np.float32)
    output = np.empty(len(configurations), dtype=bool)
    for start in range(0, len(configurations), chunk_size):
        stop = min(start + chunk_size, len(configurations))
        output[start:stop] = checker.collision_free_mask(configurations[start:stop])
    return output


def interpolated_configurations(
    starts: np.ndarray,
    goals: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return linear joint paths shaped ``(batch, steps, 7)``."""
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float32)[None, :, None]
    return (1.0 - alpha) * np.asarray(starts, dtype=np.float32)[
        :, None, :
    ] + alpha * np.asarray(goals, dtype=np.float32)[:, None, :]


def find_geometric_via(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    valid_states: np.ndarray,
    checker: FrankaCuroboCollisionChecker,
    rng: np.random.Generator,
    trials: int = 512,
) -> np.ndarray | None:
    """Find a sampled configuration giving two collision-free linear segments."""
    candidate_ids = rng.choice(
        len(valid_states), size=min(trials, len(valid_states)), replace=False
    )
    via_q = np.asarray(valid_states[candidate_ids, :7], dtype=np.float32)
    starts = np.repeat(
        np.asarray(start_q, dtype=np.float32)[None, :], len(via_q), axis=0
    )
    goals = np.repeat(np.asarray(goal_q, dtype=np.float32)[None, :], len(via_q), axis=0)
    first = interpolated_configurations(starts, via_q, VIA_INTERPOLATION_STEPS)
    second = interpolated_configurations(via_q, goals, VIA_INTERPOLATION_STEPS)
    paths = np.concatenate((first, second[:, 1:, :]), axis=1)
    valid = collision_free_in_chunks(checker, paths.reshape(-1, 7)).reshape(
        len(via_q), paths.shape[1]
    )
    valid_ids = np.flatnonzero(valid.all(axis=1))
    if not len(valid_ids):
        return None
    lengths = np.linalg.norm(via_q - start_q, axis=1) + np.linalg.norm(
        goal_q - via_q, axis=1
    )
    best = valid_ids[np.argmin(lengths[valid_ids])]
    return via_q[best].astype(np.float64)


def create_problem_set(
    *,
    dataset: Path,
    output_path: Path,
    report_path: Path,
    num_problems: int,
    state_bank_size: int,
    seed: int,
    scene: dict[str, object],
    tier_targets: dict[str, int] | None = None,
) -> dict[str, object]:
    """Create endpoint-safe, cuboid-blocked, planner-independent A-B pairs."""
    requested_tiers = dict(TIER_TARGETS if tier_targets is None else tier_targets)
    if set(requested_tiers) != set(TIER_TARGETS) or any(
        count < 0 for count in requested_tiers.values()
    ):
        raise ValueError(
            "tier_targets must contain non-negative easy, medium, and hard counts"
        )
    if num_problems != sum(requested_tiers.values()):
        raise ValueError(
            f"This test defines {sum(requested_tiers.values())} tiered problems; "
            f"received --num-problems={num_problems}"
        )
    rng = np.random.default_rng(seed)
    states, trajectory_ids, timestep_ids = vanilla_benchmark.sample_dataset_state_bank(
        dataset, state_bank_size, rng
    )
    finite_and_limited = (
        np.isfinite(states).all(axis=1)
        & np.all(states[:, :7] >= Q_LOWER, axis=1)
        & np.all(states[:, :7] <= Q_UPPER, axis=1)
        & np.all(np.abs(states[:, 7:]) <= DQ_MAX, axis=1)
    )
    limited_ids = np.flatnonzero(finite_and_limited)

    empty_checker = FrankaCuroboCollisionChecker(DEFAULT_CUROBO_CONFIG)
    obstacle_checker = FrankaCuroboCollisionChecker(
        DEFAULT_CUROBO_CONFIG, scene_model=scene
    )
    try:
        endpoint_valid = np.zeros(len(states), dtype=bool)
        for start in range(0, len(limited_ids), 16_384):
            ids = limited_ids[start : start + 16_384]
            endpoint_valid[ids] = collision_free_in_chunks(
                empty_checker, states[ids, :7]
            ) & collision_free_in_chunks(obstacle_checker, states[ids, :7])
        valid_ids = np.flatnonzero(endpoint_valid)
        valid_states = states[valid_ids]
        if len(valid_ids) < 2 * num_problems:
            raise RuntimeError("Not enough cuboid-clear endpoint configurations")

        accepted: list[dict[str, object]] = []
        tier_counts = {name: 0 for name in requested_tiers}
        used_state_ids: set[int] = set()
        used_trajectory_ids: set[int] = set()
        pair_attempts = 0
        direct_blocked_candidates = 0
        maximum_attempts = 100_000
        while len(accepted) < num_problems and pair_attempts < maximum_attempts:
            batch_size = 256
            start_ids = rng.choice(valid_ids, batch_size, replace=True)
            goal_ids = rng.choice(valid_ids, batch_size, replace=True)
            distances = np.linalg.norm(
                FrankaPanda.normalize_configuration(states[start_ids])
                - FrankaPanda.normalize_configuration(states[goal_ids]),
                axis=1,
            )
            eligible = np.asarray(
                [tier_name(float(distance)) is not None for distance in distances]
            ) & (start_ids != goal_ids)
            start_ids = start_ids[eligible]
            goal_ids = goal_ids[eligible]
            distances = distances[eligible]
            if not len(start_ids):
                pair_attempts += batch_size
                continue

            direct_paths = interpolated_configurations(
                states[start_ids, :7],
                states[goal_ids, :7],
                DIRECT_INTERPOLATION_STEPS,
            )
            flat_direct = direct_paths.reshape(-1, 7)
            empty_valid = collision_free_in_chunks(empty_checker, flat_direct).reshape(
                len(start_ids), DIRECT_INTERPOLATION_STEPS
            )
            obstacle_valid = collision_free_in_chunks(
                obstacle_checker, flat_direct
            ).reshape(len(start_ids), DIRECT_INTERPOLATION_STEPS)

            for row, (start_id, goal_id, distance) in enumerate(
                zip(start_ids, goal_ids, distances)
            ):
                tier = tier_name(float(distance))
                if tier is None or tier_counts[tier] >= requested_tiers[tier]:
                    continue
                start_id = int(start_id)
                goal_id = int(goal_id)
                start_trajectory = int(trajectory_ids[start_id])
                goal_trajectory = int(trajectory_ids[goal_id])
                if (
                    start_id in used_state_ids
                    or goal_id in used_state_ids
                    or start_trajectory in used_trajectory_ids
                    or goal_trajectory in used_trajectory_ids
                    or start_trajectory == goal_trajectory
                ):
                    continue
                # The cuboid, not self-collision, must be what blocks the direct path.
                if not empty_valid[row].all() or obstacle_valid[row].all():
                    continue
                direct_blocked_candidates += 1
                via = find_geometric_via(
                    states[start_id, :7],
                    states[goal_id, :7],
                    valid_states,
                    obstacle_checker,
                    rng,
                )
                if via is None:
                    continue
                first_collision = int(np.flatnonzero(~obstacle_valid[row])[0])
                accepted.append(
                    {
                        "start": states[start_id].copy(),
                        "goal": states[goal_id].copy(),
                        "start_trajectory": start_trajectory,
                        "start_timestep": int(timestep_ids[start_id]),
                        "goal_trajectory": goal_trajectory,
                        "goal_timestep": int(timestep_ids[goal_id]),
                        "difficulty_tier": tier,
                        "initial_distance": float(distance),
                        "direct_first_collision_index": first_collision,
                        "geometric_via_configuration": via,
                    }
                )
                tier_counts[tier] += 1
                used_state_ids.update((start_id, goal_id))
                used_trajectory_ids.update((start_trajectory, goal_trajectory))
                print(
                    f"selected problem={len(accepted) - 1:03d} tier={tier} "
                    f"distance={distance:.3f} direct_collision_step={first_collision}",
                    flush=True,
                )
                if len(accepted) == num_problems:
                    break
            pair_attempts += batch_size
        if len(accepted) != num_problems:
            raise RuntimeError(
                f"Selected only {len(accepted)}/{num_problems} problems after "
                f"{pair_attempts} pair attempts"
            )
    finally:
        empty_checker.close()
        obstacle_checker.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_name=np.asarray("franka_static_cuboid_problem_set"),
        format_version=np.asarray(1, dtype=np.int32),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        seed=np.asarray(seed, dtype=np.int64),
        starts=np.stack([item["start"] for item in accepted]),
        goals=np.stack([item["goal"] for item in accepted]),
        start_trajectory=np.asarray(
            [item["start_trajectory"] for item in accepted], dtype=np.int32
        ),
        start_timestep=np.asarray(
            [item["start_timestep"] for item in accepted], dtype=np.int32
        ),
        goal_trajectory=np.asarray(
            [item["goal_trajectory"] for item in accepted], dtype=np.int32
        ),
        goal_timestep=np.asarray(
            [item["goal_timestep"] for item in accepted], dtype=np.int32
        ),
        difficulty_tier=np.asarray([item["difficulty_tier"] for item in accepted]),
        witness_duration_seconds=np.full(num_problems, np.nan),
        witness_steps=np.full(num_problems, -1, dtype=np.int32),
        witness_initial_normalized_distance=np.asarray(
            [item["initial_distance"] for item in accepted], dtype=np.float64
        ),
        direct_interpolation_steps=np.asarray(
            DIRECT_INTERPOLATION_STEPS, dtype=np.int32
        ),
        direct_first_collision_index=np.asarray(
            [item["direct_first_collision_index"] for item in accepted],
            dtype=np.int32,
        ),
        geometric_via_configurations=np.stack(
            [item["geometric_via_configuration"] for item in accepted]
        ),
        obstacle_name=np.asarray(OBSTACLE_NAME),
        obstacle_pose=np.asarray(OBSTACLE_POSE, dtype=np.float64),
        obstacle_dims=np.asarray(OBSTACLE_DIMS, dtype=np.float64),
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_is_planner_independent": True,
        "num_problems": num_problems,
        "tier_counts": tier_counts,
        "problem_seed": seed,
        "state_bank_size": state_bank_size,
        "endpoint_candidates_clear_of_self_and_obstacle": int(len(valid_ids)),
        "pair_attempts": pair_attempts,
        "direct_blocked_candidates_evaluated_for_via": direct_blocked_candidates,
        "selection_contract": [
            "start and goal satisfy joint/velocity limits",
            "start and goal are clear of MorphIt self-collision and the cuboid",
            "65-state direct joint interpolation is self-collision-free in an empty scene",
            "the same direct interpolation collides when the cuboid is present",
            "a sampled two-segment geometric path around the cuboid is collision-free",
            "all 50 endpoint source trajectories are unique",
        ],
        "scene_model": scene,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": vanilla_benchmark.file_sha256(dataset),
        "problem_file": str(output_path.resolve()),
        "problem_file_sha256": vanilla_benchmark.file_sha256(output_path),
        "robot_config": str(DEFAULT_CUROBO_CONFIG.resolve()),
        "robot_config_sha256": vanilla_benchmark.file_sha256(DEFAULT_CUROBO_CONFIG),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def load_problem_list(path: Path, count: int) -> list[dict[str, object]]:
    """Load the shared problem representation expected by both benchmark helpers."""
    return vanilla_benchmark.load_fixed_problems(path, count)


def benchmark_collision_speed(
    configurations: np.ndarray,
    scene: dict[str, object],
    repeats: int = 100,
) -> dict[str, object]:
    """Time 256-state batched checks with and without the cuboid."""
    batch = np.clip(
        np.asarray(configurations[:256, :7], dtype=np.float32), Q_LOWER, Q_UPPER
    )
    measurements: dict[str, object] = {"batch_size": len(batch), "repeats": repeats}
    for label, current_scene in (("self_only", None), ("self_plus_cuboid", scene)):
        checker = FrankaCuroboCollisionChecker(
            DEFAULT_CUROBO_CONFIG, scene_model=current_scene
        )
        try:
            for _ in range(10):
                checker.collision_free_mask(batch)
            checker.synchronize()
            started = time.perf_counter()
            for _ in range(repeats):
                checker.collision_free_mask(batch)
            checker.synchronize()
            elapsed = time.perf_counter() - started
        finally:
            checker.close()
        measurements[label] = {
            "seconds": elapsed,
            "microseconds_per_configuration": (
                1.0e6 * elapsed / (repeats * len(batch))
            ),
            "configurations_per_second": repeats * len(batch) / elapsed,
        }
    base = float(measurements["self_only"]["microseconds_per_configuration"])
    obstacle = float(measurements["self_plus_cuboid"]["microseconds_per_configuration"])
    measurements["cuboid_overhead_fraction"] = obstacle / base - 1.0
    return measurements


def prepare_result_directory(path: Path, problems_path: Path) -> None:
    """Create an empty result directory and copy the immutable problem set."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "paths").mkdir()
    shutil.copy2(problems_path, path / "problems.npz")


def run_vanilla(
    *,
    output_dir: Path,
    problems_path: Path,
    problems: list[dict[str, object]],
    state_bank: np.ndarray,
    urdf: Path,
    scene: dict[str, object],
    seed: int,
    planning_time: float,
    goal_radius: float,
    dataset: Path,
) -> dict[str, object]:
    """Run VanillaRRT with one persistent scene-aware CUDA checker."""
    prepare_result_directory(output_dir, problems_path)
    checker = FrankaCuroboCollisionChecker(DEFAULT_CUROBO_CONFIG, scene_model=scene)
    config = {
        "seed": seed,
        "planning_time": planning_time,
        "goal_radius": goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": 0.30,
        "max_edge_time": 1.0,
        "integration_dt": 0.02,
        "extension_trials": 16,
        "acceleration_scale": 0.50,
        "max_iterations": 10_000_000,
        "problems": "problems.npz",
        "problems_source": str(problems_path.resolve()),
        "problems_sha256": vanilla_benchmark.file_sha256(problems_path),
    }
    vanilla_benchmark._WORKER_STATE_BANK = state_bank
    vanilla_benchmark._WORKER_CHECKER = checker
    vanilla_benchmark._WORKER_CONFIG = config
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for problem in problems:
            result = vanilla_benchmark.run_problem(problem)
            results.append(result)
            np.savez_compressed(
                output_dir / "paths" / f"problem_{int(result['problem_id']):03d}.npz",
                states=result["dense_path"],
                start=result["start"],
                goal=result["goal"],
                success=np.asarray(result["success"]),
                integration_dt=np.asarray(0.02),
                goal_radius=np.asarray(goal_radius),
                planner=np.asarray("VanillaRRT"),
            )
            print(
                f"vanilla problem={int(result['problem_id']):03d} "
                f"success={result['success']} "
                f"time={result['planning_time_seconds']:.3f}s nodes={result['nodes']}",
                flush=True,
            )
    finally:
        checker.close()
        vanilla_benchmark._WORKER_CHECKER = None
    elapsed = time.perf_counter() - started
    vanilla_benchmark.write_trial_csv(output_dir / "trials.csv", results)
    summary = vanilla_benchmark.make_summary(results, elapsed)
    summary["configuration"] = {
        **config,
        "planner": "VanillaRRT",
        "dataset": str(dataset.resolve()),
        "urdf": str(urdf.resolve()),
        "urdf_sha256": vanilla_benchmark.file_sha256(urdf),
        "workers": 1,
        "state_bank_size": len(state_bank),
        "problem_sampling": "planner-independent static-cuboid blocked pairs",
        "environment": (
            "one static cuboid; joint/velocity limits plus MorphIt/cuRobo "
            "self- and scene-collision checked in batched rollout edges"
        ),
        "scene_model": scene,
        "robot_collision_config": str(DEFAULT_CUROBO_CONFIG.resolve()),
        "robot_collision_config_sha256": vanilla_benchmark.file_sha256(
            DEFAULT_CUROBO_CONFIG
        ),
    }
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    vanilla_benchmark.plot_summary(output_dir / "benchmark_summary.png", results)
    return summary


def run_flow(
    *,
    output_dir: Path,
    problems_path: Path,
    problems: list[dict[str, object]],
    state_bank: np.ndarray,
    urdf: Path,
    scene: dict[str, object],
    seed: int,
    planning_time: float,
    goal_radius: float,
    dataset: Path,
    checkpoint: Path,
) -> dict[str, object]:
    """Run FlowEBRRT with the identical problems, scene, and time budget."""
    prepare_result_directory(output_dir, problems_path)
    checker = FrankaCuroboCollisionChecker(DEFAULT_CUROBO_CONFIG, scene_model=scene)
    generator = flow_benchmark.FrankaFlowEdgeGenerator(
        checkpoint_path=checkpoint,
        device="cuda",
        sample_steps=16,
        clamp_outputs=True,
        seed=seed,
    )
    planner_args = argparse.Namespace(
        seed=seed,
        acceleration_scale=0.50,
        goal_radius=goal_radius,
        max_random_edge_time=0.30,
        max_iterations=10_000_000,
        planning_time=planning_time,
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
                goal_radius=goal_radius,
            )
            flow_benchmark.write_trials(output_dir / "trials.partial.csv", results)
            print(
                f"flow problem={int(result['problem_id']):03d} "
                f"success={result['success']} "
                f"time={result['planning_time_seconds']:.3f}s nodes={result['nodes']}",
                flush=True,
            )
            gc.collect()
    finally:
        checker.close()
    elapsed = time.perf_counter() - started
    flow_benchmark.write_trials(output_dir / "trials.csv", results)
    (output_dir / "trials.partial.csv").unlink(missing_ok=True)
    summary = flow_benchmark.make_summary(results, elapsed)
    configuration = {
        "planner": "FlowEBRRT",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": flow_benchmark.file_sha256(checkpoint),
        "checkpoint_epoch": generator.checkpoint_epoch,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": flow_benchmark.file_sha256(dataset),
        "problems": "problems.npz",
        "problems_source": str(problems_path.resolve()),
        "problems_sha256": flow_benchmark.file_sha256(problems_path),
        "num_problems": len(problems),
        "problem_ids": [int(problem["problem_id"]) for problem in problems],
        "problem_sampling": "planner-independent static-cuboid blocked pairs",
        "tuning_selection": None,
        "seed": seed,
        "state_bank_size": len(state_bank),
        "device": str(generator.device),
        "sample_steps": 16,
        "planning_time": planning_time,
        "max_iterations": 10_000_000,
        "goal_radius": goal_radius,
        "goal_metric": "normalized_7d_joint_position_l2",
        "goal_sampling_probability": 0.30,
        "max_random_edge_time": 0.30,
        "acceleration_scale": 0.50,
        "flow_prefetch_batch_size": 16,
        "minimum_flow_prefix_steps": 5,
        "flow_prefix_truncation": False,
        "num_sorted_edge_trials": 32,
        "num_random_edges": 1,
        "epsilon_random": 0.05,
        "integration_dt": generator.dt,
        "urdf": str(urdf.resolve()),
        "urdf_sha256": flow_benchmark.file_sha256(urdf),
        "collision_asset_sha256": flow_benchmark.collision_asset_hashes(urdf),
        "environment": (
            "one static cuboid; joint/velocity limits plus MorphIt/cuRobo "
            "self- and scene-collision checked in batched rollout edges"
        ),
        "scene_model": scene,
        "robot_collision_config": str(DEFAULT_CUROBO_CONFIG.resolve()),
        "robot_collision_config_sha256": flow_benchmark.file_sha256(
            DEFAULT_CUROBO_CONFIG
        ),
    }
    summary["configuration"] = configuration
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8"
    )
    flow_benchmark.plot_summary(output_dir / "benchmark_summary.png", results)
    return summary


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a verification/comparison command while preserving its full output."""
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main() -> None:
    """Generate problems, run both planners, audit paths, and compare outcomes."""
    args = parse_args()
    if args.num_problems != DEFAULT_NUM_PROBLEMS:
        raise ValueError(
            f"This requested test must contain exactly {DEFAULT_NUM_PROBLEMS} problems"
        )
    if args.planning_time <= 0.0 or args.state_bank_size < 10_000:
        raise ValueError(
            "planning-time must be positive and state-bank-size at least 10,000"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene = scene_model()
    scene_path = output_root / "static_obstacle_scene.json"
    problems_path = output_root / "static_obstacle_problems_25.npz"
    generation_report_path = output_root / "problem_generation.json"

    expected_scene = json.dumps(scene, indent=2, sort_keys=True)
    if scene_path.exists() and scene_path.read_text(encoding="utf-8") != expected_scene:
        raise FileExistsError(
            f"Existing scene differs from requested obstacle: {scene_path}"
        )
    scene_path.write_text(expected_scene, encoding="utf-8")
    if not problems_path.exists():
        report = create_problem_set(
            dataset=args.dataset.resolve(),
            output_path=problems_path,
            report_path=generation_report_path,
            num_problems=args.num_problems,
            state_bank_size=args.state_bank_size,
            seed=args.problem_seed,
            scene=scene,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.problems_only:
        print(f"problems saved to {problems_path}")
        return

    problems = load_problem_list(problems_path, args.num_problems)
    planner_rng = np.random.default_rng(args.planner_seed)
    state_bank, _, _ = vanilla_benchmark.sample_dataset_state_bank(
        args.dataset.resolve(), args.state_bank_size, planner_rng
    )
    timing_path = output_root / "collision_timing.json"
    timing = benchmark_collision_speed(state_bank, scene)
    timing_path.write_text(
        json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"collision_timing": timing}, indent=2), flush=True)
    vanilla_dir = output_root / "vanilla"
    flow_dir = output_root / "flow"
    comparison_dir = output_root / "comparison"
    if (vanilla_dir / "summary.json").exists():
        vanilla_summary = json.loads(
            (vanilla_dir / "summary.json").read_text(encoding="utf-8")
        )
    else:
        vanilla_summary = run_vanilla(
            output_dir=vanilla_dir,
            problems_path=problems_path,
            problems=problems,
            state_bank=state_bank,
            urdf=args.urdf.resolve(),
            scene=scene,
            seed=args.planner_seed,
            planning_time=args.planning_time,
            goal_radius=args.goal_radius,
            dataset=args.dataset.resolve(),
        )
    if (flow_dir / "summary.json").exists():
        flow_summary = json.loads(
            (flow_dir / "summary.json").read_text(encoding="utf-8")
        )
    else:
        flow_summary = run_flow(
            output_dir=flow_dir,
            problems_path=problems_path,
            problems=problems,
            state_bank=state_bank,
            urdf=args.urdf.resolve(),
            scene=scene,
            seed=args.planner_seed,
            planning_time=args.planning_time,
            goal_radius=args.goal_radius,
            dataset=args.dataset.resolve(),
            checkpoint=args.checkpoint.resolve(),
        )

    audit_script = FRANKA_TESTING / "audit_franka_rrt_results.py"
    for label, results_dir in (("vanilla", vanilla_dir), ("flow", flow_dir)):
        run_logged(
            [
                sys.executable,
                str(audit_script),
                "--results-dir",
                str(results_dir),
                "--urdf",
                str(args.urdf.resolve()),
                "--scene-json",
                str(scene_path),
                "--expected-count",
                str(args.num_problems),
            ],
            output_root / f"{label}_audit.log",
        )
    comparison_dir.mkdir(exist_ok=True)
    run_logged(
        [
            sys.executable,
            str(FRANKA_TESTING / "compare_franka_rrt_results.py"),
            "--vanilla-dir",
            str(vanilla_dir),
            "--flow-dir",
            str(flow_dir),
            "--output-dir",
            str(comparison_dir),
        ],
        output_root / "comparison.log",
    )
    paired_path = comparison_dir / "comparison_to_vanilla.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_model": scene,
        "problem_file": str(problems_path),
        "problem_file_sha256": vanilla_benchmark.file_sha256(problems_path),
        "vanilla": {
            "successes": vanilla_summary["num_successes"],
            "success_rate": vanilla_summary["success_rate"],
            "mean_time_all": vanilla_summary["planning_time_seconds_all"]["mean"],
            "median_time_all": vanilla_summary["planning_time_seconds_all"]["median"],
        },
        "flow": {
            "successes": flow_summary["num_successes"],
            "success_rate": flow_summary["success_rate"],
            "mean_time_all": flow_summary["planning_time_seconds_all"]["mean"],
            "median_time_all": flow_summary["planning_time_seconds_all"]["median"],
        },
        "paired_outcomes": paired["paired_outcomes"],
        "path_audits": paired["path_audits"],
    }
    final_path = output_root / "static_obstacle_comparison.json"
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    print(f"static-obstacle comparison saved to {final_path}", flush=True)


if __name__ == "__main__":
    main()
