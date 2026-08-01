#!/usr/bin/env python3
"""Create a diverse, witness-backed Franka A-to-B benchmark problem set.

Each problem starts from a state in one CuRobo trajectory and follows a stored
acceleration subsequence through the exact dynamics used by the Franka RRT
adapter. The integrated endpoint becomes the benchmark goal. Consequently,
every selected problem has an explicit limit-valid, self-collision-free
kinodynamic witness in the same empty-world model used by both planners.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_BENCHMARK_PROBLEMS_OUTPUT as DEFAULT_OUTPUT,
    DEFAULT_RAW_TRAJECTORY_DATASET as DEFAULT_DATASET,
    DEFAULT_URDF,
)

if str(FLOWMRMP_SRC) not in sys.path:
    sys.path.insert(0, str(FLOWMRMP_SRC))

from Agents.FrankaPanda import (  # noqa: E402
    DDQ_MAX,
    DQ_MAX,
    Q_LOWER,
    Q_UPPER,
    FrankaPanda,
    FrankaSelfCollisionChecker,
)


@dataclass(frozen=True)
class Tier:
    name: str
    fraction: float
    distance_min: float
    distance_max: float
    action_min: int
    action_max: int


TIERS = (
    Tier("easy", 0.40, 0.65, 1.00, 25, 60),
    Tier("medium", 0.40, 1.00, 1.40, 45, 90),
    Tier("hard", 0.20, 1.40, 1.80, 70, 120),
)
GOAL_RADIUS = 0.50
MIN_NORMALIZED_LIMIT_MARGIN = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--attempts-per-trajectory", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_counts(total: int) -> dict[str, int]:
    if total < len(TIERS):
        raise ValueError(f"num-problems must be at least {len(TIERS)}")
    counts: dict[str, int] = {}
    assigned = 0
    for tier in TIERS[:-1]:
        count = int(round(total * tier.fraction))
        counts[tier.name] = count
        assigned += count
    counts[TIERS[-1].name] = total - assigned
    return counts


def rollout(start: np.ndarray, actions: np.ndarray, dt: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    states = np.empty((len(actions) + 1, 14), dtype=np.float64)
    states[0] = np.asarray(start, dtype=np.float64)
    for index, action in enumerate(actions):
        current = states[index]
        states[index + 1, :7] = (
            current[:7] + current[7:] * dt + 0.5 * action * dt**2
        )
        states[index + 1, 7:] = current[7:] + action * dt
    return states


def normalized_limit_margin(states: np.ndarray) -> float:
    q = states[:, :7]
    dq = states[:, 7:]
    q_range = Q_UPPER - Q_LOWER
    margins = (
        (q - Q_LOWER) / q_range,
        (Q_UPPER - q) / q_range,
        1.0 - np.abs(dq) / DQ_MAX,
    )
    return float(min(np.min(values) for values in margins))


def is_collision_free(
    states: np.ndarray, checker: FrankaSelfCollisionChecker
) -> bool:
    return all(not checker.in_collision(state[:7]) for state in states)


def percentile_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def farthest_point_select(candidates: list[dict], count: int) -> list[dict]:
    features = []
    for candidate in candidates:
        start_norm = FrankaPanda.normalize_state(candidate["start"])
        goal_norm = FrankaPanda.normalize_state(candidate["goal"])
        features.append(
            np.concatenate(
                (
                    start_norm,
                    goal_norm,
                    [candidate["duration_seconds"] / 2.4],
                    [candidate["initial_normalized_distance"] / 1.8],
                )
            )
        )
    points = np.asarray(features, dtype=np.float64)
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    selected = [first]
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    min_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(min_distance))
        selected.append(index)
        distance = np.sum((points - points[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    return [candidates[index] for index in selected]


def main() -> None:
    args = parse_args()
    if args.num_problems <= 0:
        raise ValueError("num-problems must be positive")
    if args.candidate_multiplier < 1 or args.attempts_per_trajectory < 1:
        raise ValueError("candidate-multiplier and attempts-per-trajectory must be positive")

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    audit_path = (
        args.audit_output.resolve()
        if args.audit_output is not None
        else output_path.with_suffix(".audit.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = allocate_counts(args.num_problems)
    pool_targets = {
        name: max(count * args.candidate_multiplier, count + 20)
        for name, count in counts.items()
    }
    pools: dict[str, list[dict]] = {tier.name: [] for tier in TIERS}
    rng = np.random.default_rng(args.seed)
    checker = FrankaSelfCollisionChecker(args.urdf)
    attempted_segments = 0
    rejected = {
        "distance": 0,
        "nonfinite": 0,
        "acceleration_limit": 0,
        "state_limit_or_margin": 0,
        "self_collision": 0,
    }

    try:
        with h5py.File(dataset_path, "r") as source:
            dataset_dt = float(source.attrs.get("interpolation_dt", 0.02))
            if not np.isclose(dataset_dt, 0.02, rtol=0.0, atol=1e-12):
                raise ValueError(f"Expected dataset dt=0.02, found {dataset_dt}")
            trajectory_names = np.asarray(sorted(source.keys()))
            for name in trajectory_names[rng.permutation(len(trajectory_names))]:
                if all(len(pools[key]) >= pool_targets[key] for key in pools):
                    break
                group = source[str(name)]
                q = np.asarray(group["positions"], dtype=np.float64)
                dq = np.asarray(group["velocities"], dtype=np.float64)
                ddq = np.asarray(group["accelerations"], dtype=np.float64)
                if not (q.shape == dq.shape == ddq.shape and q.ndim == 2 and q.shape[1] == 7):
                    continue
                available_tiers = [
                    tier for tier in TIERS
                    if len(pools[tier.name]) < pool_targets[tier.name]
                    and len(q) > tier.action_min
                ]
                rng.shuffle(available_tiers)
                accepted = False
                for tier in available_tiers:
                    maximum = min(tier.action_max, len(q) - 1)
                    if maximum < tier.action_min:
                        continue
                    for _ in range(args.attempts_per_trajectory):
                        attempted_segments += 1
                        action_count = int(rng.integers(tier.action_min, maximum + 1))
                        start_index = int(rng.integers(0, len(q) - action_count))
                        actions = ddq[start_index : start_index + action_count]
                        start = np.concatenate((q[start_index], dq[start_index]))
                        if not np.isfinite(start).all() or not np.isfinite(actions).all():
                            rejected["nonfinite"] += 1
                            continue
                        if np.max(np.abs(actions) - DDQ_MAX) > 1e-6:
                            rejected["acceleration_limit"] += 1
                            continue
                        witness = rollout(start, actions, dataset_dt)
                        if not np.isfinite(witness).all():
                            rejected["nonfinite"] += 1
                            continue
                        margin = normalized_limit_margin(witness)
                        if margin < MIN_NORMALIZED_LIMIT_MARGIN:
                            rejected["state_limit_or_margin"] += 1
                            continue
                        goal = witness[-1].copy()
                        distance = float(
                            np.linalg.norm(
                                FrankaPanda.normalize_state(start)
                                - FrankaPanda.normalize_state(goal)
                            )
                        )
                        if not tier.distance_min <= distance < tier.distance_max:
                            rejected["distance"] += 1
                            continue
                        if not is_collision_free(witness, checker):
                            rejected["self_collision"] += 1
                            continue
                        source_goal = np.concatenate(
                            (q[start_index + action_count], dq[start_index + action_count])
                        )
                        pools[tier.name].append(
                            {
                                "tier": tier.name,
                                "trajectory": int(name),
                                "start_timestep": start_index,
                                "goal_timestep": start_index + action_count,
                                "start": start,
                                "goal": goal,
                                "source_goal": source_goal,
                                "actions": actions.copy(),
                                "witness": witness,
                                "duration_seconds": action_count * dataset_dt,
                                "initial_normalized_distance": distance,
                                "minimum_normalized_limit_margin": margin,
                                "source_endpoint_position_error": float(
                                    np.linalg.norm(goal[:7] - source_goal[:7])
                                ),
                                "source_endpoint_velocity_error": float(
                                    np.linalg.norm(goal[7:] - source_goal[7:])
                                ),
                            }
                        )
                        accepted = True
                        break
                    if accepted:
                        break
            if not all(len(pools[name]) >= pool_targets[name] for name in pools):
                status = {name: len(values) for name, values in pools.items()}
                raise RuntimeError(
                    f"Insufficient valid candidates: {status}; targets={pool_targets}"
                )
    finally:
        checker.close()

    selected: list[dict] = []
    for tier in TIERS:
        selected.extend(farthest_point_select(pools[tier.name], counts[tier.name]))
    if len({item["trajectory"] for item in selected}) != len(selected):
        raise RuntimeError("Selected problems do not use unique source trajectories")

    witness_offsets = [0]
    action_offsets = [0]
    witness_parts = []
    action_parts = []
    for item in selected:
        witness_parts.append(item["witness"].astype(np.float32))
        action_parts.append(item["actions"].astype(np.float32))
        witness_offsets.append(witness_offsets[-1] + len(item["witness"]))
        action_offsets.append(action_offsets[-1] + len(item["actions"]))

    created_utc = datetime.now(timezone.utc).isoformat()
    np.savez_compressed(
        output_path,
        format_name=np.asarray("franka_reachable_witness_problem_set"),
        format_version=np.asarray(1, dtype=np.int32),
        created_utc=np.asarray(created_utc),
        seed=np.asarray(args.seed, dtype=np.int64),
        witness_dt=np.asarray(0.02, dtype=np.float64),
        recommended_goal_radius=np.asarray(GOAL_RADIUS, dtype=np.float64),
        starts=np.stack([item["start"] for item in selected]).astype(np.float64),
        goals=np.stack([item["goal"] for item in selected]).astype(np.float64),
        source_goals=np.stack([item["source_goal"] for item in selected]).astype(np.float64),
        start_trajectory=np.asarray([item["trajectory"] for item in selected], dtype=np.int32),
        goal_trajectory=np.asarray([item["trajectory"] for item in selected], dtype=np.int32),
        start_timestep=np.asarray([item["start_timestep"] for item in selected], dtype=np.int32),
        goal_timestep=np.asarray([item["goal_timestep"] for item in selected], dtype=np.int32),
        difficulty_tier=np.asarray([item["tier"] for item in selected]),
        witness_duration_seconds=np.asarray([item["duration_seconds"] for item in selected]),
        witness_steps=np.asarray([len(item["actions"]) for item in selected], dtype=np.int32),
        witness_initial_normalized_distance=np.asarray(
            [item["initial_normalized_distance"] for item in selected]
        ),
        witness_minimum_normalized_limit_margin=np.asarray(
            [item["minimum_normalized_limit_margin"] for item in selected]
        ),
        source_endpoint_position_error=np.asarray(
            [item["source_endpoint_position_error"] for item in selected]
        ),
        source_endpoint_velocity_error=np.asarray(
            [item["source_endpoint_velocity_error"] for item in selected]
        ),
        witness_state_offsets=np.asarray(witness_offsets, dtype=np.int64),
        witness_states=np.concatenate(witness_parts, axis=0),
        witness_action_offsets=np.asarray(action_offsets, dtype=np.int64),
        witness_actions=np.concatenate(action_parts, axis=0),
    )

    tier_summaries = {}
    for tier in TIERS:
        items = [item for item in selected if item["tier"] == tier.name]
        tier_summaries[tier.name] = {
            "count": len(items),
            "distance_range": [tier.distance_min, tier.distance_max],
            "action_range": [tier.action_min, tier.action_max],
            "initial_normalized_distance": percentile_summary(
                [item["initial_normalized_distance"] for item in items]
            ),
            "witness_duration_seconds": percentile_summary(
                [item["duration_seconds"] for item in items]
            ),
        }
    audit = {
        "all_checks_passed": True,
        "format_name": "franka_reachable_witness_problem_set",
        "format_version": 1,
        "created_utc": created_utc,
        "problem_file": str(output_path),
        "problem_file_bytes": output_path.stat().st_size,
        "problem_file_sha256": file_sha256(output_path),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": file_sha256(dataset_path),
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": file_sha256(args.urdf.resolve()),
        "num_problems": len(selected),
        "unique_source_trajectories": len({item["trajectory"] for item in selected}),
        "selection_is_planner_outcome_independent": True,
        "selection_method": (
            "physics-only filtering followed by farthest-point sampling in "
            "normalized start/goal/duration/difficulty feature space"
        ),
        "reachability_contract": (
            "goal is the endpoint of the included acceleration sequence under the "
            "exact Franka RRT integration model at dt=0.02 s"
        ),
        "quality_checks": {
            "finite_start_goal_actions_and_witness": True,
            "acceleration_limits": True,
            "joint_and_velocity_limits_at_every_witness_state": True,
            "minimum_normalized_limit_margin": MIN_NORMALIZED_LIMIT_MARGIN,
            "self_collision_free_at_every_witness_state": True,
            "initial_distance_strictly_above_goal_radius": True,
            "unique_source_trajectory_per_problem": True,
        },
        "recommended_goal_radius": GOAL_RADIUS,
        "tier_summaries": tier_summaries,
        "overall": {
            "initial_normalized_distance": percentile_summary(
                [item["initial_normalized_distance"] for item in selected]
            ),
            "witness_duration_seconds": percentile_summary(
                [item["duration_seconds"] for item in selected]
            ),
            "minimum_normalized_limit_margin": percentile_summary(
                [item["minimum_normalized_limit_margin"] for item in selected]
            ),
            "source_endpoint_position_error": percentile_summary(
                [item["source_endpoint_position_error"] for item in selected]
            ),
            "source_endpoint_velocity_error": percentile_summary(
                [item["source_endpoint_velocity_error"] for item in selected]
            ),
        },
        "candidate_generation": {
            "attempted_segments": attempted_segments,
            "accepted_pool_sizes": {name: len(values) for name, values in pools.items()},
            "pool_targets": pool_targets,
            "rejections": rejected,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"problem set saved to {output_path}")
    print(f"construction audit saved to {audit_path}")


if __name__ == "__main__":
    main()
