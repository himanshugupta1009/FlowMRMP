#!/usr/bin/env python3
"""Independently audit a witness-backed Franka A-to-B problem set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_BENCHMARK_PROBLEMS as DEFAULT_PROBLEMS,
    DEFAULT_URDF,
)

MAIN_SCRIPTS = Path(__file__).resolve().parents[1]
for module_path in (MAIN_SCRIPTS, FLOWMRMP_SRC):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import (  # noqa: E402
    DDQ_MAX,
    DQ_MAX,
    Q_LOWER,
    Q_UPPER,
    FrankaPanda,
    FrankaSelfCollisionChecker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rollout(start: np.ndarray, actions: np.ndarray, dt: float) -> np.ndarray:
    states = np.empty((len(actions) + 1, 14), dtype=np.float64)
    states[0] = start
    for index, action in enumerate(actions):
        current = states[index]
        states[index + 1, :7] = (
            current[:7] + current[7:] * dt + 0.5 * action * dt**2
        )
        states[index + 1, 7:] = current[7:] + action * dt
    return states


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    path = args.problems.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else path.with_suffix(".independent_audit.json")
    )
    failures: list[dict[str, object]] = []
    with np.load(path) as data:
        required = {
            "starts",
            "goals",
            "start_trajectory",
            "goal_trajectory",
            "difficulty_tier",
            "witness_dt",
            "witness_state_offsets",
            "witness_states",
            "witness_action_offsets",
            "witness_actions",
            "witness_initial_normalized_distance",
            "recommended_goal_radius",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Problem file is missing arrays: {missing}")
        starts = np.asarray(data["starts"], dtype=np.float64)
        goals = np.asarray(data["goals"], dtype=np.float64)
        trajectories = np.asarray(data["start_trajectory"], dtype=np.int64)
        goal_trajectories = np.asarray(data["goal_trajectory"], dtype=np.int64)
        tiers = np.asarray(data["difficulty_tier"]).astype(str)
        dt = float(data["witness_dt"])
        state_offsets = np.asarray(data["witness_state_offsets"], dtype=np.int64)
        all_states = np.asarray(data["witness_states"], dtype=np.float64)
        action_offsets = np.asarray(data["witness_action_offsets"], dtype=np.int64)
        all_actions = np.asarray(data["witness_actions"], dtype=np.float64)
        saved_distances = np.asarray(
            data["witness_initial_normalized_distance"], dtype=np.float64
        )
        goal_radius = float(data["recommended_goal_radius"])

    count = len(starts)
    if count != args.expected_count:
        failures.append({"reason": "unexpected problem count", "actual": count})
    if starts.shape != (count, 14) or goals.shape != starts.shape:
        failures.append({"reason": "malformed start/goal arrays"})
    if len(set(trajectories.tolist())) != count:
        failures.append({"reason": "source trajectories are not unique"})
    if not np.array_equal(trajectories, goal_trajectories):
        failures.append({"reason": "a problem crosses source trajectories"})
    if len(state_offsets) != count + 1 or len(action_offsets) != count + 1:
        failures.append({"reason": "malformed witness offsets"})

    checker = FrankaSelfCollisionChecker(args.urdf)
    max_rollout_residual = 0.0
    total_witness_states = 0
    actual_distances = []
    durations = []
    try:
        for problem_id in range(count):
            states = all_states[state_offsets[problem_id] : state_offsets[problem_id + 1]]
            actions = all_actions[
                action_offsets[problem_id] : action_offsets[problem_id + 1]
            ]
            total_witness_states += len(states)
            if len(states) != len(actions) + 1 or len(actions) == 0:
                failures.append({"problem_id": problem_id, "reason": "bad witness length"})
                continue
            reintegrated = rollout(starts[problem_id], actions, dt)
            residual = float(np.max(np.abs(reintegrated - states)))
            max_rollout_residual = max(max_rollout_residual, residual)
            if residual > 2e-5:
                failures.append(
                    {"problem_id": problem_id, "reason": "witness reintegration mismatch", "max_abs": residual}
                )
            if not np.allclose(states[0], starts[problem_id], atol=2e-6, rtol=0.0):
                failures.append({"problem_id": problem_id, "reason": "witness start mismatch"})
            if not np.allclose(states[-1], goals[problem_id], atol=2e-5, rtol=0.0):
                failures.append({"problem_id": problem_id, "reason": "witness goal mismatch"})
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                failures.append({"problem_id": problem_id, "reason": "non-finite witness"})
                continue
            if np.max(np.abs(actions) - DDQ_MAX) > 1e-5:
                failures.append({"problem_id": problem_id, "reason": "acceleration limit violation"})
            q = states[:, :7]
            dq = states[:, 7:]
            if not (
                np.all(q >= Q_LOWER - 1e-9)
                and np.all(q <= Q_UPPER + 1e-9)
                and np.all(np.abs(dq) <= DQ_MAX + 1e-9)
            ):
                failures.append({"problem_id": problem_id, "reason": "q/dq limit violation"})
            for state_index, state in enumerate(states):
                if checker.in_collision(state[:7]):
                    failures.append(
                        {
                            "problem_id": problem_id,
                            "state_index": state_index,
                            "reason": "self-collision in witness",
                        }
                    )
                    break
            distance = float(
                np.linalg.norm(
                    FrankaPanda.normalize_state(starts[problem_id])
                    - FrankaPanda.normalize_state(goals[problem_id])
                )
            )
            actual_distances.append(distance)
            durations.append(len(actions) * dt)
            if distance <= goal_radius:
                failures.append({"problem_id": problem_id, "reason": "problem starts inside goal radius"})
            if not np.isclose(distance, saved_distances[problem_id], atol=2e-6, rtol=0.0):
                failures.append({"problem_id": problem_id, "reason": "saved distance mismatch"})
    finally:
        checker.close()

    actual_distances_array = np.asarray(actual_distances, dtype=np.float64)
    durations_array = np.asarray(durations, dtype=np.float64)
    tier_counts = {
        tier: int(np.sum(tiers == tier)) for tier in sorted(set(tiers.tolist()))
    }
    report = {
        "all_checks_passed": not failures,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "problem_file": str(path),
        "problem_file_bytes": path.stat().st_size,
        "problem_file_sha256": file_sha256(path),
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": file_sha256(args.urdf.resolve()),
        "num_problems": count,
        "unique_source_trajectories": len(set(trajectories.tolist())),
        "tier_counts": tier_counts,
        "goal_radius": goal_radius,
        "total_witness_states_checked": total_witness_states,
        "max_witness_reintegration_abs_residual": max_rollout_residual,
        "initial_normalized_distance": percentile_summary(actual_distances_array),
        "witness_duration_seconds": percentile_summary(durations_array),
        "checks": {
            "exact_dynamics_witness": True,
            "finite": True,
            "joint_velocity_acceleration_limits": True,
            "self_collision_at_every_witness_state": True,
            "unique_same_trajectory_sources": True,
            "nontrivial_outside_goal_radius": True,
        },
        "failures": failures,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print(f"independent audit saved to {output}")


if __name__ == "__main__":
    main()
