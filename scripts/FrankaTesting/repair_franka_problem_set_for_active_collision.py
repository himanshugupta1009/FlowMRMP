#!/usr/bin/env python3
"""Reuse a Franka witness set while truncating paths at new-model collisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from franka_paths import (
    CORE_RRT_SRC,
    DEFAULT_RAW_TRAJECTORY_DATASET,
    DEFAULT_URDF,
)

MAIN_SCRIPTS = Path(__file__).resolve().parents[1]
for module_path in (MAIN_SCRIPTS, CORE_RRT_SRC):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import FrankaPanda, FrankaSelfCollisionChecker  # noqa: E402
from create_franka_reachable_problem_set import (  # noqa: E402
    file_sha256,
    normalized_limit_margin,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "benchmarks" / "franka_reachable_morphit_ab_100.npz"
DEFAULT_OUTPUT = (
    ROOT / "data" / "benchmarks" / "franka_reachable_morphit295_ab_100_goal_r040.npz"
)


def parse_args() -> argparse.Namespace:
    """Parse source, dataset, output, and goal-radius arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RAW_TRAJECTORY_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    return parser.parse_args()


def difficulty_tier(distance: float) -> str:
    """Classify the same normalized position-distance ranges as the source set."""
    if 0.65 <= distance < 1.0:
        return "easy"
    if 1.0 <= distance < 1.4:
        return "medium"
    if 1.4 <= distance < 1.8:
        return "hard"
    return "repaired"


def main() -> None:
    """Truncate only colliding witnesses and preserve every unaffected problem."""
    args = parse_args()
    input_path = args.input.resolve()
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".repair.json")
    )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing problem set: {output_path}"
        )
    with np.load(input_path) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}

    state_offsets = arrays["witness_state_offsets"]
    action_offsets = arrays["witness_action_offsets"]
    all_states = arrays["witness_states"]
    all_actions = arrays["witness_actions"]
    witness_dt = float(arrays["witness_dt"])
    num_problems = len(arrays["starts"])
    checker = FrankaSelfCollisionChecker(args.urdf)
    repaired_states: list[np.ndarray] = []
    repaired_actions: list[np.ndarray] = []
    new_state_offsets = [0]
    new_action_offsets = [0]
    changes: list[dict[str, object]] = []
    try:
        with h5py.File(dataset_path, "r") as dataset:
            for problem_id in range(num_problems):
                state_start, state_stop = state_offsets[problem_id : problem_id + 2]
                action_start, action_stop = action_offsets[problem_id : problem_id + 2]
                states = np.asarray(
                    all_states[int(state_start) : int(state_stop)], dtype=np.float64
                )
                actions = np.asarray(
                    all_actions[int(action_start) : int(action_stop)], dtype=np.float64
                )
                valid = checker.collision_free_mask(states[:, :7])
                invalid_ids = np.flatnonzero(~valid)
                if len(invalid_ids):
                    first_invalid = int(invalid_ids[0])
                    if first_invalid < 2:
                        raise RuntimeError(
                            f"Problem {problem_id} collides too early to preserve"
                        )
                    old_state_count = len(states)
                    states = states[:first_invalid]
                    actions = actions[: first_invalid - 1]
                    new_goal = states[-1].copy()
                    distance = FrankaPanda.get_goal_distance(states[0], new_goal)
                    if distance <= args.goal_radius:
                        raise RuntimeError(
                            f"Problem {problem_id} repair would begin inside goal radius"
                        )
                    trajectory_id = int(arrays["start_trajectory"][problem_id])
                    goal_timestep = int(arrays["start_timestep"][problem_id]) + len(
                        actions
                    )
                    trajectory = dataset[str(trajectory_id)]
                    source_goal = np.concatenate(
                        (
                            np.asarray(trajectory["positions"][goal_timestep]),
                            np.asarray(trajectory["velocities"][goal_timestep]),
                        )
                    )
                    arrays["goals"][problem_id] = new_goal
                    arrays["source_goals"][problem_id] = source_goal
                    arrays["goal_trajectory"][problem_id] = trajectory_id
                    arrays["goal_timestep"][problem_id] = goal_timestep
                    arrays["difficulty_tier"][problem_id] = difficulty_tier(distance)
                    arrays["witness_duration_seconds"][problem_id] = (
                        len(actions) * witness_dt
                    )
                    arrays["witness_steps"][problem_id] = len(actions)
                    arrays["witness_initial_normalized_distance"][problem_id] = distance
                    arrays["witness_minimum_normalized_limit_margin"][problem_id] = (
                        normalized_limit_margin(states)
                    )
                    arrays["source_endpoint_position_error"][problem_id] = float(
                        np.linalg.norm(new_goal[:7] - source_goal[:7])
                    )
                    arrays["source_endpoint_velocity_error"][problem_id] = float(
                        np.linalg.norm(new_goal[7:] - source_goal[7:])
                    )
                    changes.append(
                        {
                            "problem_id": problem_id,
                            "first_invalid_original_state": first_invalid,
                            "old_witness_states": old_state_count,
                            "new_witness_states": len(states),
                            "new_witness_actions": len(actions),
                            "new_goal_timestep": goal_timestep,
                            "new_initial_normalized_distance": distance,
                            "new_difficulty_tier": str(
                                arrays["difficulty_tier"][problem_id]
                            ),
                        }
                    )
                repaired_states.append(states.astype(all_states.dtype, copy=False))
                repaired_actions.append(actions.astype(all_actions.dtype, copy=False))
                new_state_offsets.append(new_state_offsets[-1] + len(states))
                new_action_offsets.append(new_action_offsets[-1] + len(actions))
    finally:
        checker.close()

    arrays["witness_states"] = np.concatenate(repaired_states, axis=0)
    arrays["witness_actions"] = np.concatenate(repaired_actions, axis=0)
    arrays["witness_state_offsets"] = np.asarray(new_state_offsets, dtype=np.int64)
    arrays["witness_action_offsets"] = np.asarray(new_action_offsets, dtype=np.int64)
    arrays["recommended_goal_radius"] = np.asarray(args.goal_radius, dtype=np.float64)
    arrays["created_utc"] = np.asarray(datetime.now(timezone.utc).isoformat())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "goal_radius": args.goal_radius,
        "num_problems": num_problems,
        "unchanged_problem_count": num_problems - len(changes),
        "changed_problem_count": len(changes),
        "changes": changes,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
