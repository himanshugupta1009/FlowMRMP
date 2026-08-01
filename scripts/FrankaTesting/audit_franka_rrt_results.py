#!/usr/bin/env python3
"""Audit saved Franka planner paths for limits, self-collision, and goals."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_URDF,
    path_label,
)

if str(FLOWMRMP_SRC) not in sys.path:
    sys.path.insert(0, str(FLOWMRMP_SRC))

from Agents.FrankaPanda import (  # noqa: E402
    DQ_MAX,
    FrankaPanda,
    FrankaSelfCollisionChecker,
    Q_LOWER,
    Q_UPPER,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collision_asset_hashes(urdf_path: Path) -> dict[str, str]:
    robot_dir = urdf_path.resolve().parent
    paths = [urdf_path.resolve(), *sorted(robot_dir.glob("*.srdf"))]
    paths.extend(
        sorted(
            path
            for path in (robot_dir / "meshes" / "collision").rglob("*")
            if path.is_file()
        )
    )
    return {
        path_label(path): file_sha256(path)
        for path in paths
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pybullet"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Require this many unique trials and saved paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_path = results_dir / "summary.json"
    trials_path = results_dir / "trials.csv"
    if not summary_path.is_file() or not trials_path.is_file():
        raise FileNotFoundError("A complete result requires summary.json and trials.csv")
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    with trials_path.open(newline="", encoding="utf-8") as stream:
        trial_rows = list(csv.DictReader(stream))
    trial_by_id = {int(row["problem_id"]): row for row in trial_rows}
    trial_ids = [int(row["problem_id"]) for row in trial_rows]
    trial_id_set = set(trial_ids)
    path_files = sorted((results_dir / "paths").glob("problem_*.npz"))
    if not path_files:
        raise FileNotFoundError(f"No problem paths in {results_dir / 'paths'}")
    checker = FrankaSelfCollisionChecker(args.urdf)
    failures = []
    expected_count = (
        int(args.expected_count)
        if args.expected_count is not None
        else int(summary_data["num_problems"])
    )
    configuration = summary_data.get("configuration", {})
    if configuration.get("goal_metric") != "normalized_7d_joint_position_l2":
        failures.append(
            {
                "reason": (
                    "configuration does not identify the normalized 7D "
                    "joint-position goal metric"
                )
            }
        )
    run_config_path = results_dir / "run_config.json"
    run_config_matches_summary: bool | None = None
    if run_config_path.is_file():
        run_config_data = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config_matches_summary = run_config_data == configuration
        if not run_config_matches_summary:
            failures.append(
                {"reason": "run_config.json differs from summary configuration"}
            )
    elif configuration.get("planner") == "FlowEBRRT":
        failures.append({"reason": "FlowEBRRT result is missing run_config.json"})
    tuning_info = configuration.get("tuning_selection")
    if tuning_info is not None:
        tuning_path = results_dir / str(tuning_info.get("file"))
        if (
            not tuning_path.is_file()
            or file_sha256(tuning_path) != tuning_info.get("sha256")
        ):
            failures.append({"reason": "tuning selection is missing or changed"})
        else:
            tuning_report = json.loads(tuning_path.read_text(encoding="utf-8"))
            selected = tuning_report.get("selected", {})
            if selected != tuning_info.get("selected") or (
                int(selected.get("sample_steps", -1))
                != int(configuration.get("sample_steps", -2))
                or int(selected.get("minimum_prefix_steps", -1))
                != int(configuration.get("minimum_flow_prefix_steps", -2))
                or bool(selected.get("truncate"))
                != bool(configuration.get("flow_prefix_truncation"))
            ):
                failures.append(
                    {"reason": "executed planner flags differ from tuning selection"}
                )
    configured_ids = configuration.get("problem_ids")
    expected_ids = (
        set(int(value) for value in configured_ids)
        if configured_ids is not None
        else set(range(expected_count))
    )
    path_ids = {
        int(path.stem.rsplit("_", 1)[1])
        for path in path_files
    }
    if len(trial_ids) != len(trial_id_set):
        failures.append({"reason": "duplicate problem IDs in trials.csv"})
    if len(trial_rows) != expected_count or trial_id_set != expected_ids:
        failures.append(
            {
                "reason": "trial set is incomplete or unexpected",
                "expected_count": expected_count,
                "actual_count": len(trial_rows),
                "missing_ids": sorted(expected_ids - trial_id_set),
                "extra_ids": sorted(trial_id_set - expected_ids),
            }
        )
    if len(path_files) != expected_count or path_ids != expected_ids:
        failures.append(
            {
                "reason": "saved path set is incomplete or unexpected",
                "expected_count": expected_count,
                "actual_count": len(path_files),
                "missing_ids": sorted(expected_ids - path_ids),
                "extra_ids": sorted(path_ids - expected_ids),
            }
        )
    invalid_success_labels = [
        int(row["problem_id"])
        for row in trial_rows
        if row.get("success", "").strip().lower() not in {"true", "false"}
    ]
    if invalid_success_labels:
        failures.append(
            {
                "reason": "invalid boolean success label",
                "problem_ids": invalid_success_labels,
            }
        )
    csv_successes = sum(row["success"].strip().lower() == "true" for row in trial_rows)
    if (
        int(summary_data.get("num_problems", -1)) != len(trial_rows)
        or int(summary_data.get("num_successes", -1)) != csv_successes
    ):
        failures.append({"reason": "summary.json and trials.csv counts disagree"})
    numeric_trial_fields = (
        "planning_time_seconds",
        "goal_radius",
        "initial_normalized_distance",
        "final_normalized_distance",
    )
    invalid_numeric_ids: set[int] = set()
    for row in trial_rows:
        problem_id = int(row["problem_id"])
        try:
            values = {
                field: float(row[field]) for field in numeric_trial_fields
            }
        except (KeyError, ValueError):
            invalid_numeric_ids.add(problem_id)
            failures.append(
                {"problem_id": problem_id, "reason": "malformed numeric trial field"}
            )
            continue
        if not all(np.isfinite(value) for value in values.values()):
            invalid_numeric_ids.add(problem_id)
            failures.append(
                {"problem_id": problem_id, "reason": "non-finite numeric trial field"}
            )
        if values["planning_time_seconds"] < 0.0:
            invalid_numeric_ids.add(problem_id)
            failures.append(
                {"problem_id": problem_id, "reason": "negative planning time"}
            )
        if values["goal_radius"] <= 0.0:
            invalid_numeric_ids.add(problem_id)
            failures.append(
                {"problem_id": problem_id, "reason": "non-positive goal radius"}
            )
    valid_planning_values = np.asarray(
        [
            float(row["planning_time_seconds"])
            for row in trial_rows
            if int(row["problem_id"]) not in invalid_numeric_ids
        ],
        dtype=np.float64,
    )
    if len(valid_planning_values) == len(trial_rows) and len(trial_rows):
        expected_time_summary = {
            "mean": float(valid_planning_values.mean()),
            "median": float(np.median(valid_planning_values)),
            "min": float(valid_planning_values.min()),
            "max": float(valid_planning_values.max()),
            "p95": float(np.percentile(valid_planning_values, 95)),
        }
        saved_time_summary = summary_data.get("planning_time_seconds_all", {})
        if any(
            key not in saved_time_summary
            or not np.isclose(
                float(saved_time_summary[key]), value, rtol=0.0, atol=1e-9
            )
            for key, value in expected_time_summary.items()
        ):
            failures.append({"reason": "summary planning-time statistics disagree"})
        expected_success_rate = csv_successes / len(trial_rows)
        if not np.isclose(
            float(summary_data.get("success_rate", np.nan)),
            expected_success_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append({"reason": "summary success rate disagrees"})
    planning_budget = configuration.get("planning_time")
    if planning_budget is None:
        failures.append({"reason": "configuration is missing planning-time budget"})
    else:
        planning_budget = float(planning_budget)
        if not np.isfinite(planning_budget) or planning_budget <= 0.0:
            failures.append({"reason": "invalid configured planning-time budget"})
        else:
            overruns = [
                int(row["problem_id"])
                for row in trial_rows
                if int(row["problem_id"]) not in invalid_numeric_ids
                and np.isfinite(float(row["planning_time_seconds"]))
                and float(row["planning_time_seconds"]) > planning_budget + 0.25
            ]
        if np.isfinite(planning_budget) and planning_budget > 0.0 and overruns:
            failures.append(
                {
                    "reason": "planner exceeded time budget by more than 0.25 seconds",
                    "problem_ids": overruns,
                }
            )

    local_problems_path = results_dir / "problems.npz"
    configured_problems = configuration.get("problems")
    if local_problems_path.is_file():
        problems_path = local_problems_path
    elif configured_problems is not None:
        candidate = Path(configured_problems)
        if not candidate.is_absolute():
            candidate = results_dir / candidate
        problems_path = candidate
    else:
        problems_path = local_problems_path
    if not problems_path.is_file():
        failures.append(
            {
                "reason": "canonical A-B problem file is missing",
                "expected_path": str(problems_path),
            }
        )
        problem_starts = problem_goals = np.empty((0, 14), dtype=np.float64)
        problems_sha256 = None
    else:
        problems_sha256 = file_sha256(problems_path)
        expected_problem_sha256 = configuration.get("problems_sha256")
        if (
            configuration.get("planner") == "FlowEBRRT"
            and expected_problem_sha256 is None
        ):
            failures.append(
                {"reason": "FlowEBRRT configuration is missing problems SHA-256"}
            )
        if (
            expected_problem_sha256 is not None
            and problems_sha256 != expected_problem_sha256
        ):
            failures.append(
                {
                    "reason": "canonical A-B problem file hash mismatch",
                    "expected_sha256": expected_problem_sha256,
                    "actual_sha256": problems_sha256,
                }
            )
        with np.load(problems_path) as problem_data:
            problem_starts = np.asarray(problem_data["starts"], dtype=np.float64)
            problem_goals = np.asarray(problem_data["goals"], dtype=np.float64)
    problem_arrays_valid = bool(
        problem_starts.ndim == 2
        and problem_goals.ndim == 2
        and problem_starts.shape == problem_goals.shape
        and problem_starts.shape[1:] == (14,)
        and len(problem_starts) > max(expected_ids, default=-1)
        and np.isfinite(problem_starts).all()
        and np.isfinite(problem_goals).all()
    )
    if not problem_arrays_valid:
        failures.append({"reason": "canonical A-B arrays are malformed or incomplete"})
    total_states = 0
    successful_paths = 0
    try:
        if problem_arrays_valid:
            for label, states_to_check in (
                ("start", problem_starts[list(sorted(expected_ids))]),
                ("goal", problem_goals[list(sorted(expected_ids))]),
            ):
                for row_index, state in zip(sorted(expected_ids), states_to_check):
                    if not (
                        np.all(state[:7] >= Q_LOWER)
                        and np.all(state[:7] <= Q_UPPER)
                        and np.all(np.abs(state[7:]) <= DQ_MAX)
                    ):
                        failures.append(
                            {
                                "problem_id": int(row_index),
                                "reason": f"canonical {label} violates q/dq limits",
                            }
                        )
                    elif checker.in_collision(state[:7]):
                        failures.append(
                            {
                                "problem_id": int(row_index),
                                "reason": f"canonical {label} self-collides",
                            }
                        )
        for path_file in path_files:
            with np.load(path_file) as data:
                states = np.asarray(data["states"], dtype=np.float64)
                start = np.asarray(data["start"], dtype=np.float64)
                goal = np.asarray(data["goal"], dtype=np.float64)
                success = bool(data["success"])
                goal_radius = float(data["goal_radius"])
            if states.ndim != 2 or states.shape[1] != 14 or not len(states):
                failures.append({"file": path_file.name, "reason": "malformed path"})
                continue
            problem_id = int(path_file.stem.rsplit("_", 1)[1])
            trial_row = trial_by_id.get(problem_id)
            if trial_row is None:
                failures.append(
                    {"file": path_file.name, "reason": "path has no trial row"}
                )
                continue
            if problem_id in invalid_numeric_ids:
                continue
            if not np.isfinite(start).all() or not np.isfinite(goal).all():
                failures.append(
                    {"file": path_file.name, "reason": "non-finite start or goal"}
                )
                continue
            if success != (trial_row["success"].strip().lower() == "true"):
                failures.append(
                    {"file": path_file.name, "reason": "path/trial success mismatch"}
                )
                continue
            if not np.isclose(goal_radius, float(trial_row["goal_radius"])):
                failures.append(
                    {"file": path_file.name, "reason": "path/trial goal radius mismatch"}
                )
                continue
            recomputed_initial_distance = float(
                FrankaPanda.get_goal_distance(start, goal)
            )
            if not np.isclose(
                recomputed_initial_distance,
                float(trial_row["initial_normalized_distance"]),
                rtol=0.0,
                atol=1e-9,
            ):
                failures.append(
                    {"file": path_file.name, "reason": "CSV initial distance mismatch"}
                )
                continue
            if not np.allclose(states[0], start, rtol=0.0, atol=1e-10):
                failures.append(
                    {"file": path_file.name, "reason": "path does not begin at start"}
                )
                continue
            if problem_id >= len(problem_starts) or problem_id >= len(problem_goals):
                failures.append(
                    {"file": path_file.name, "reason": "problem ID absent from problems file"}
                )
                continue
            if len(problem_starts):
                if not np.allclose(start, problem_starts[problem_id], rtol=0.0, atol=1e-10):
                    failures.append(
                        {"file": path_file.name, "reason": "start differs from problems file"}
                    )
                    continue
                if not np.allclose(goal, problem_goals[problem_id], rtol=0.0, atol=1e-10):
                    failures.append(
                        {"file": path_file.name, "reason": "goal differs from problems file"}
                    )
                    continue
            total_states += len(states)
            if not np.isfinite(states).all():
                failures.append({"file": path_file.name, "reason": "non-finite state"})
                continue
            invalid_limit = next(
                (
                    index
                    for index, state in enumerate(states)
                    if not (
                        np.all(state[:7] >= Q_LOWER)
                        and np.all(state[:7] <= Q_UPPER)
                        and np.all(np.abs(state[7:]) <= DQ_MAX)
                    )
                ),
                None,
            )
            if invalid_limit is not None:
                failures.append(
                    {
                        "file": path_file.name,
                        "reason": "joint/velocity limit",
                        "waypoint": invalid_limit,
                    }
                )
                continue
            collision = next(
                (
                    index
                    for index, state in enumerate(states)
                    if checker.in_collision(state[:7])
                ),
                None,
            )
            if collision is not None:
                failures.append(
                    {
                        "file": path_file.name,
                        "reason": "self-collision",
                        "waypoint": collision,
                    }
                )
                continue
            final_distance = float(
                FrankaPanda.get_goal_distance(states[-1], goal)
            )
            if not np.isclose(
                final_distance,
                float(trial_row["final_normalized_distance"]),
                rtol=0.0,
                atol=1e-9,
            ):
                failures.append(
                    {"file": path_file.name, "reason": "CSV final distance mismatch"}
                )
                continue
            if success:
                successful_paths += 1
                if final_distance > goal_radius + 1e-9:
                    failures.append(
                        {
                            "file": path_file.name,
                            "reason": "successful path ends outside goal",
                            "final_distance": final_distance,
                            "goal_radius": goal_radius,
                        }
                    )
    finally:
        checker.close()

    collision_hashes = collision_asset_hashes(args.urdf)
    configured_collision_hashes = configuration.get("collision_asset_sha256")
    if configuration.get("planner") == "FlowEBRRT":
        if configured_collision_hashes is None:
            failures.append(
                {"reason": "FlowEBRRT configuration lacks collision-asset hashes"}
            )
        elif configured_collision_hashes != collision_hashes:
            failures.append(
                {"reason": "collision assets differ from benchmark configuration"}
            )
    configured_urdf_sha256 = configuration.get("urdf_sha256")
    actual_urdf_sha256 = file_sha256(args.urdf)
    if (
        configured_urdf_sha256 is not None
        and configured_urdf_sha256 != actual_urdf_sha256
    ):
        failures.append({"reason": "URDF differs from benchmark configuration"})

    artifact_hashes = {
        "summary.json": file_sha256(summary_path),
        "trials.csv": file_sha256(trials_path),
        "problems.npz": problems_sha256,
        "paths": {
            str(path.relative_to(results_dir)): file_sha256(path)
            for path in path_files
        },
    }
    if run_config_path.is_file():
        artifact_hashes["run_config.json"] = file_sha256(run_config_path)
    tuning_selection_path = results_dir / "tuning_selection.json"
    if tuning_selection_path.is_file():
        artifact_hashes["tuning_selection.json"] = file_sha256(
            tuning_selection_path
        )
    source_hashes = {
        path_label(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
        path_label(FLOWMRMP_SRC / "Agents" / "FrankaPanda.py"): file_sha256(
            FLOWMRMP_SRC / "Agents" / "FrankaPanda.py"
        ),
    }

    summary = {
        "results_dir": str(results_dir),
        "num_paths": len(path_files),
        "expected_paths": expected_count,
        "num_successful_paths": successful_paths,
        "total_states_checked": total_states,
        "problems_file": str(problems_path),
        "problems_sha256": problems_sha256,
        "run_config_matches_summary": run_config_matches_summary,
        "audited_artifact_sha256": artifact_hashes,
        "collision_asset_sha256": collision_hashes,
        "audit_runtime_source_file_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "canonical_problem_provenance_verified": bool(
            problems_sha256 is not None
            and not any(
                "problem" in str(failure.get("reason", ""))
                or "canonical" in str(failure.get("reason", ""))
                or "start differs" in str(failure.get("reason", ""))
                or "goal differs" in str(failure.get("reason", ""))
                for failure in failures
            )
        ),
        "all_paths_valid": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "checks": [
            "finite 14D states",
            "Franka joint limits",
            "Franka velocity limits",
            "PyBullet self-collision",
            "successful endpoint within normalized goal radius",
            "trials/paths/summary/run-config completeness and numeric agreement",
            "saved path starts plus canonical finite, limit-valid, self-collision-free A-B provenance",
            "per-problem planner time budget (0.25-second tolerance)",
            "collision assets and immutable result artifact fingerprints",
        ],
    }
    output = results_dir / "path_validity_audit.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Path audit failed for {len(failures)} paths")


if __name__ == "__main__":
    main()
