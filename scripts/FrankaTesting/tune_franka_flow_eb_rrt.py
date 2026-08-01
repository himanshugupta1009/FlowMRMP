#!/usr/bin/env python3
"""Run a predeclared small FlowEBRRT grid and select one final configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from franka_paths import (
    DEFAULT_BENCHMARK_PROBLEMS as DEFAULT_PROBLEMS,
    DEFAULT_CHECKPOINT,
    FLOWMRMP_ROOT,
    path_label,
)

DEFAULT_OUTPUT = FLOWMRMP_ROOT / "results" / "franka_flow_eb_rrt" / "tuning_v3"
TUNING_PROBLEM_IDS = (4, 10, 17, 37, 41, 57, 64, 81)
FIXED_BENCHMARK_KEYS = (
    "checkpoint_sha256",
    "problems_sha256",
    "dataset_sha256",
    "urdf_sha256",
    "collision_asset_sha256",
    "source_file_sha256",
    "device",
    "state_bank_size",
    "seed",
    "goal_radius",
    "goal_sampling_probability",
    "max_random_edge_time",
    "acceleration_scale",
    "max_iterations",
    "flow_prefetch_batch_size",
    "num_sorted_edge_trials",
    "num_random_edges",
    "epsilon_random",
    "integration_dt",
    "flow_training_dataset_sha256",
    "environment",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planning-time", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_completed_run(
    run_dir: Path,
    config: dict[str, object],
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    problems_sha256: str,
    expected_device: str,
    expected_source_hashes: dict[str, str],
    expected_dataset_sha256: str,
    expected_urdf_sha256: str,
    expected_collision_hashes: dict[str, str],
) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    saved = summary.get("configuration", {})
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "problems_sha256": problems_sha256,
        "problem_ids": list(TUNING_PROBLEM_IDS),
        "sample_steps": int(config["sample_steps"]),
        "minimum_flow_prefix_steps": int(config["minimum_prefix_steps"]),
        "flow_prefix_truncation": bool(config["truncate"]),
        "device": expected_device,
        "goal_radius": 0.50,
        "seed": 20260721,
        "num_sorted_edge_trials": 16,
        "num_random_edges": 1,
        "flow_prefetch_batch_size": 16,
        "epsilon_random": 0.05,
        "goal_sampling_probability": 0.30,
        "max_random_edge_time": 0.30,
        "acceleration_scale": 0.50,
        "state_bank_size": 100_000,
        "source_file_sha256": expected_source_hashes,
        "dataset_sha256": expected_dataset_sha256,
        "urdf_sha256": expected_urdf_sha256,
        "collision_asset_sha256": expected_collision_hashes,
    }
    mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if not np.isclose(float(saved.get("planning_time", np.nan)), args.planning_time):
        mismatches["planning_time"] = {
            "expected": args.planning_time,
            "actual": saved.get("planning_time"),
        }
    if mismatches:
        raise ValueError(f"Cannot resume incompatible tuning run {run_dir}: {mismatches}")


def score_result(path: Path, config: dict[str, object]) -> dict[str, object]:
    rows = read_trials(path / "trials.csv")
    if [int(row["problem_id"]) for row in rows] != list(TUNING_PROBLEM_IDS):
        raise ValueError(f"Unexpected tuning problem order in {path}")
    success = np.asarray([row["success"].lower() == "true" for row in rows])
    initial = np.asarray([float(row["initial_normalized_distance"]) for row in rows])
    final = np.asarray([float(row["final_normalized_distance"]) for row in rows])
    goal_radius = float(rows[0]["goal_radius"])
    gap = np.maximum(final[~success] - goal_radius, 0.0)
    progress = np.clip(
        (initial - final) / np.maximum(initial - goal_radius, 1e-12), 0.0, 1.0
    )
    flow_time = sum(float(row["flow_model_seconds"]) for row in rows)
    return {
        **config,
        "result_dir": path.name,
        "num_successes": int(success.sum()),
        "median_failure_goal_gap": 0.0 if not gap.size else float(np.median(gap)),
        "median_normalized_progress": float(np.median(progress)),
        "total_flow_model_seconds": float(flow_time),
    }


def selection_key(result: dict[str, object]) -> tuple:
    # Rounding implements the documented "effectively tied" preference while
    # avoiding decisions on insignificant timer or floating-point jitter.
    return (
        int(result["num_successes"]),
        -round(float(result["median_failure_goal_gap"]), 6),
        round(float(result["median_normalized_progress"]), 6),
        -round(float(result["total_flow_model_seconds"]), 3),
        -int(result["sample_steps"]),
        int(bool(result["truncate"])),
        -int(result["minimum_prefix_steps"]),
    )


def plot_results(path: Path, results: list[dict[str, object]], selected: str) -> None:
    labels = [str(result["name"]) for result in results]
    colors = ["#f28e2b" if label == selected else "#4c78a8" for label in labels]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(labels, [result["num_successes"] for result in results], color=colors)
    axes[0, 0].set(title="Tuning successes", ylabel="Count")
    axes[0, 1].bar(
        labels,
        [result["median_failure_goal_gap"] for result in results],
        color=colors,
    )
    axes[0, 1].set(title="Median failure gap beyond goal radius", ylabel="Normalized distance")
    axes[1, 0].bar(
        labels,
        [result["median_normalized_progress"] for result in results],
        color=colors,
    )
    axes[1, 0].set(title="Median normalized goal progress", ylabel="Fraction")
    axes[1, 1].bar(
        labels,
        [result["total_flow_model_seconds"] for result in results],
        color=colors,
    )
    axes[1, 1].set(title="Total flow-model time", ylabel="Seconds")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty tuning directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = file_sha256(args.checkpoint.resolve())
    problems_sha256 = file_sha256(DEFAULT_PROBLEMS)
    if args.device == "auto":
        expected_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        expected_device = args.device
    from benchmark_franka_flow_eb_rrt import (
        DEFAULT_DATASET,
        DEFAULT_URDF,
        collision_asset_hashes,
        source_file_hashes,
    )

    expected_source_hashes = source_file_hashes()
    expected_dataset_sha256 = file_sha256(DEFAULT_DATASET)
    expected_urdf_sha256 = file_sha256(DEFAULT_URDF)
    expected_collision_hashes = collision_asset_hashes(DEFAULT_URDF)
    grid = []
    for sample_steps in (8, 16):
        grid.extend(
            [
                {
                    "name": f"s{sample_steps:02d}_trunc_p05",
                    "sample_steps": sample_steps,
                    "truncate": True,
                    "minimum_prefix_steps": 5,
                },
                {
                    "name": f"s{sample_steps:02d}_trunc_p15",
                    "sample_steps": sample_steps,
                    "truncate": True,
                    "minimum_prefix_steps": 15,
                },
                {
                    "name": f"s{sample_steps:02d}_full",
                    "sample_steps": sample_steps,
                    "truncate": False,
                    "minimum_prefix_steps": 5,
                },
            ]
        )

    results = []
    benchmark_script = (
        FLOWMRMP_ROOT
        / "scripts"
        / "FrankaTesting"
        / "benchmark_franka_flow_eb_rrt.py"
    )
    for config in grid:
        run_dir = output_dir / str(config["name"])
        summary_path = run_dir / "summary.json"
        if args.resume and summary_path.is_file():
            validate_completed_run(
                run_dir,
                config,
                args,
                checkpoint_sha256=checkpoint_sha256,
                problems_sha256=problems_sha256,
                expected_device=expected_device,
                expected_source_hashes=expected_source_hashes,
                expected_dataset_sha256=expected_dataset_sha256,
                expected_urdf_sha256=expected_urdf_sha256,
                expected_collision_hashes=expected_collision_hashes,
            )
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    f"Incomplete/non-resumable tuning run exists: {run_dir}. "
                    "Move it aside and rerun this configuration."
                )
            command = [
                sys.executable,
                str(benchmark_script),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--device",
                args.device,
                "--sample-steps",
                str(config["sample_steps"]),
                "--problem-ids",
                ",".join(str(value) for value in TUNING_PROBLEM_IDS),
                "--planning-time",
                str(args.planning_time),
                "--minimum-flow-prefix-steps",
                str(config["minimum_prefix_steps"]),
                "--output-dir",
                str(run_dir),
            ]
            if not config["truncate"]:
                command.append("--no-flow-prefix-truncation")
            log_path = output_dir / f"{config['name']}.log"
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=FLOWMRMP_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        result = score_result(run_dir, config)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    selected = max(results, key=selection_key)
    fixed_configurations = []
    for result in results:
        saved_summary = json.loads(
            (
                output_dir
                / str(result["result_dir"])
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        saved_configuration = saved_summary["configuration"]
        fixed_configurations.append(
            {key: saved_configuration.get(key) for key in FIXED_BENCHMARK_KEYS}
        )
    fixed_benchmark_configuration = fixed_configurations[0]
    if any(
        configuration != fixed_benchmark_configuration
        for configuration in fixed_configurations[1:]
    ):
        raise ValueError("Tuning grid runs differ in a supposedly fixed configuration")
    checkpoint_path = args.checkpoint.resolve()
    try:
        checkpoint_display_path = str(checkpoint_path.relative_to(FLOWMRMP_ROOT))
    except ValueError:
        checkpoint_display_path = str(checkpoint_path)
    report = {
        "checkpoint": checkpoint_display_path,
        "checkpoint_sha256": checkpoint_sha256,
        "problems": path_label(DEFAULT_PROBLEMS),
        "problems_sha256": problems_sha256,
        "resolved_device": expected_device,
        "source_file_sha256": expected_source_hashes,
        "tuning_script_sha256": file_sha256(Path(__file__).resolve()),
        "fixed_benchmark_configuration": fixed_benchmark_configuration,
        "problem_ids": list(TUNING_PROBLEM_IDS),
        "problem_selection": (
            "Fixed before any FlowEBRRT result was observed, using only the frozen "
            "Vanilla-RRT table: both Vanilla successes plus six failures spanning "
            "low, middle, and high initial distance, final gap, and normalized "
            "progress. The remaining 92 problems are excluded from configuration "
            "selection."
        ),
        "planning_time_seconds_per_problem": args.planning_time,
        "selection_order": [
            "highest success count",
            "lowest median failure gap beyond radius",
            "highest median normalized progress",
            "lowest total flow-model time",
            "effective ties prefer 8 flow steps, truncation, and prefix 5",
        ],
        "selected": selected,
        "all_results": results,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_results(output_dir / "tuning_comparison.png", results, str(selected["name"]))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
