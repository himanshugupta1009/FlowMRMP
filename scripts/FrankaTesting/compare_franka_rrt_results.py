#!/usr/bin/env python3
"""Create a paired Vanilla-RRT versus FlowEBRRT Franka comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from franka_paths import existing_result_dir

DEFAULT_VANILLA = existing_result_dir(
    "franka_vanilla_rrt", "franka_reachable_100_goal_r025_max50"
)
TUNING_PROBLEM_IDS = (4, 10, 17, 37, 41, 57, 64, 81)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_trials(path: Path) -> dict[int, dict[str, str]]:
    with (path / "trials.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["problem_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate problem IDs in {path / 'trials.csv'}")
    return result


def validate_fresh_audit(path: Path) -> dict[str, object]:
    audit_path = path / "path_validity_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing path audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_paths_valid"):
        raise ValueError(f"Path audit did not pass: {audit_path}")
    manifest = audit.get("audited_artifact_sha256")
    if not isinstance(manifest, dict):
        raise ValueError(f"Path audit has no immutable artifact manifest: {audit_path}")
    for name, expected_hash in manifest.items():
        if name == "paths":
            continue
        file_path = path / name
        if not file_path.is_file() or expected_hash != file_sha256(file_path):
            raise ValueError(f"Result changed after path audit: {file_path}")
    for required_name in ("summary.json", "trials.csv", "problems.npz"):
        if required_name not in manifest:
            raise ValueError(f"Audit manifest omits {required_name}: {audit_path}")
    saved_paths = manifest.get("paths", {})
    actual_path_files = sorted((path / "paths").glob("problem_*.npz"))
    actual_paths = {
        str(file_path.relative_to(path)): file_sha256(file_path)
        for file_path in actual_path_files
    }
    if saved_paths != actual_paths:
        raise ValueError(f"Saved paths changed after path audit: {path}")
    return audit


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def planner_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    success = np.asarray([as_bool(row["success"]) for row in rows])
    planning = np.asarray([float(row["planning_time_seconds"]) for row in rows])
    successful = planning[success]
    return {
        "num_problems": len(rows),
        "num_successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "planning_time_all_mean": float(planning.mean()),
        "planning_time_all_median": float(np.median(planning)),
        "planning_time_success_mean": (
            None if not successful.size else float(successful.mean())
        ),
    }


def paired_metrics(
    vanilla_rows: list[dict[str, str]], flow_rows: list[dict[str, str]]
) -> dict[str, object]:
    """Summarize two planners on an already-verified paired row subset."""
    vanilla_success = np.asarray([as_bool(row["success"]) for row in vanilla_rows])
    flow_success = np.asarray([as_bool(row["success"]) for row in flow_rows])
    vanilla_final = np.asarray(
        [float(row["final_normalized_distance"]) for row in vanilla_rows]
    )
    flow_final = np.asarray(
        [float(row["final_normalized_distance"]) for row in flow_rows]
    )
    return {
        "vanilla": planner_metrics(vanilla_rows),
        "flow_eb_rrt": planner_metrics(flow_rows),
        "paired_outcomes": {
            "both_succeeded": int(np.sum(vanilla_success & flow_success)),
            "flow_only_succeeded": int(np.sum(~vanilla_success & flow_success)),
            "vanilla_only_succeeded": int(np.sum(vanilla_success & ~flow_success)),
            "neither_succeeded": int(np.sum(~vanilla_success & ~flow_success)),
        },
        "final_distance": {
            "median_vanilla": float(np.median(vanilla_final)),
            "median_flow": float(np.median(flow_final)),
            "median_flow_minus_vanilla": float(
                np.median(flow_final - vanilla_final)
            ),
            "fraction_flow_closer": float(np.mean(flow_final < vanilla_final)),
        },
    }


def main() -> None:
    args = parse_args()
    vanilla_audit = validate_fresh_audit(args.vanilla_dir)
    flow_audit = validate_fresh_audit(args.flow_dir)
    vanilla = load_trials(args.vanilla_dir)
    flow = load_trials(args.flow_dir)
    if set(vanilla) != set(flow):
        raise ValueError(
            "Paired comparison requires identical complete problem-ID sets; "
            f"missing from Flow={sorted(set(vanilla) - set(flow))}, "
            f"extra in Flow={sorted(set(flow) - set(vanilla))}"
        )
    problem_ids = sorted(vanilla)
    if not problem_ids:
        raise ValueError("The result folders contain no trials")
    vanilla_rows = [vanilla[index] for index in problem_ids]
    flow_rows = [flow[index] for index in problem_ids]
    provenance_fields = (
        "start_trajectory",
        "start_timestep",
        "goal_trajectory",
        "goal_timestep",
        "seed",
    )
    for problem_id, vanilla_row, flow_row in zip(
        problem_ids, vanilla_rows, flow_rows
    ):
        for field in provenance_fields:
            if vanilla_row[field] != flow_row[field]:
                raise ValueError(
                    f"Problem {problem_id} differs in A-B provenance field {field}"
                )
        if not np.isclose(
            float(vanilla_row["goal_radius"]), float(flow_row["goal_radius"])
        ):
            raise ValueError(f"Problem {problem_id} uses a different goal radius")
        if not np.isclose(
            float(vanilla_row["initial_normalized_distance"]),
            float(flow_row["initial_normalized_distance"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Problem {problem_id} uses a different A-B distance")

    vanilla_summary = json.loads(
        (args.vanilla_dir / "summary.json").read_text(encoding="utf-8")
    )
    flow_summary = json.loads(
        (args.flow_dir / "summary.json").read_text(encoding="utf-8")
    )
    vanilla_budget = float(vanilla_summary["configuration"]["planning_time"])
    flow_budget = float(flow_summary["configuration"]["planning_time"])
    if not np.isclose(vanilla_budget, flow_budget):
        raise ValueError(
            f"Planning-time budgets differ: Vanilla={vanilla_budget}, Flow={flow_budget}"
        )
    for planner_name, rows, budget in (
        ("Vanilla", vanilla_rows, vanilla_budget),
        ("Flow", flow_rows, flow_budget),
    ):
        invalid_times = [
            int(row["problem_id"])
            for row in rows
            if not np.isfinite(float(row["planning_time_seconds"]))
            or float(row["planning_time_seconds"]) < 0.0
            or float(row["planning_time_seconds"]) > budget + 0.25
        ]
        if invalid_times:
            raise ValueError(
                f"{planner_name} trial times violate the configured budget: "
                f"{invalid_times}"
            )
    vanilla_goal_radius = float(vanilla_summary["configuration"]["goal_radius"])
    flow_goal_radius = float(flow_summary["configuration"]["goal_radius"])
    if not np.isclose(vanilla_goal_radius, flow_goal_radius):
        raise ValueError("Configured goal radii differ")
    vanilla_goal_metric = vanilla_summary["configuration"].get("goal_metric")
    flow_goal_metric = flow_summary["configuration"].get("goal_metric")
    if (
        vanilla_goal_metric != "normalized_7d_joint_position_l2"
        or flow_goal_metric != vanilla_goal_metric
    ):
        raise ValueError(
            "Both planners must use the normalized 7D joint-position goal metric"
        )
    if any(
        not np.isclose(float(row["goal_radius"]), vanilla_goal_radius)
        for row in vanilla_rows
    ) or any(
        not np.isclose(float(row["goal_radius"]), flow_goal_radius)
        for row in flow_rows
    ):
        raise ValueError("A trial row goal radius differs from its configuration")
    vanilla_workers = int(vanilla_summary["configuration"].get("workers", 1))
    if (
        int(vanilla_summary["num_problems"]) != len(problem_ids)
        or int(flow_summary["num_problems"]) != len(problem_ids)
    ):
        raise ValueError("A summary count does not match the paired trial set")
    vanilla_problems_path = args.vanilla_dir / "problems.npz"
    flow_problems_path = args.flow_dir / "problems.npz"
    if not vanilla_problems_path.is_file() or not flow_problems_path.is_file():
        raise FileNotFoundError("Both result folders must contain problems.npz")
    vanilla_problems_sha256 = file_sha256(vanilla_problems_path)
    flow_problems_sha256 = file_sha256(flow_problems_path)
    if vanilla_problems_sha256 != flow_problems_sha256:
        raise ValueError("Vanilla and Flow result folders contain different A-B files")
    configured_problem_sha256 = flow_summary["configuration"].get("problems_sha256")
    if (
        configured_problem_sha256 is not None
        and configured_problem_sha256 != flow_problems_sha256
    ):
        raise ValueError("Flow result configuration does not match its A-B file hash")
    with np.load(vanilla_problems_path) as problem_data:
        witness_backed = (
            "witness_states" in problem_data.files
            and "witness_actions" in problem_data.files
            and "difficulty_tier" in problem_data.files
        )
        tier_counts = (
            {
                str(tier): int(np.sum(problem_data["difficulty_tier"].astype(str) == tier))
                for tier in sorted(set(problem_data["difficulty_tier"].astype(str).tolist()))
            }
            if witness_backed
            else None
        )
    vanilla_success = np.asarray([as_bool(row["success"]) for row in vanilla_rows])
    flow_success = np.asarray([as_bool(row["success"]) for row in flow_rows])
    vanilla_final = np.asarray(
        [float(row["final_normalized_distance"]) for row in vanilla_rows]
    )
    flow_final = np.asarray(
        [float(row["final_normalized_distance"]) for row in flow_rows]
    )
    paired_all = paired_metrics(vanilla_rows, flow_rows)
    nontrivial_indices = [
        index
        for index, row in enumerate(vanilla_rows)
        if float(row["initial_normalized_distance"]) > vanilla_goal_radius
    ]
    initially_satisfied_ids = [
        problem_id
        for problem_id, row in zip(problem_ids, vanilla_rows)
        if float(row["initial_normalized_distance"]) <= vanilla_goal_radius
    ]
    summary = {
        "shared_problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "problems_sha256": flow_problems_sha256,
        "path_audits": {
            "vanilla_sha256": file_sha256(
                args.vanilla_dir / "path_validity_audit.json"
            ),
            "flow_sha256": file_sha256(args.flow_dir / "path_validity_audit.json"),
            "vanilla_total_states_checked": vanilla_audit["total_states_checked"],
            "flow_total_states_checked": flow_audit["total_states_checked"],
        },
        "planning_time_budget_seconds": flow_budget,
        "goal_radius": float(flow_rows[0]["goal_radius"]),
        "goal_metric": flow_goal_metric,
        "witness_backed_problem_set": witness_backed,
        "difficulty_tier_counts": tier_counts,
        **paired_all,
        "nontrivial_start_subset": {
            "excluded_initially_satisfied_problem_ids": initially_satisfied_ids,
            **paired_metrics(
                [vanilla_rows[index] for index in nontrivial_indices],
                [flow_rows[index] for index in nontrivial_indices],
            ),
        },
        "fairness_notes": [
            "Per-problem planner wall time and success rate use identical A-B pairs, goal radius, and time budget.",
            "The nontrivial subset excludes problems whose starts already satisfy the normalized 7D position-only goal test.",
            (
                "Both planner benchmarks ran sequentially with one problem at a time."
                if vanilla_workers == 1
                else (
                    f"Vanilla used {vanilla_workers} parallel workers whereas "
                    "FlowEBRRT ran sequentially; aggregate benchmark wall times "
                    "are not directly compared."
                )
            ),
            "Flow edges use the dataset acceleration limit; acceleration_scale applies only to random fallback controls.",
            (
                "Every A-B pair has an included same-trajectory acceleration witness "
                "that is valid under the benchmark dynamics and safety checks."
                if witness_backed
                else "The A-B endpoints come from dataset200k support."
            ),
            "The source trajectories also supplied the flow-training corpus, so this remains an in-distribution planning benchmark rather than unseen-trajectory generalization.",
            "Both planners use the same PyBullet proxy URDF with fixed gripper origins at y=+/-0.065 m; this differs from the CuRobo source robot's +/-0.04 m gripper geometry but keeps the paired comparison controlled.",
        ],
    }
    tuning_ids = set(TUNING_PROBLEM_IDS)
    tuning_info = flow_summary["configuration"].get("tuning_selection")
    if tuning_info is not None:
        tuning_path = args.flow_dir / str(tuning_info.get("file"))
        if (
            not tuning_path.is_file()
            or file_sha256(tuning_path) != tuning_info.get("sha256")
        ):
            raise ValueError("Final Flow run's tuning selection is missing or changed")
        tuning_report = json.loads(tuning_path.read_text(encoding="utf-8"))
        if tuning_report.get("problem_ids") != list(TUNING_PROBLEM_IDS):
            raise ValueError("Tuning selection uses an unexpected problem subset")
        if tuning_report.get("selected") != tuning_info.get("selected"):
            raise ValueError("Embedded tuning selection differs from run configuration")
    if (
        tuning_info is not None
        and tuning_ids.issubset(problem_ids)
        and len(problem_ids) > len(tuning_ids)
    ):
        held_out_indices = [
            index for index, problem_id in enumerate(problem_ids)
            if problem_id not in tuning_ids
        ]
        summary["planner_tuning_protocol"] = {
            "tuning_problem_ids": list(TUNING_PROBLEM_IDS),
            "tuning_problem_count": len(TUNING_PROBLEM_IDS),
            "planner_tuning_held_out_problem_count": len(held_out_indices),
            "note": (
                "Held out means excluded from FlowEBRRT configuration selection; "
                "it does not mean held out from the flow model's source corpus."
            ),
        }
        summary["planner_tuning_held_out_subset"] = paired_metrics(
            [vanilla_rows[index] for index in held_out_indices],
            [flow_rows[index] for index in held_out_indices],
        )

    output_dir = (args.output_dir or args.flow_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison_to_vanilla.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(
        ["Vanilla RRT", "FlowEBRRT"],
        [summary["vanilla"]["success_rate"], summary["flow_eb_rrt"]["success_rate"]],
        color=["#999999", "#4c78a8"],
    )
    axes[0, 0].set(title="Success rate on identical A-B problems", ylim=(0.0, 1.0))
    axes[0, 1].bar(
        ["Vanilla RRT", "FlowEBRRT"],
        [
            summary["vanilla"]["planning_time_all_mean"],
            summary["flow_eb_rrt"]["planning_time_all_mean"],
        ],
        color=["#999999", "#4c78a8"],
    )
    axes[0, 1].set(title="Mean planning time", ylabel="Seconds")
    color = np.where(flow_success, "#2ca02c", "#d62728")
    axes[1, 0].scatter(vanilla_final, flow_final, c=color, alpha=0.8)
    limit = max(float(vanilla_final.max()), float(flow_final.max()))
    axes[1, 0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1, 0].set(
        title="Paired final goal distance",
        xlabel="Vanilla RRT",
        ylabel="FlowEBRRT",
    )
    axes[1, 1].bar(
        np.arange(len(problem_ids)),
        flow_final - vanilla_final,
        color=np.where(flow_final < vanilla_final, "#2ca02c", "#d62728"),
        width=0.9,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(
        title="Flow minus Vanilla final distance",
        xlabel="Shared problem",
        ylabel="Distance difference",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_to_vanilla.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"comparison saved to {output_dir}")


if __name__ == "__main__":
    main()
