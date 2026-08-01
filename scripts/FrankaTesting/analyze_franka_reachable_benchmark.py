#!/usr/bin/env python3
"""Create tiered and paired analysis for the reachable Franka A-B benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TIERS = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_trials(path: Path) -> dict[int, dict[str, object]]:
    trials: dict[int, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            pid = int(raw["problem_id"])
            trials[pid] = {
                "success": raw["success"].lower() == "true",
                "time": float(raw["planning_time_seconds"]),
                "tier": raw["difficulty_tier"],
                "distance": float(raw["initial_normalized_distance"]),
                "witness_duration": float(raw["witness_duration_seconds"]),
                "final_distance": float(raw["final_normalized_distance"]),
            }
    return trials


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - spread), min(1.0, center + spread)]


def planner_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if bool(row["success"])]
    all_times = [float(row["time"]) for row in rows]
    success_times = [float(row["time"]) for row in successful]
    return {
        "problems": len(rows),
        "successes": len(successful),
        "success_rate": len(successful) / len(rows),
        "success_rate_wilson_95": wilson(len(successful), len(rows)),
        "all_time_seconds": {
            "mean": float(np.mean(all_times)),
            "median": float(np.median(all_times)),
            "p95": percentile(all_times, 95),
        },
        "successful_time_seconds": {
            "mean": float(np.mean(success_times)) if success_times else None,
            "median": float(np.median(success_times)) if success_times else None,
            "p95": percentile(success_times, 95),
        },
    }


def exact_two_sided_sign_test(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vanilla = load_trials(args.vanilla_dir / "trials.csv")
    flow = load_trials(args.flow_dir / "trials.csv")
    if set(vanilla) != set(flow):
        raise ValueError("Vanilla and FlowEBRRT problem IDs do not match")

    ids = sorted(vanilla)
    for pid in ids:
        if vanilla[pid]["tier"] != flow[pid]["tier"]:
            raise ValueError(f"Difficulty tier mismatch for problem {pid}")

    analysis: dict[str, object] = {
        "problem_count": len(ids),
        "overall": {
            "vanilla_rrt": planner_metrics([vanilla[pid] for pid in ids]),
            "flow_eb_rrt": planner_metrics([flow[pid] for pid in ids]),
        },
        "by_tier": {},
    }
    for tier in TIERS:
        tier_ids = [pid for pid in ids if vanilla[pid]["tier"] == tier]
        analysis["by_tier"][tier] = {
            "problem_count": len(tier_ids),
            "initial_distance": {
                "median": float(np.median([vanilla[pid]["distance"] for pid in tier_ids])),
                "range": [
                    float(min(vanilla[pid]["distance"] for pid in tier_ids)),
                    float(max(vanilla[pid]["distance"] for pid in tier_ids)),
                ],
            },
            "witness_duration_seconds": {
                "median": float(np.median([vanilla[pid]["witness_duration"] for pid in tier_ids])),
                "range": [
                    float(min(vanilla[pid]["witness_duration"] for pid in tier_ids)),
                    float(max(vanilla[pid]["witness_duration"] for pid in tier_ids)),
                ],
            },
            "vanilla_rrt": planner_metrics([vanilla[pid] for pid in tier_ids]),
            "flow_eb_rrt": planner_metrics([flow[pid] for pid in tier_ids]),
        }

    both = [pid for pid in ids if vanilla[pid]["success"] and flow[pid]["success"]]
    vanilla_only = [pid for pid in ids if vanilla[pid]["success"] and not flow[pid]["success"]]
    flow_only = [pid for pid in ids if flow[pid]["success"] and not vanilla[pid]["success"]]
    neither = [pid for pid in ids if not vanilla[pid]["success"] and not flow[pid]["success"]]
    paired_deltas = [float(flow[pid]["time"]) - float(vanilla[pid]["time"]) for pid in both]
    analysis["paired"] = {
        "both_succeeded": len(both),
        "vanilla_only_succeeded": len(vanilla_only),
        "flow_only_succeeded": len(flow_only),
        "neither_succeeded": len(neither),
        "vanilla_only_problem_ids": vanilla_only,
        "flow_only_problem_ids": flow_only,
        "neither_problem_ids": neither,
        "exact_mcnemar_p_value": exact_two_sided_sign_test(len(vanilla_only), len(flow_only)),
        "both_success_time": {
            "problem_count": len(both),
            "flow_faster_count": sum(delta < 0 for delta in paired_deltas),
            "flow_faster_fraction": sum(delta < 0 for delta in paired_deltas) / len(both),
            "median_flow_minus_vanilla_seconds": float(np.median(paired_deltas)),
            "median_vanilla_seconds": float(np.median([vanilla[pid]["time"] for pid in both])),
            "median_flow_seconds": float(np.median([flow[pid]["time"] for pid in both])),
        },
    }

    output_json = args.output_dir / "reachable_benchmark_analysis.json"
    output_json.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    x = np.arange(len(TIERS))
    width = 0.34
    vanilla_rates = [analysis["by_tier"][tier]["vanilla_rrt"]["success_rate"] * 100 for tier in TIERS]
    flow_rates = [analysis["by_tier"][tier]["flow_eb_rrt"]["success_rate"] * 100 for tier in TIERS]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    bars_v = ax.bar(x - width / 2, vanilla_rates, width, label="Vanilla RRT", color="#243B53")
    bars_f = ax.bar(x + width / 2, flow_rates, width, label="FlowEBRRT", color="#2CB1BC")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, [tier.title() for tier in TIERS])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for bars in (bars_v, bars_f):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Reachable Franka A–B benchmark by difficulty tier")
    fig.savefig(args.output_dir / "reachable_success_by_tier.png", dpi=180)
    plt.close(fig)
    print(json.dumps(analysis, indent=2))
    print(f"analysis saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
