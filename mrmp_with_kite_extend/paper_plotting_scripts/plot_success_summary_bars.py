#!/usr/bin/env python3
"""Compact success-rate summary bar charts for paper figures."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, summary_radius, system_results_root

RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")


@dataclass(frozen=True)
class PlannerSpec:
    csv_name: str
    label: str
    color: str
    hatch: str = ""


PARADIGMS = {
    "centralized": {
        "title": "cRRT",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT", "#8C97A6"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE", "#0072B2", "//"),
        ),
    },
    "prrt": {
        "title": "pRRT",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT", "#8C97A6"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE", "#0072B2", "//"),
        ),
    },
    "kcbs": {
        "title": "KCBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "KCBS", "#8C97A6"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "KCBS+KiTE", "#0072B2", "//"),
            PlannerSpec("dbRRT_KCBS_results.csv", "KCBS+idb-RRT", "#CC79A7", ".."),
        ),
    },
}

SYSTEMS = ("UCYCLE", "SOC", "QUAD")
SYSTEM_LABELS = {
    "UCYCLE": "UC",
    "SOC": "SOC",
    "QUAD": "DI",
}
SYSTEM_ENVIRONMENTS = {
    "SOC": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "UCYCLE": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "QUAD": ("large_cluttered_3d_env", "swap_3d_env"),
}


def read_success_rate(csv_path: Path) -> float | None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    successes = sum(row["Success"].strip().lower() == "true" for row in rows)
    return successes / len(rows)


def collect_summary(root: Path, ucycle_results_root: Path | None = None):
    summary = {
        paradigm: {
            system: {planner.label: {"rates": [], "n": 0} for planner in spec["planners"]}
            for system in SYSTEMS
        }
        for paradigm, spec in PARADIGMS.items()
    }

    for paradigm, spec in PARADIGMS.items():
        for system in SYSTEMS:
            system_root = system_results_root(root, system, ucycle_results_root)
            for environment in SYSTEM_ENVIRONMENTS[system]:
                env_dir = environment_results_dir(system_root, system, environment)
                if not env_dir.exists():
                    continue
                for run_dir in sorted(env_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    match = RUN_DIR_RE.match(run_dir.name)
                    if not match or match.group("system") != system:
                        continue
                    for planner in spec["planners"]:
                        csv_path = run_dir / "csvs" / planner.csv_name
                        if not csv_path.exists():
                            continue
                        rate = read_success_rate(csv_path)
                        if rate is None:
                            continue
                        cell = summary[paradigm][system][planner.label]
                        cell["rates"].append(rate)
                        cell["n"] += 1

    compact = {
        paradigm: {
            system: {
                planner.label: {
                    "success": 100.0 * mean(summary[paradigm][system][planner.label]["rates"])
                    if summary[paradigm][system][planner.label]["rates"]
                    else math.nan,
                    "n": summary[paradigm][system][planner.label]["n"],
                }
                for planner in spec["planners"]
            }
            for system in SYSTEMS
        }
        for paradigm, spec in PARADIGMS.items()
    }
    return compact


def plot(summary, output_path: Path) -> None:
    plt.rcParams.update({
        "figure.figsize": (7.05, 2.05),
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 8.2,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.1,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, sharey=True)

    for ax, (paradigm, spec) in zip(axes, PARADIGMS.items()):
        planners = spec["planners"]
        x = np.arange(len(SYSTEMS))
        width = min(0.28, 0.78 / len(planners))
        offsets = (np.arange(len(planners)) - (len(planners) - 1) / 2) * width

        for planner_index, planner in enumerate(planners):
            values = [summary[paradigm][system][planner.label]["success"] for system in SYSTEMS]
            bars = ax.bar(
                x + offsets[planner_index],
                values,
                width=width * 0.92,
                label=planner.label,
                color=planner.color,
                edgecolor="#28313A",
                linewidth=0.55,
                hatch=planner.hatch,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                if not math.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 2.0, 101.0),
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                )

        ax.set_title(spec["title"], pad=4)
        ax.set_xticks(x, [SYSTEM_LABELS[system] for system in SYSTEMS])
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.58, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=1.5)
        ax.tick_params(axis="y", pad=1.5)

    axes[0].set_ylabel("Mean Success Rate (%)", fontsize=9.8, fontweight="bold")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.35,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.83), pad=0.18, w_pad=0.55)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = ["paradigm", "system", "radius", "planner", "mean_success_rate", "conditions"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paradigm, spec in PARADIGMS.items():
            for system in SYSTEMS:
                for planner in spec["planners"]:
                    cell = summary[paradigm][system][planner.label]
                    writer.writerow({
                        "paradigm": paradigm,
                        "system": system,
                        "radius": summary_radius(system, UCYCLE_RADIUS),
                        "planner": planner.label,
                        "mean_success_rate": f"{cell['success']:.6f}",
                        "conditions": cell["n"],
                    })


def main() -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None
    output = Path("paper_plots/figures/success_rate/success_rate_summary_bars.png")
    summary = collect_summary(root, ucycle_root)
    plot(summary, output)
    write_csv(summary, csv_output_path(output))
    print(f"Wrote {output}")
    print(f"Wrote {csv_output_path(output)}")


if __name__ == "__main__":
    main()
