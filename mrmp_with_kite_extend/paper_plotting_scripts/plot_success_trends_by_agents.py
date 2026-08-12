#!/usr/bin/env python3
"""Success-rate trends vs. agents with environments averaged out."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
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
DBCBS_NORMALIZED_CSV = Path("paper_plots/data/dbcbs_normalized.csv")
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")


@dataclass(frozen=True)
class PlannerSpec:
    key: str
    csv_name: str
    label: str
    color: str
    marker: str
    linestyle: str
    external: bool = False


PARADIGMS = {
    "centralized": {
        "title": "cRRT",
        "planners": (
            PlannerSpec("crrt", "CRRT_results.csv", "cRRT", "#6F7B8A", "o", "--"),
            PlannerSpec("crrt_kite", "K-TI_EB_CRRT_results.csv", "cRRT+KiTE", "#0072B2", "s", "-"),
        ),
    },
    "prrt": {
        "title": "pRRT",
        "planners": (
            PlannerSpec("prrt", "PRRT_results.csv", "pRRT", "#6F7B8A", "o", "--"),
            PlannerSpec("prrt_kite", "KTI_EB_PRRT_results.csv", "pRRT+KiTE", "#0072B2", "s", "-"),
        ),
    },
    "kcbs": {
        "title": "KCBS",
        "planners": (
            PlannerSpec("kcbs", "RRT_KCBS_results.csv", "KCBS", "#6F7B8A", "o", "--"),
            PlannerSpec("kcbs_kite", "KTI_EB_RRT_KCBS_results.csv", "KCBS+KiTE", "#0072B2", "s", "-"),
            PlannerSpec("kcbs_idb", "dbRRT_KCBS_results.csv", "KCBS+idb-RRT", "#CC79A7", "^", "-"),
            PlannerSpec("dbcbs", "", "dbCBS", "#D55E00", "D", "--", external=True),
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


def load_dbcbs_success() -> dict[tuple[str, str, int], float]:
    if not DBCBS_NORMALIZED_CSV.exists():
        return {}
    data = {}
    with DBCBS_NORMALIZED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] == "UCYCLE" and row.get("radius", "0.3") != UCYCLE_RADIUS:
                continue
            if not row["success_rate"]:
                continue
            data[(row["system"], row["environment"], int(row["agents"]))] = float(row["success_rate"])
    return data


def collect_raw_points(root: Path, paradigm: str, ucycle_results_root: Path | None = None):
    dbcbs_success = load_dbcbs_success()
    spec = PARADIGMS[paradigm]
    raw = {
        system: {planner.label: defaultdict(list) for planner in spec["planners"]}
        for system in SYSTEMS
    }

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
                agents = int(match.group("agents"))
                for planner in spec["planners"]:
                    if planner.external:
                        rate = dbcbs_success.get((system, environment, agents))
                    else:
                        csv_path = run_dir / "csvs" / planner.csv_name
                        rate = read_success_rate(csv_path) if csv_path.exists() else None
                    if rate is not None and math.isfinite(rate):
                        raw[system][planner.label][agents].append(100.0 * rate)
    return raw


def summarize(raw):
    summary = {}
    for system, planner_data in raw.items():
        summary[system] = {}
        for planner, by_agent in planner_data.items():
            summary[system][planner] = [
                {
                    "agents": agents,
                    "success_rate": mean(values),
                    "min_success_rate": min(values),
                    "max_success_rate": max(values),
                    "environment_conditions": len(values),
                }
                for agents, values in sorted(by_agent.items())
            ]
    return summary


def system_xticks(system: str) -> list[int]:
    if system == "QUAD":
        return [2, 5, 10, 15, 20, 25, 30]
    return [3, 5, 10, 15, 20, 25, 30]


def plot(summary, paradigm: str, output_path: Path) -> None:
    planners = PARADIGMS[paradigm]["planners"]
    plt.rcParams.update({
        "figure.figsize": (5.35, 1.86),
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 8.0,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 6.9,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, sharey=True)
    for index, (ax, system) in enumerate(zip(axes, SYSTEMS)):
        all_agents = set()
        for planner in planners:
            points = summary[system][planner.label]
            if not points:
                continue
            agents = [int(point["agents"]) for point in points]
            rates = [float(point["success_rate"]) for point in points]
            all_agents.update(agents)
            ax.plot(
                agents,
                rates,
                label=planner.label,
                color=planner.color,
                marker=planner.marker,
                linestyle=planner.linestyle,
                linewidth=1.45 if "KiTE" in planner.label else 1.15,
                markersize=3.7,
                markerfacecolor="white",
                markeredgewidth=0.95,
                zorder=4 if "KiTE" in planner.label else 3,
            )

        ax.set_title(SYSTEM_LABELS[system], pad=3, fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Robots", labelpad=1.0, fontsize=10.0, fontweight="bold")
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        if index == 0:
            ax.set_ylabel("Success Rate (%)", fontsize=10.0, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        if all_agents:
            ticks = [tick for tick in system_xticks(system) if min(all_agents) <= tick <= max(all_agents)]
            ax.set_xlim(min(all_agents) - 0.65, max(all_agents) + 0.65)
            ax.set_xticks(ticks)
        ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.58, zorder=0)
        ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.34, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=1.2)
        ax.tick_params(axis="y", pad=1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    # KCBS has dbCBS only for some systems, so collect handles across all axes.
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(4, len(handles)),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.45,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.80), pad=0.16, w_pad=0.42)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(summary, paradigm: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "paradigm",
            "system",
            "radius",
            "planner",
            "agents",
            "mean_success_rate",
            "min_success_rate",
            "max_success_rate",
            "environment_conditions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for planner in PARADIGMS[paradigm]["planners"]:
                for point in summary[system][planner.label]:
                    writer.writerow({
                        "paradigm": paradigm,
                        "system": system,
                        "radius": summary_radius(system, UCYCLE_RADIUS),
                        "planner": planner.label,
                        "agents": point["agents"],
                        "mean_success_rate": f"{point['success_rate']:.6f}",
                        "min_success_rate": f"{point['min_success_rate']:.6f}",
                        "max_success_rate": f"{point['max_success_rate']:.6f}",
                        "environment_conditions": point["environment_conditions"],
                    })


def main() -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None
    output_root = Path("paper_plots/figures/success_rate")
    for paradigm in PARADIGMS:
        raw = collect_raw_points(root, paradigm, ucycle_root)
        summary = summarize(raw)
        output = output_root / f"success_rate_trend_{paradigm}.png"
        plot(summary, paradigm, output)
        write_csv(summary, paradigm, csv_output_path(output))
        print(f"Wrote {output}")
        print(f"Wrote {csv_output_path(output)}")


if __name__ == "__main__":
    main()
