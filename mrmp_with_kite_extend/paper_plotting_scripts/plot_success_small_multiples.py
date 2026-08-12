#!/usr/bin/env python3
"""Small-multiple success-rate plots by system and environment."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, row_radius, system_results_root

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
    marker: str
    linestyle: str


PARADIGMS = {
    "centralized": {
        "title": "cRRT",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT", "#6F7B8A", "o", "--"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE", "#0072B2", "s", "-"),
        ),
    },
    "prrt": {
        "title": "pRRT",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT", "#6F7B8A", "o", "--"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE", "#0072B2", "s", "-"),
        ),
    },
    "kcbs": {
        "title": "KCBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "KCBS", "#6F7B8A", "o", "--"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "KCBS+KiTE", "#0072B2", "s", "-"),
        ),
    },
}

SYSTEMS = ("UCYCLE", "SOC", "QUAD")
SYSTEM_LABELS = {
    "UCYCLE": "UC",
    "SOC": "SOC",
    "QUAD": "DI",
}
ENVIRONMENT_GRID = (
    "small_cluttered_env",
    "large_cluttered_env",
    "swap_env",
    "narrow_corridor_env",
    "large_cluttered_3d_env",
    "swap_3d_env",
)
SYSTEM_ENVIRONMENTS = {
    "SOC": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "UCYCLE": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "QUAD": ("large_cluttered_3d_env", "swap_3d_env"),
}
ENVIRONMENT_LABELS = {
    "small_cluttered_env": "Small\nCluttered",
    "large_cluttered_env": "Large\nCluttered",
    "swap_env": "Swap",
    "narrow_corridor_env": "Narrow\nCorridor",
    "large_cluttered_3d_env": "Large\nCluttered 3D",
    "swap_3d_env": "Swap 3D",
}


def read_success_rate(csv_path: Path) -> tuple[int, int, float] | None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    successes = sum(row["Success"].strip().lower() == "true" for row in rows)
    return successes, len(rows), successes / len(rows)


def collect(results_root: Path, paradigm: str, ucycle_results_root: Path | None = None):
    planners = PARADIGMS[paradigm]["planners"]
    data = {
        system: {
            environment: {planner.label: [] for planner in planners}
            for environment in SYSTEM_ENVIRONMENTS[system]
        }
        for system in SYSTEMS
    }
    for system in SYSTEMS:
        root = system_results_root(results_root, system, ucycle_results_root)
        for environment in SYSTEM_ENVIRONMENTS[system]:
            env_dir = environment_results_dir(root, system, environment)
            if not env_dir.exists():
                continue
            for run_dir in sorted(env_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                match = RUN_DIR_RE.match(run_dir.name)
                if not match or match.group("system") != system:
                    continue
                agents = int(match.group("agents"))
                for planner in planners:
                    csv_path = run_dir / "csvs" / planner.csv_name
                    if not csv_path.exists():
                        continue
                    result = read_success_rate(csv_path)
                    if result is None:
                        continue
                    successes, total, rate = result
                    data[system][environment][planner.label].append({
                        "agents": agents,
                        "successes": successes,
                        "total": total,
                        "success_rate": 100.0 * rate,
                        "csv_path": str(csv_path),
                    })
    for system_data in data.values():
        for env_data in system_data.values():
            for points in env_data.values():
                points.sort(key=lambda row: int(row["agents"]))
    return data


def plot(data, paradigm: str, output_path: Path) -> None:
    planners = PARADIGMS[paradigm]["planners"]
    plt.rcParams.update({
        "figure.figsize": (7.05, 3.55),
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 7.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.25,
        "lines.markersize": 3.0,
        "axes.linewidth": 0.65,
    })
    fig, axes = plt.subplots(len(SYSTEMS), len(ENVIRONMENT_GRID), sharey=True)

    for row, system in enumerate(SYSTEMS):
        system_envs = set(SYSTEM_ENVIRONMENTS[system])
        for col, environment in enumerate(ENVIRONMENT_GRID):
            ax = axes[row][col]
            if environment not in system_envs:
                ax.axis("off")
                continue
            all_agents = set()
            for planner in planners:
                points = data[system][environment][planner.label]
                if not points:
                    continue
                agents = [int(point["agents"]) for point in points]
                rates = [float(point["success_rate"]) for point in points]
                all_agents.update(agents)
                ax.plot(
                    agents,
                    rates,
                    color=planner.color,
                    marker=planner.marker,
                    linestyle=planner.linestyle,
                    markerfacecolor="white",
                    markeredgewidth=0.85,
                    zorder=4 if "KiTE" in planner.label else 3,
                )
            if row == 0:
                ax.set_title(ENVIRONMENT_LABELS[environment], pad=3)
            if col == 0:
                ax.set_ylabel(f"{SYSTEM_LABELS[system]}\nSuccess (%)", labelpad=5, fontsize=8.3, fontweight="bold")
            if row == len(SYSTEMS) - 1 or system == "QUAD":
                ax.set_xlabel("Robots", labelpad=1.0, fontsize=8.3, fontweight="bold")
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 50, 100])
            if all_agents:
                if system == "QUAD":
                    xticks = [tick for tick in (2, 5, 10, 15, 20, 25, 30) if min(all_agents) <= tick <= max(all_agents)]
                else:
                    xticks = [tick for tick in (3, 5, 10, 15, 20, 25, 30) if min(all_agents) <= tick <= max(all_agents)]
                ax.set_xlim(min(all_agents) - 0.6, max(all_agents) + 0.6)
                ax.set_xticks(xticks)
            ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.45, zorder=0)
            ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.28, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", pad=0.8)
            ax.tick_params(axis="y", pad=0.8)

    handles = [
        Line2D([0], [0], color=p.color, marker=p.marker, linestyle=p.linestyle,
               markerfacecolor="white", markeredgewidth=0.85, lw=1.35, label=p.label)
        for p in planners
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False, columnspacing=0.9, handlelength=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.18, w_pad=0.35, h_pad=0.40)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(data, paradigm: str, output_path: Path) -> None:
    planners = PARADIGMS[paradigm]["planners"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = ["paradigm", "system", "radius", "environment", "planner", "agents", "successes", "total", "success_rate", "csv_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for environment in SYSTEM_ENVIRONMENTS[system]:
                for planner in planners:
                    for point in data[system][environment][planner.label]:
                        writer.writerow({
                            "paradigm": paradigm,
                            "system": system,
                            "radius": row_radius(system, str(point["csv_path"])),
                            "environment": environment,
                            "planner": planner.label,
                            "agents": point["agents"],
                            "successes": point["successes"],
                            "total": point["total"],
                            "success_rate": f"{point['success_rate']:.6f}",
                            "csv_path": point["csv_path"],
                        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--ucycle-results-root", type=Path, default=None)
    parser.add_argument("--paradigm", choices=sorted(PARADIGMS))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_plots/figures/success_rate"))
    args = parser.parse_args()

    paradigms = [args.paradigm] if args.paradigm else list(PARADIGMS)
    for paradigm in paradigms:
        data = collect(args.results_root, paradigm, args.ucycle_results_root)
        output = args.output_dir / f"success_small_multiples_{paradigm}.png"
        plot(data, paradigm, output)
        write_csv(data, paradigm, csv_output_path(output))
        print(f"Wrote {output}")
        print(f"Wrote {csv_output_path(output)}")


if __name__ == "__main__":
    main()
