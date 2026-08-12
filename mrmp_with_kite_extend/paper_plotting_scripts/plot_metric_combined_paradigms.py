#!/usr/bin/env python3
"""Combined CT/PT figures with one column per coordination paradigm."""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, row_radius, system_results_root

RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)
DBCBS_NORMALIZED_CSV = Path("paper_plots/data/dbcbs_normalized.csv")
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")


@dataclass(frozen=True)
class PlannerSpec:
    csv_name: str
    label: str
    role: str
    external: bool = False


@dataclass(frozen=True)
class MetricSpec:
    key: str
    csv_column: str
    external_column: str
    label: str
    log_y: bool
    output_name: str


PARADIGMS = {
    "centralized": {
        "row_label": "cRRT",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT", "Rand"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE (Ours)", "+KiTE (Ours)"),
        ),
    },
    "prrt": {
        "row_label": "pRRT",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT", "Rand"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE (Ours)", "+KiTE (Ours)"),
        ),
    },
    "kcbs": {
        "row_label": "CBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "CBS", "Rand"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "CBS+KiTE (Ours)", "+KiTE (Ours)"),
            PlannerSpec("dbRRT_KCBS_results.csv", "CBS+idb-RRT", "+idb-RRT"),
            PlannerSpec("", "dbCBS", "dbCBS", external=True),
        ),
    },
}
PARADIGM_ORDER = ("centralized", "prrt", "kcbs")
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
ENVIRONMENT_LABELS = {
    "narrow_corridor_env": "Corridor",
    "swap_env": "Swap",
    "small_cluttered_env": "Small Cluttered",
    "large_cluttered_env": "Large Cluttered",
    "swap_3d_env": "Swap 3D",
    "large_cluttered_3d_env": "Large Cluttered 3D",
}
ENVIRONMENT_COLORS = {
    "small_cluttered_env": "#A50F15",
    "large_cluttered_env": "#E66101",
    "swap_env": "#006B3C",
    "narrow_corridor_env": "#8A8A8A",
    "large_cluttered_3d_env": "#E69F00",
    "swap_3d_env": "#332288",
}
ROLE_STYLES = {
    "Rand": {"marker": "o", "linestyle": "--", "alpha": 0.88, "linewidth": 1.0, "zorder": 3},
    "+KiTE (Ours)": {"marker": "s", "linestyle": "-", "alpha": 0.88, "linewidth": 1.0, "zorder": 6},
    "+idb-RRT": {"marker": "X", "linestyle": ":", "alpha": 0.88, "linewidth": 1.0, "zorder": 3},
    "dbCBS": {"marker": "D", "linestyle": "-.", "alpha": 0.88, "linewidth": 1.0, "zorder": 3},
}
ROLE_LEGEND_LABELS = {
    "Rand": "Rand (dashed)",
    "+KiTE (Ours)": r"+KiTE ($\mathbf{Ours}$, solid)",
    "+idb-RRT": "+idb-RRT (dotted)",
    "dbCBS": "dbCBS (dash-dot)",
}
METRICS = {
    "computation_time": MetricSpec(
        key="computation_time",
        csv_column="Computation Time (s)",
        external_column="time",
        label="Computation Time (CT, s)",
        log_y=True,
        output_name="computation_time_all_paradigms",
    ),
    "total_path_time": MetricSpec(
        key="total_path_time",
        csv_column="Total Path Costs",
        external_column="cost",
        label="Total Path Time (PT)",
        log_y=False,
        output_name="total_path_time_all_paradigms",
    ),
}


def load_dbcbs_metric(metric: MetricSpec) -> dict[tuple[str, str, int], float]:
    if not DBCBS_NORMALIZED_CSV.exists():
        return {}
    values = {}
    with DBCBS_NORMALIZED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] == "UCYCLE" and row.get("radius", "0.3") != UCYCLE_RADIUS:
                continue
            raw = row.get(metric.external_column, "")
            if raw:
                values[(row["system"], row["environment"], int(row["agents"]))] = float(raw)
    return values


def successful_mean(csv_path: Path, column: str) -> float | None:
    values = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Success"].strip().lower() != "true":
                continue
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)
    return mean(values) if values else None


def collect(results_root: Path, metric: MetricSpec, ucycle_results_root: Path | None = None):
    external_values = load_dbcbs_metric(metric)
    data = {
        paradigm: {
            system: {
                environment: {planner.label: [] for planner in PARADIGMS[paradigm]["planners"]}
                for environment in SYSTEM_ENVIRONMENTS[system]
            }
            for system in SYSTEMS
        }
        for paradigm in PARADIGM_ORDER
    }

    for paradigm in PARADIGM_ORDER:
        planners = PARADIGMS[paradigm]["planners"]
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
                        if planner.external:
                            value = external_values.get((system, environment, agents))
                            if value is None:
                                continue
                            csv_path = DBCBS_NORMALIZED_CSV
                        else:
                            csv_path = run_dir / "csvs" / planner.csv_name
                            if not csv_path.exists():
                                continue
                            value = successful_mean(csv_path, metric.csv_column)
                            if value is None:
                                continue
                        data[paradigm][system][environment][planner.label].append({
                            "agents": agents,
                            "value": value,
                            "csv_path": str(csv_path),
                        })
    for paradigm_data in data.values():
        for system_data in paradigm_data.values():
            for env_data in system_data.values():
                for points in env_data.values():
                    points.sort(key=lambda row: int(row["agents"]))
    return data


def system_xticks(system: str) -> list[int]:
    if system == "QUAD":
        return [2, 5, 10, 15, 20, 25, 30]
    return [3, 5, 10, 15, 20, 25, 30]


def dodged_marker_positions(
    marker_records: list[dict[str, object]],
    environment_order: tuple[str, ...],
    system: str,
) -> list[dict[str, object]]:
    env_index = {environment: idx for idx, environment in enumerate(environment_order)}
    spread = 0.28 if len(environment_order) >= 4 else 0.18
    if system == "QUAD":
        spread *= 0.85
    y_tolerance = 0.03

    records_by_agent: dict[int, list[dict[str, object]]] = {}
    for record in marker_records:
        records_by_agent.setdefault(int(record["agent"]), []).append(record)

    groups: list[list[dict[str, object]]] = []
    for records in records_by_agent.values():
        values = [float(record["value"]) for record in records]
        span = max(values) - min(values) if values else 0.0
        tolerance = max(span * 0.015, y_tolerance)
        current: list[dict[str, object]] = []
        for record in sorted(records, key=lambda item: float(item["value"])):
            if current and abs(float(record["value"]) - float(current[-1]["value"])) > tolerance:
                groups.append(current)
                current = []
            current.append(record)
        if current:
            groups.append(current)

    dodged: list[dict[str, object]] = []
    for records in groups:
        if len(records) == 1:
            record = dict(records[0])
            record["x"] = float(record["agent"])
            dodged.append(record)
            continue
        ordered = sorted(records, key=lambda record: (env_index[str(record["environment"])], str(record["role"])))
        center = (len(ordered) - 1) / 2.0
        for idx, record in enumerate(ordered):
            shifted = dict(record)
            shifted["x"] = float(record["agent"]) + (idx - center) * spread / max(center, 1.0)
            dodged.append(shifted)
    return dodged


def positive_limits(values: list[float]) -> tuple[float, float]:
    positive = [value for value in values if value > 0]
    if not positive:
        return (1e-2, 1.0)
    return max(min(positive) / 1.8, 1e-4), max(positive) * 1.8


def linear_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    return 0.0, max(values) * 1.12


def plot(data, metric: MetricSpec, output_path: Path) -> None:
    plt.rcParams.update({
        "figure.figsize": (7.05, 5.65),
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.65,
    })
    fig, axes = plt.subplots(len(SYSTEMS), len(PARADIGM_ORDER))

    for row, system in enumerate(SYSTEMS):
        for col, paradigm in enumerate(PARADIGM_ORDER):
            planners = PARADIGMS[paradigm]["planners"]
            draw_order = sorted(planners, key=lambda planner: 1 if planner.role == "+KiTE (Ours)" else 0)
            ax = axes[row][col]
            all_agents = set()
            all_values = []
            for environment in SYSTEM_ENVIRONMENTS[system]:
                for planner in planners:
                    for point in data[paradigm][system][environment][planner.label]:
                        all_agents.add(int(point["agents"]))
                        all_values.append(float(point["value"]))
            ticks = [tick for tick in system_xticks(system) if all_agents and min(all_agents) <= tick <= max(all_agents)]
            tick_set = set(ticks)

            marker_records: list[dict[str, object]] = []
            for environment in SYSTEM_ENVIRONMENTS[system]:
                color = ENVIRONMENT_COLORS[environment]
                for planner in draw_order:
                    points = data[paradigm][system][environment][planner.label]
                    if not points:
                        continue
                    agents = [int(point["agents"]) for point in points]
                    values = [float(point["value"]) for point in points]
                    style = ROLE_STYLES[planner.role]
                    marker_indices = [idx for idx, agent in enumerate(agents) if agent in tick_set]
                    ax.plot(
                        agents,
                        values,
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=style["alpha"],
                        zorder=style["zorder"],
                    )
                    for idx in marker_indices:
                        marker_records.append({
                            "agent": agents[idx],
                            "value": values[idx],
                            "role": planner.role,
                            "environment": environment,
                            "color": color,
                            "marker": style["marker"],
                            "markersize": 4.0,
                            "alpha": style["alpha"],
                            "zorder": style["zorder"] + 0.2,
                        })

            for marker in dodged_marker_positions(marker_records, SYSTEM_ENVIRONMENTS[system], system):
                ax.plot(
                    [float(marker["x"])],
                    [float(marker["value"])],
                    color=str(marker["color"]),
                    marker=str(marker["marker"]),
                    linestyle="none",
                    markersize=float(marker["markersize"]),
                    markerfacecolor="none",
                    markeredgewidth=0.95,
                    alpha=float(marker["alpha"]),
                    zorder=float(marker["zorder"]),
                )

            if row == 0:
                ax.set_title(PARADIGMS[paradigm]["row_label"], pad=4)
            if col == 0:
                ax.set_ylabel(SYSTEM_LABELS[system], labelpad=5, fontsize=10.0, fontweight="bold")
            else:
                ax.tick_params(labelleft=False)
            if row == len(SYSTEMS) - 1:
                ax.set_xlabel("Robots", labelpad=1.0, fontsize=10.0, fontweight="bold")
            if all_agents:
                ax.set_xlim(min(all_agents) - 0.6, max(all_agents) + 0.6)
                ax.set_xticks(ticks)
            if metric.log_y:
                ax.set_yscale("log")
                ax.set_ylim(*positive_limits(all_values))
            else:
                ax.set_ylim(*linear_limits(all_values))
            ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.45, zorder=0, which="major")
            ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.28, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", pad=0.8)
            ax.tick_params(axis="y", pad=0.8)

    fig.supylabel(metric.label, x=0.008, fontsize=10.0, fontweight="bold")

    role_handles = [
        Line2D([0, 1, 2], [0, 0, 0], color="#222222", marker=style["marker"],
               linestyle=style["linestyle"], markevery=[0, 1, 2],
               markerfacecolor="none", markeredgewidth=1.05,
               markersize=5.6 if role == "+KiTE (Ours)" else (5.2 if role == "+idb-RRT" else 4.8),
               lw=1.65, label=ROLE_LEGEND_LABELS[role])
        for role, style in ROLE_STYLES.items()
    ]
    env_handles = [
        Line2D([0], [0], color=ENVIRONMENT_COLORS[env], lw=1.8, label=label)
        for env, label in ENVIRONMENT_LABELS.items()
    ]
    variant_label = Line2D([], [], color="none", linestyle="none", label="Variant:")
    environment_label = Line2D([], [], color="none", linestyle="none", label="Environment:")
    fig.legend(
        handles=[variant_label] + role_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=len(role_handles) + 1,
        frameon=False,
        columnspacing=0.90,
        handlelength=2.85,
        handletextpad=0.34,
    )
    fig.legend(
        handles=[environment_label] + env_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(env_handles) + 1,
        frameon=False,
        columnspacing=0.38,
        handlelength=0.95,
        handletextpad=0.24,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.030, 0, 1, 0.875), pad=0.16, w_pad=0.40, h_pad=0.58)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(data, metric: MetricSpec, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = ["metric", "paradigm", "system", "radius", "environment", "planner", "agents", "value", "csv_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paradigm in PARADIGM_ORDER:
            for system in SYSTEMS:
                for environment in SYSTEM_ENVIRONMENTS[system]:
                    for planner in PARADIGMS[paradigm]["planners"]:
                        for point in data[paradigm][system][environment][planner.label]:
                            writer.writerow({
                                "metric": metric.key,
                                "paradigm": paradigm,
                                "system": system,
                                "radius": row_radius(system, str(point["csv_path"])),
                                "environment": environment,
                                "planner": planner.label,
                                "agents": point["agents"],
                                "value": f"{point['value']:.6f}",
                                "csv_path": point["csv_path"],
                            })


def main() -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None
    output_root = Path("paper_plots/figures/metrics")
    alias_roots = {
        "computation_time": Path("paper_plots/figures/computation_time"),
        "total_path_time": Path("paper_plots/figures/total_path_cost"),
    }
    for metric in METRICS.values():
        data = collect(root, metric, ucycle_root)
        output = output_root / f"{metric.output_name}.png"
        plot(data, metric, output)
        csv_output = csv_output_path(output)
        write_csv(data, metric, csv_output)
        print(f"Wrote {output}")
        print(f"Wrote {csv_output}")

        alias_root = alias_roots[metric.key]
        alias_output = alias_root / f"{metric.output_name}.png"
        alias_csv_output = csv_output_path(alias_output)
        alias_output.parent.mkdir(parents=True, exist_ok=True)
        alias_csv_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, alias_output)
        shutil.copy2(csv_output, alias_csv_output)
        print(f"Wrote {alias_output}")
        print(f"Wrote {alias_csv_output}")


if __name__ == "__main__":
    main()
