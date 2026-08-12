#!/usr/bin/env python3
"""Plot successful-trial mean metrics vs. number of agents for paper figures."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class MetricSpec:
    key: str
    column: str
    ylabel: str
    title: str
    log_y: bool
    output_dir: str


PARADIGMS = {
    "centralized": {
        "title": "centralized planning",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE"),
        ),
    },
    "prrt": {
        "title": "prioritized planning",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE"),
        ),
    },
    "kcbs": {
        "title": "KCBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "KCBS"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "KCBS+KiTE"),
            PlannerSpec("dbRRT_KCBS_results.csv", "KCBS+idb-RRT"),
        ),
    },
}

METRICS = {
    "computation_time": MetricSpec(
        key="computation_time",
        column="Computation Time (s)",
        ylabel="Mean CT (s)",
        title="Computation Time (CT)",
        log_y=True,
        output_dir="computation_time",
    ),
    "total_path_cost": MetricSpec(
        key="total_path_cost",
        column="Total Path Costs",
        ylabel="Mean PT",
        title="Total Path Time (PT)",
        log_y=False,
        output_dir="total_path_cost",
    ),
}

SYSTEM_ENVIRONMENTS = {
    "SOC": (
        "small_cluttered_env",
        "large_cluttered_env",
        "swap_env",
        "narrow_corridor_env",
    ),
    "UCYCLE": (
        "small_cluttered_env",
        "large_cluttered_env",
        "swap_env",
        "narrow_corridor_env",
    ),
    "QUAD": (
        "large_cluttered_3d_env",
        "swap_3d_env",
    ),
}

SYSTEM_LABELS = {
    "SOC": "SOC",
    "UCYCLE": "UC",
    "QUAD": "DI",
}

ENVIRONMENT_LABELS = {
    "small_cluttered_env": "Small",
    "large_cluttered_env": "Large",
    "swap_env": "Swap",
    "narrow_corridor_env": "Corridor",
    "large_cluttered_3d_env": "Large 3D",
    "swap_3d_env": "Swap 3D",
}

ENVIRONMENT_COLORS = {
    "small_cluttered_env": "#A50F15",
    "large_cluttered_env": "#E66101",
    "swap_env": "#006B3C",
    "narrow_corridor_env": "#8A8A8A",
    "large_cluttered_3d_env": "#E69F00",
    "swap_3d_env": "#332288",
}

METHOD_STYLE_SEQUENCE = (
    {"linestyle": "--", "marker": "o"},
    {"linestyle": "-", "marker": "s"},
    {"linestyle": ":", "marker": "^"},
)


def method_styles(planners: tuple[PlannerSpec, ...]) -> dict[str, dict[str, str]]:
    return {planner.label: METHOD_STYLE_SEQUENCE[index] for index, planner in enumerate(planners)}



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
        ordered = sorted(records, key=lambda record: (env_index[str(record["environment"])], str(record["planner"])))
        center = (len(ordered) - 1) / 2.0
        for idx, record in enumerate(ordered):
            shifted = dict(record)
            shifted["x"] = float(record["agent"]) + (idx - center) * spread / max(center, 1.0)
            dodged.append(shifted)
    return dodged

def successful_metric_mean(csv_path: Path, metric: MetricSpec) -> tuple[float | None, int, int]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    values = []
    for row in rows:
        if row["Success"].strip().lower() != "true":
            continue
        try:
            value = float(row[metric.column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)

    return (mean(values) if values else None), len(values), len(rows)


def collect_metric_data(
    results_root: Path,
    environment: str,
    system: str,
    planners: tuple[PlannerSpec, ...],
    metric: MetricSpec,
) -> dict[str, list[dict[str, float | int | str]]]:
    env_dir = environment_results_dir(results_root, system, environment)
    if not env_dir.exists():
        raise FileNotFoundError(f"Missing environment results directory: {env_dir}")

    series: dict[str, list[dict[str, float | int | str]]] = {planner.label: [] for planner in planners}

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

            value, successes, total = successful_metric_mean(csv_path, metric)
            if value is None:
                continue
            series[planner.label].append(
                {
                    "agents": agents,
                    "value": value,
                    "successes": successes,
                    "total": total,
                    "csv_path": str(csv_path),
                }
            )

    any_points = False
    for points in series.values():
        points.sort(key=lambda row: int(row["agents"]))
        any_points = any_points or bool(points)
    if not any_points:
        raise ValueError(f"No successful-trial metric data found in {env_dir} with system={system}")

    return series


def write_systems_summary_csv(
    data: dict[str, dict[str, dict[str, list[dict[str, float | int | str]]]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "radius", "environment", "planner", "agents", "value", "successful_trials", "total_trials", "csv_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system, env_data in data.items():
            for environment, series in env_data.items():
                for planner, points in series.items():
                    for point in points:
                        writer.writerow(
                            {
                                "system": system,
                                "radius": row_radius(system, str(point["csv_path"])),
                                "environment": environment,
                                "planner": planner,
                                "agents": point["agents"],
                                "value": f"{float(point['value']):.6f}",
                                "successful_trials": point["successes"],
                                "total_trials": point["total"],
                                "csv_path": point["csv_path"],
                            }
                        )


def write_multi_env_summary_csv(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["radius", "environment", "planner", "agents", "value", "successful_trials", "total_trials", "csv_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for environment, series in data.items():
            for planner, points in series.items():
                for point in points:
                    writer.writerow(
                        {
                            "radius": row_radius("UCYCLE" if "UCYCLE_" in str(point["csv_path"]) else "", str(point["csv_path"])),
                            "environment": environment,
                            "planner": planner,
                            "agents": point["agents"],
                            "value": f"{float(point['value']):.6f}",
                            "successful_trials": point["successes"],
                            "total_trials": point["total"],
                            "csv_path": point["csv_path"],
                        }
                    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.05, 2.62),
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.05,
        }
    )


def system_xticks(system: str) -> list[int]:
    if system == "QUAD":
        return [2, 5, 10, 15, 20, 25, 30]
    return [3, 5, 10, 15, 20, 25, 30]


def draw_system_panel(
    ax: plt.Axes,
    env_data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    system: str,
    planners: tuple[PlannerSpec, ...],
    metric: MetricSpec,
    show_ylabel: bool,
    y_limits: tuple[float, float] | None = None,
) -> None:
    all_agents: set[int] = set()
    plotted_values: list[float] = []
    styles = method_styles(planners)

    marker_records: list[dict[str, object]] = []
    for environment in SYSTEM_ENVIRONMENTS[system]:
        series = env_data[environment]
        color = ENVIRONMENT_COLORS[environment]
        for planner in planners:
            points = series[planner.label]
            if not points:
                continue
            agents = [int(point["agents"]) for point in points]
            values = [float(point["value"]) for point in points]
            all_agents.update(agents)
            plotted_values.extend(values)
            style = styles[planner.label]
            ax.plot(
                agents,
                values,
                color=color,
                linestyle=style["linestyle"],
                alpha=0.95,
                zorder=3,
            )
            for agent, value in zip(agents, values):
                marker_records.append({
                    "agent": agent,
                    "value": value,
                    "planner": planner.label,
                    "environment": environment,
                    "color": color,
                    "marker": style["marker"],
                    "markersize": 4.0,
                    "alpha": 0.95,
                    "zorder": 3.2,
                })

    for marker in dodged_marker_positions(marker_records, SYSTEM_ENVIRONMENTS[system], system):
        ax.plot(
            [float(marker["x"])],
            [float(marker["value"])],
            color=str(marker["color"]),
            marker=str(marker["marker"]),
            linestyle="none",
            markersize=float(marker["markersize"]),
            markerfacecolor="white",
            markeredgewidth=1.05,
            alpha=float(marker["alpha"]),
            zorder=float(marker["zorder"]),
        )

    ax.set_title(SYSTEM_LABELS[system], pad=4, fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Robots", labelpad=1.5, fontsize=10.0, fontweight="bold")
    if metric.log_y:
        ax.set_yscale("log")
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        else:
            positive = [value for value in plotted_values if value > 0]
            if positive:
                ymin = max(min(positive) * 0.65, 1e-3)
                ymax = max(positive) * 1.55
                ax.set_ylim(ymin, ymax)
    elif y_limits is not None:
        ax.set_ylim(*y_limits)
    if show_ylabel:
        ax.set_ylabel(metric.ylabel, fontsize=10.0, fontweight="bold")
    else:
        ax.tick_params(labelleft=False)
    ticks = [tick for tick in system_xticks(system) if min(all_agents) <= tick <= max(all_agents)]
    ax.set_xlim(min(all_agents) - 0.6, max(all_agents) + 0.6)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", rotation=0, pad=1.5)
    ax.tick_params(axis="y", pad=1.5)
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.65, which="major")
    ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles(planners: tuple[PlannerSpec, ...]):
    from matplotlib.lines import Line2D

    env_handles = [
        Line2D([0], [0], color=ENVIRONMENT_COLORS[environment], lw=2.0, label=ENVIRONMENT_LABELS[environment])
        for environment in (
            "small_cluttered_env",
            "large_cluttered_env",
            "swap_env",
            "narrow_corridor_env",
            "large_cluttered_3d_env",
            "swap_3d_env",
        )
    ]
    styles = method_styles(planners)
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            marker=styles[planner.label]["marker"],
            linestyle=styles[planner.label]["linestyle"],
            markerfacecolor="white",
            markeredgewidth=1.05,
            lw=1.55,
            label=planner.label,
        )
        for planner in planners
    ]
    return method_handles + env_handles


def plot_metric_legend(output_path: Path, planners: tuple[PlannerSpec, ...]) -> None:
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (7.0, 0.42), "legend.fontsize": 7})
    fig = plt.figure()
    handles = legend_handles(planners)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=min(8, len(handles)),
        frameon=False,
        columnspacing=0.85,
        handlelength=1.75,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def global_metric_limits(
    data: dict[str, dict[str, dict[str, list[dict[str, float | int | str]]]]],
    metric: MetricSpec,
) -> tuple[float, float] | None:
    if not metric.log_y:
        return None
    values = [
        float(point["value"])
        for env_data in data.values()
        for series in env_data.values()
        for points in series.values()
        for point in points
        if float(point["value"]) > 0
    ]
    if not values:
        return None
    return max(min(values) / 1.8, 1e-3), max(values) * 1.8


def plot_all_systems_metric(
    data: dict[str, dict[str, dict[str, list[dict[str, float | int | str]]]]],
    planners: tuple[PlannerSpec, ...],
    metric: MetricSpec,
    output_stem: Path,
) -> None:
    configure_matplotlib()
    y_limits = global_metric_limits(data, metric)
    fig, axes = plt.subplots(1, 3, sharey=metric.log_y)
    for index, system in enumerate(("UCYCLE", "SOC", "QUAD")):
        draw_system_panel(
            axes[index],
            data[system],
            system=system,
            planners=planners,
            metric=metric,
            show_ylabel=index == 0,
            y_limits=y_limits,
        )

    fig.legend(
        handles=legend_handles(planners),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3 if len(planners) == 3 else 4,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.95,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80), pad=0.22, w_pad=0.65)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_system_metric(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    system: str,
    planners: tuple[PlannerSpec, ...],
    metric: MetricSpec,
    output_stem: Path,
) -> None:
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (4.9, 2.85), "lines.linewidth": 1.65, "lines.markersize": 4.6})
    fig, ax = plt.subplots()
    draw_system_panel(ax, data, system=system, planners=planners, metric=metric, show_ylabel=True)
    ax.set_title(SYSTEM_LABELS[system], pad=7, fontsize=10.4, fontweight="bold")
    fig.tight_layout(pad=0.35)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def collect_system_data(
    results_root: Path,
    system: str,
    planners: tuple[PlannerSpec, ...],
    metric: MetricSpec,
) -> dict[str, dict[str, list[dict[str, float | int | str]]]]:
    return {
        environment: collect_metric_data(
            results_root=results_root,
            environment=environment,
            system=system,
            planners=planners,
            metric=metric,
        )
        for environment in SYSTEM_ENVIRONMENTS[system]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate successful-trial mean metric scaling plots.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--ucycle-results-root", type=Path, default=None)
    parser.add_argument("--metric", choices=sorted(METRICS), required=True)
    parser.add_argument("--paradigm", choices=sorted(PARADIGMS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_plots/figures"))
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric = METRICS[args.metric]
    paradigm = PARADIGMS[args.paradigm]
    planners = paradigm["planners"]
    output_root = args.output_dir / metric.output_dir

    systems_data = {
        system: collect_system_data(
            system_results_root(args.results_root, system, args.ucycle_results_root),
            system,
            planners,
            metric,
        )
        for system in ("UCYCLE", "SOC", "QUAD")
    }

    combined_stem = output_root / f"{metric.key}_{args.paradigm}_all_systems"
    if args.write_csv:
        write_systems_summary_csv(systems_data, csv_output_path(combined_stem))
    plot_all_systems_metric(systems_data, planners, metric, combined_stem)
    legend_path = output_root / f"{metric.key}_{args.paradigm}_legend.png"
    plot_metric_legend(legend_path, planners)
    print(f"Wrote {combined_stem.with_suffix('.png')}")
    print(f"Wrote {legend_path}")
    if args.write_csv:
        print(f"Wrote {csv_output_path(combined_stem)}")

    for system, data in systems_data.items():
        output_stem = output_root / f"{metric.key}_{args.paradigm}_{system}_all_envs_no_legend"
        if system == "UCYCLE":
            output_stem = output_stem
        if args.write_csv:
            write_multi_env_summary_csv(data, csv_output_path(output_stem))
        plot_system_metric(data, system, planners, metric, output_stem)
        print(f"Wrote {output_stem.with_suffix('.png')}")
        if args.write_csv:
            print(f"Wrote {csv_output_path(output_stem)}")


if __name__ == "__main__":
    main()
