#!/usr/bin/env python3
"""Plot KiTE/baseline metric ratios vs. number of agents."""

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

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, summary_radius, system_results_root


RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")


@dataclass(frozen=True)
class PlannerPair:
    baseline_csv: str
    kite_csv: str
    baseline_label: str
    kite_label: str
    ratio_label: str


@dataclass(frozen=True)
class MetricSpec:
    key: str
    column: str
    ylabel: str
    title: str
    output_dir: str


PARADIGMS = {
    "centralized": {
        "title": "centralized planning",
        "pair": PlannerPair(
            baseline_csv="CRRT_results.csv",
            kite_csv="K-TI_EB_CRRT_results.csv",
            baseline_label="cRRT",
            kite_label="cRRT+KiTE",
            ratio_label="cRRT+KiTE / cRRT",
        ),
    },
    "prrt": {
        "title": "prioritized planning",
        "pair": PlannerPair(
            baseline_csv="PRRT_results.csv",
            kite_csv="KTI_EB_PRRT_results.csv",
            baseline_label="pRRT",
            kite_label="pRRT+KiTE",
            ratio_label="pRRT+KiTE / pRRT",
        ),
    },
    "kcbs": {
        "title": "KCBS",
        "pair": PlannerPair(
            baseline_csv="RRT_KCBS_results.csv",
            kite_csv="KTI_EB_RRT_KCBS_results.csv",
            baseline_label="KCBS",
            kite_label="KCBS+KiTE",
            ratio_label="KCBS+KiTE / KCBS",
        ),
    },
}

METRICS = {
    "computation_time": MetricSpec(
        key="computation_time",
        column="Computation Time (s)",
        ylabel="Mean CT ratio",
        title="Computation Time (CT) ratio",
        output_dir="computation_time_ratio",
    ),
    "total_path_cost": MetricSpec(
        key="total_path_cost",
        column="Total Path Costs",
        ylabel="Mean PT ratio",
        title="Total Path Time (PT) ratio",
        output_dir="total_path_cost_ratio",
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
    "small_cluttered_env": "#C51B7D",
    "large_cluttered_env": "#FE6100",
    "swap_env": "#6B8E23",
    "narrow_corridor_env": "#8A8A8A",
    "large_cluttered_3d_env": "#E69F00",
    "swap_3d_env": "#332288",
}


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
        if math.isfinite(value) and value > 0:
            values.append(value)

    return (mean(values) if values else None), len(values), len(rows)


def collect_ratio_data(
    results_root: Path,
    environment: str,
    system: str,
    pair: PlannerPair,
    metric: MetricSpec,
) -> list[dict[str, float | int | str]]:
    env_dir = environment_results_dir(results_root, system, environment)
    if not env_dir.exists():
        raise FileNotFoundError(f"Missing environment results directory: {env_dir}")

    points = []
    for run_dir in sorted(env_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_RE.match(run_dir.name)
        if not match or match.group("system") != system:
            continue

        baseline_path = run_dir / "csvs" / pair.baseline_csv
        kite_path = run_dir / "csvs" / pair.kite_csv
        if not baseline_path.exists() or not kite_path.exists():
            continue

        baseline_mean, baseline_successes, baseline_total = successful_metric_mean(baseline_path, metric)
        kite_mean, kite_successes, kite_total = successful_metric_mean(kite_path, metric)
        if baseline_mean is None or kite_mean is None or baseline_mean <= 0:
            continue

        agents = int(match.group("agents"))
        points.append(
            {
                "agents": agents,
                "ratio": kite_mean / baseline_mean,
                "baseline_mean": baseline_mean,
                "kite_mean": kite_mean,
                "baseline_successes": baseline_successes,
                "baseline_total": baseline_total,
                "kite_successes": kite_successes,
                "kite_total": kite_total,
                "baseline_csv": str(baseline_path),
                "kite_csv": str(kite_path),
            }
        )

    points.sort(key=lambda row: int(row["agents"]))
    return points


def collect_system_data(
    results_root: Path,
    system: str,
    pair: PlannerPair,
    metric: MetricSpec,
) -> dict[str, list[dict[str, float | int | str]]]:
    return {
        environment: collect_ratio_data(results_root, environment, system, pair, metric)
        for environment in SYSTEM_ENVIRONMENTS[system]
    }


def write_systems_summary_csv(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "radius",
        "environment",
        "agents",
        "ratio",
        "baseline_mean",
        "kite_mean",
        "baseline_successes",
        "baseline_total",
        "kite_successes",
        "kite_total",
        "baseline_csv",
        "kite_csv",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system, env_data in data.items():
            for environment, points in env_data.items():
                for point in points:
                    writer.writerow(
                        {
                            "system": system,
                            "radius": summary_radius(system, UCYCLE_RADIUS),
                            "environment": environment,
                            "agents": point["agents"],
                            "ratio": f"{float(point['ratio']):.6f}",
                            "baseline_mean": f"{float(point['baseline_mean']):.6f}",
                            "kite_mean": f"{float(point['kite_mean']):.6f}",
                            "baseline_successes": point["baseline_successes"],
                            "baseline_total": point["baseline_total"],
                            "kite_successes": point["kite_successes"],
                            "kite_total": point["kite_total"],
                            "baseline_csv": point["baseline_csv"],
                            "kite_csv": point["kite_csv"],
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
    env_data: dict[str, list[dict[str, float | int | str]]],
    system: str,
    metric: MetricSpec,
    show_ylabel: bool,
    y_limits: tuple[float, float] | None = None,
) -> None:
    all_agents: set[int] = set()
    ratios = []

    for environment in SYSTEM_ENVIRONMENTS[system]:
        points = env_data[environment]
        if not points:
            continue
        agents = [int(point["agents"]) for point in points]
        values = [float(point["ratio"]) for point in points]
        all_agents.update(agents)
        ratios.extend(values)
        ax.plot(
            agents,
            values,
            color=ENVIRONMENT_COLORS[environment],
            marker="o",
            linestyle="-",
            markerfacecolor="white",
            markeredgewidth=1.05,
            alpha=0.95,
            zorder=3,
        )

    ax.axhline(1.0, color="#2F3437", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_title(SYSTEM_LABELS[system], pad=4, fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Robots", labelpad=1.5, fontsize=10.0, fontweight="bold")
    ax.set_yscale("log")
    if not all_agents:
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_ylim(*(y_limits or (0.5, 2.0)))
        if show_ylabel:
            ax.set_ylabel(metric.ylabel, fontsize=10.0, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.65, which="major")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif ratios:
        ymin = max(min(ratios) * 0.65, 1e-3)
        ymax = max(ratios) * 1.55
        ax.set_ylim(ymin, ymax)
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


def legend_handles():
    from matplotlib.lines import Line2D

    return [
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


def plot_ratio_legend(output_path: Path) -> None:
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (5.5, 0.36), "legend.fontsize": 7})
    fig = plt.figure()
    handles = legend_handles()
    fig.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def global_ratio_limits(data: dict[str, dict[str, list[dict[str, float | int | str]]]]) -> tuple[float, float]:
    values = [
        float(point["ratio"])
        for env_data in data.values()
        for points in env_data.values()
        for point in points
        if float(point["ratio"]) > 0
    ]
    if not values:
        return (0.5, 2.0)
    lo = min(values)
    hi = max(values)
    ymin = max(lo / 1.8, 1e-3)
    ymax = hi * 1.8
    ymin = min(ymin, 0.95)
    ymax = max(ymax, 1.05)
    return ymin, ymax


def plot_all_systems_ratio(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    metric: MetricSpec,
    output_stem: Path,
) -> None:
    configure_matplotlib()
    y_limits = global_ratio_limits(data)
    fig, axes = plt.subplots(1, 3, sharey=True)
    for index, system in enumerate(("UCYCLE", "SOC", "QUAD")):
        draw_system_panel(
            axes[index],
            data[system],
            system=system,
            metric=metric,
            show_ylabel=index == 0,
            y_limits=y_limits,
        )

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=6,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.95,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80), pad=0.22, w_pad=0.65)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KiTE/baseline successful-trial mean ratio plots.")
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
    pair = PARADIGMS[args.paradigm]["pair"]
    output_root = args.output_dir / "ratios" / metric.output_dir

    systems_data = {
        system: collect_system_data(
            system_results_root(args.results_root, system, args.ucycle_results_root),
            system,
            pair,
            metric,
        )
        for system in ("UCYCLE", "SOC", "QUAD")
    }

    output_stem = output_root / f"{metric.key}_ratio_{args.paradigm}_all_systems"
    if args.write_csv:
        write_systems_summary_csv(systems_data, csv_output_path(output_stem))
    plot_all_systems_ratio(systems_data, metric, output_stem)
    legend_path = output_root / f"{metric.key}_ratio_{args.paradigm}_legend.png"
    plot_ratio_legend(legend_path)
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {legend_path}")
    if args.write_csv:
        print(f"Wrote {csv_output_path(output_stem)}")


if __name__ == "__main__":
    main()
