#!/usr/bin/env python3
"""Compact CBS-method comparison heatmap."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, summary_radius, system_results_root

RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    csv_name: str
    external: bool = False


BASELINE = MethodSpec("kcbs", "KCBS", "RRT_KCBS_results.csv")
METHODS = (
    MethodSpec("kite", "KCBS\n+KiTE", "KTI_EB_RRT_KCBS_results.csv"),
    MethodSpec("idb", "KCBS\n+idb-RRT", "dbRRT_KCBS_results.csv"),
    MethodSpec("dbcbs", "dbCBS", "", external=True),
)
DBCBS_NORMALIZED_CSV = Path("paper_plots/data/dbcbs_normalized.csv")
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")

SYSTEMS = ("UCYCLE", "QUAD")
SYSTEM_LABELS = {"UCYCLE": "UC", "QUAD": "DI"}
SYSTEM_ENVIRONMENTS = {
    "UCYCLE": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "QUAD": ("large_cluttered_3d_env", "swap_3d_env"),
}
METRIC_COLUMNS = {"time": "Computation Time (s)", "cost": "Total Path Costs"}


def rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def success_rate(csv_path: Path) -> float | None:
    data = rows(csv_path)
    if not data:
        return None
    return sum(row["Success"].strip().lower() == "true" for row in data) / len(data)


def successful_stat(csv_path: Path, column: str, statistic: str) -> float | None:
    values = []
    for row in rows(csv_path):
        if row["Success"].strip().lower() != "true":
            continue
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            values.append(value)
    if not values:
        return None
    if statistic == "mean":
        return mean(values)
    if statistic == "median":
        return median(values)
    raise ValueError(f"Unknown statistic: {statistic}")


def paired_successful_ratio(
    baseline_path: Path,
    method_path: Path,
    column: str,
    statistic: str,
) -> tuple[float | None, int]:
    baseline_rows = {row["Seed"]: row for row in rows(baseline_path)}
    method_rows = {row["Seed"]: row for row in rows(method_path)}
    baseline_values = []
    method_values = []
    for seed in sorted(set(baseline_rows) & set(method_rows), key=int):
        baseline_row = baseline_rows[seed]
        method_row = method_rows[seed]
        if baseline_row["Success"].strip().lower() != "true":
            continue
        if method_row["Success"].strip().lower() != "true":
            continue
        try:
            baseline_value = float(baseline_row[column])
            method_value = float(method_row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(baseline_value) and math.isfinite(method_value)):
            continue
        if baseline_value <= 0 or method_value <= 0:
            continue
        baseline_values.append(baseline_value)
        method_values.append(method_value)
    if not baseline_values or not method_values:
        return None, 0
    if statistic == "mean":
        return mean(method_values) / mean(baseline_values), len(method_values)
    if statistic == "median":
        return median(method_values) / median(baseline_values), len(method_values)
    raise ValueError(f"Unknown statistic: {statistic}")

def matched_run_dirs(root: Path, system: str, environment: str):
    env_dir = environment_results_dir(root, system, environment)
    for run_dir in sorted(env_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_RE.match(run_dir.name)
        if match and match.group("system") == system:
            yield run_dir


def run_metadata(run_dir: Path) -> tuple[str, int] | None:
    match = RUN_DIR_RE.match(run_dir.name)
    if not match:
        return None
    return match.group("system"), int(match.group("agents"))


def load_dbcbs_results(path: Path) -> dict[tuple[str, str, int], dict[str, float | None]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run paper_plotting_scripts/extract_dbcbs_results.py before plotting."
        )
    external = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] == "UCYCLE" and row.get("radius", "0.3") != UCYCLE_RADIUS:
                continue
            key = (row["system"], row["environment"], int(row["agents"]))
            external[key] = {
                "success": float(row["success_rate"]) if row["success_rate"] else None,
                "time": float(row["time"]) if row["time"] else None,
                "cost": float(row["cost"]) if row["cost"] else None,
            }
    return external


def geometric_mean(values: list[float]) -> float:
    return math.exp(mean(math.log(value) for value in values if value > 0))


def aggregate_ratio(values: list[float], aggregation: str = "geomean") -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return math.nan
    if aggregation == "median":
        return median(positive)
    if aggregation == "geomean":
        return geometric_mean(positive)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def collect_summary(root: Path, statistic: str = "mean", ucycle_results_root: Path | None = None):
    dbcbs_results = load_dbcbs_results(DBCBS_NORMALIZED_CSV)
    summary = {
        system: {
            method.key: {
                "success_rates": [],
                "time_ratios": [],
                "cost_ratios": [],
                "success_n": 0,
                "time_n": 0,
                "cost_n": 0,
            }
            for method in METHODS
        }
        for system in SYSTEMS
    }

    for system in SYSTEMS:
        system_root = system_results_root(root, system, ucycle_results_root)
        for environment in SYSTEM_ENVIRONMENTS[system]:
            for run_dir in matched_run_dirs(system_root, system, environment):
                metadata = run_metadata(run_dir)
                if metadata is None:
                    continue
                _, agents = metadata
                baseline_path = run_dir / "csvs" / BASELINE.csv_name
                if not baseline_path.exists():
                    continue
                baseline_time = successful_stat(baseline_path, METRIC_COLUMNS["time"], statistic)
                baseline_cost = successful_stat(baseline_path, METRIC_COLUMNS["cost"], statistic)

                for method in METHODS:
                    if method.external:
                        external = dbcbs_results.get((system, environment, agents))
                        if external is None:
                            continue
                        sr = external["success"]
                        method_time = external["time"]
                        method_cost = external["cost"]
                    else:
                        method_path = run_dir / "csvs" / method.csv_name
                        if not method_path.exists():
                            continue
                        sr = success_rate(method_path)
                        method_time = successful_stat(method_path, METRIC_COLUMNS["time"], statistic)
                        method_cost = successful_stat(method_path, METRIC_COLUMNS["cost"], statistic)

                    if sr is not None:
                        summary[system][method.key]["success_rates"].append(sr)
                        summary[system][method.key]["success_n"] += 1

                    if baseline_time and method_time:
                        summary[system][method.key]["time_ratios"].append(method_time / baseline_time)
                        summary[system][method.key]["time_n"] += 1

                    if baseline_cost and method_cost:
                        summary[system][method.key]["cost_ratios"].append(method_cost / baseline_cost)
                        summary[system][method.key]["cost_n"] += 1

    compact = {system: {} for system in SYSTEMS}
    for system in SYSTEMS:
        for method in METHODS:
            cell = summary[system][method.key]
            compact[system][method.key] = {
                "success": 100.0 * mean(cell["success_rates"]) if cell["success_rates"] else math.nan,
                "time_ratio": aggregate_ratio(cell["time_ratios"]),
                "cost_ratio": aggregate_ratio(cell["cost_ratios"]),
                "success_n": cell["success_n"],
                "time_n": cell["time_n"],
                "cost_n": cell["cost_n"],
            }
    return compact


def collect_paired_summary(root: Path, statistic: str = "mean", ucycle_results_root: Path | None = None):
    dbcbs_results = load_dbcbs_results(DBCBS_NORMALIZED_CSV)
    summary = {
        system: {
            method.key: {
                "time_ratios": [],
                "cost_ratios": [],
                "time_n": 0,
                "cost_n": 0,
                "time_trials": 0,
                "cost_trials": 0,
            }
            for method in METHODS
        }
        for system in SYSTEMS
    }

    for system in SYSTEMS:
        system_root = system_results_root(root, system, ucycle_results_root)
        for environment in SYSTEM_ENVIRONMENTS[system]:
            for run_dir in matched_run_dirs(system_root, system, environment):
                metadata = run_metadata(run_dir)
                if metadata is None:
                    continue
                _, agents = metadata
                baseline_path = run_dir / "csvs" / BASELINE.csv_name
                if not baseline_path.exists():
                    continue
                baseline_time = successful_stat(baseline_path, METRIC_COLUMNS["time"], statistic)
                baseline_cost = successful_stat(baseline_path, METRIC_COLUMNS["cost"], statistic)

                for method in METHODS:
                    if method.external:
                        external = dbcbs_results.get((system, environment, agents))
                        if external is None:
                            continue
                        method_time = external["time"]
                        method_cost = external["cost"]
                        if baseline_time and method_time:
                            summary[system][method.key]["time_ratios"].append(method_time / baseline_time)
                            summary[system][method.key]["time_n"] += 1
                        if baseline_cost and method_cost:
                            summary[system][method.key]["cost_ratios"].append(method_cost / baseline_cost)
                            summary[system][method.key]["cost_n"] += 1
                        continue

                    method_path = run_dir / "csvs" / method.csv_name
                    if not method_path.exists():
                        continue
                    time_ratio, shared_time = paired_successful_ratio(
                        baseline_path, method_path, METRIC_COLUMNS["time"], statistic
                    )
                    if time_ratio:
                        summary[system][method.key]["time_ratios"].append(time_ratio)
                        summary[system][method.key]["time_n"] += 1
                        summary[system][method.key]["time_trials"] += shared_time
                    cost_ratio, shared_cost = paired_successful_ratio(
                        baseline_path, method_path, METRIC_COLUMNS["cost"], statistic
                    )
                    if cost_ratio:
                        summary[system][method.key]["cost_ratios"].append(cost_ratio)
                        summary[system][method.key]["cost_n"] += 1
                        summary[system][method.key]["cost_trials"] += shared_cost

    compact = {system: {} for system in SYSTEMS}
    for system in SYSTEMS:
        for method in METHODS:
            cell = summary[system][method.key]
            compact[system][method.key] = {
                "time_ratio": aggregate_ratio(cell["time_ratios"]),
                "cost_ratio": aggregate_ratio(cell["cost_ratios"]),
                "time_n": cell["time_n"],
                "cost_n": cell["cost_n"],
                "time_trials": cell["time_trials"],
                "cost_trials": cell["cost_trials"],
            }
    return compact

def matrix(summary, key: str) -> np.ndarray:
    return np.array([[summary[system][method.key][key] for method in METHODS] for system in SYSTEMS], dtype=float)


def annotate_percent(ax, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "--" if math.isnan(value) else f"{value:.0f}%"
            ax.text(col, row, text, ha="center", va="center", fontsize=8)


def annotate_ratio(ax, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "--" if math.isnan(value) else f"{value:.2f}x"
            text_color = "#111111"
            if math.isfinite(value) and value > 0 and abs(math.log10(value)) > 0.42:
                text_color = "white"
            ax.text(col, row, text, ha="center", va="center", fontsize=7.8, color=text_color)



def style_axis(ax, title: str) -> None:
    ax.set_title(title, pad=2.0)
    ax.set_xticks(range(len(METHODS)), [method.label for method in METHODS], rotation=0, ha="center")
    ax.set_yticks(range(len(SYSTEMS)), [SYSTEM_LABELS[system] for system in SYSTEMS])
    ax.tick_params(axis="x", length=0, pad=0.15)
    ax.tick_params(axis="y", length=0, pad=3.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def ratio_colorbar_enabled() -> bool:
    return os.environ.get("SHOW_RATIO_COLORBAR", "0").lower() in {"1", "true", "yes", "on"}


def plot(summary, output_path: Path, show_colorbar: bool | None = None) -> None:
    if show_colorbar is None:
        show_colorbar = ratio_colorbar_enabled()

    plt.rcParams.update({
        "figure.figsize": (4.45, 0.88) if show_colorbar else (3.45, 0.76),
        "figure.dpi": 180, "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 8.5 if show_colorbar else 7.6,
        "axes.titlesize": 8.5 if show_colorbar else 7.7,
        "xtick.labelsize": 7.2 if show_colorbar else 6.4,
        "ytick.labelsize": 8.0 if show_colorbar else 7.0,
    })
    time = matrix(summary, "time_ratio")
    cost = matrix(summary, "cost_ratio")
    log_time = np.log10(time)
    log_cost = np.log10(cost)

    fig = plt.figure()
    if show_colorbar:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.09)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        cax = fig.add_subplot(gs[0, 2])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.045)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        cax = None

    ratio_lim = math.log10(4.0)
    im0 = axes[0].imshow(log_time, aspect="auto", cmap="RdYlGn_r", vmin=-ratio_lim, vmax=ratio_lim)
    annotate_ratio(axes[0], time)
    style_axis(axes[0], "CT Ratio")

    im1 = axes[1].imshow(log_cost, aspect="auto", cmap="RdYlGn_r", vmin=-ratio_lim, vmax=ratio_lim)
    annotate_ratio(axes[1], cost)
    style_axis(axes[1], "PT Ratio")
    axes[1].tick_params(labelleft=False)

    if show_colorbar and cax is not None:
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_ticks([math.log10(0.25), math.log10(0.5), 0, math.log10(2.0), math.log10(4.0)])
        cbar.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
        cbar.ax.tick_params(labelsize=7, length=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if show_colorbar:
        fig.subplots_adjust(left=0.12, right=0.93, bottom=0.125, top=0.895)
    else:
        fig.subplots_adjust(left=0.160, right=0.985, bottom=0.145, top=0.855)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def write_csv(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "radius",
            "method",
            "time_ratio",
            "cost_ratio",
            "time_conditions",
            "cost_conditions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for method in METHODS:
                cell = summary[system][method.key]
                writer.writerow({
                    "system": system,
                    "radius": summary_radius(system, UCYCLE_RADIUS),
                    "method": method.label,
                    "time_ratio": f"{cell['time_ratio']:.6f}",
                    "cost_ratio": f"{cell['cost_ratio']:.6f}",
                    "time_conditions": cell["time_n"],
                    "cost_conditions": cell["cost_n"],
                })


def write_paired_csv(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "radius",
            "method",
            "time_ratio",
            "cost_ratio",
            "time_conditions",
            "cost_conditions",
            "time_shared_trials",
            "cost_shared_trials",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for method in METHODS:
                cell = summary[system][method.key]
                writer.writerow({
                    "system": system,
                    "radius": summary_radius(system, UCYCLE_RADIUS),
                    "method": method.label,
                    "time_ratio": f"{cell['time_ratio']:.6f}",
                    "cost_ratio": f"{cell['cost_ratio']:.6f}",
                    "time_conditions": cell["time_n"],
                    "cost_conditions": cell["cost_n"],
                    "time_shared_trials": cell["time_trials"],
                    "cost_shared_trials": cell["cost_trials"],
                })

def main() -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None

    geomean_output = Path("paper_plots/figures/summary/cbs_variant_metric_ratio_heatmap_mean.png")
    geomean_summary = collect_summary(root, statistic="mean", ucycle_results_root=ucycle_root)
    plot(geomean_summary, geomean_output)
    write_csv(geomean_summary, csv_output_path(geomean_output))
    print(f"Wrote {geomean_output}")
    print(f"Wrote {csv_output_path(geomean_output)}")

    median_output = Path("paper_plots/figures/summary/cbs_variant_metric_ratio_heatmap_median.png")
    median_summary = collect_summary(root, statistic="median", ucycle_results_root=ucycle_root)
    plot(median_summary, median_output)
    write_csv(median_summary, csv_output_path(median_output))
    print(f"Wrote {median_output}")
    print(f"Wrote {csv_output_path(median_output)}")

    paired_geomean_output = Path("paper_plots/figures/summary/cbs_variant_metric_ratio_heatmap_paired_mean.png")
    paired_geomean_summary = collect_paired_summary(root, statistic="mean", ucycle_results_root=ucycle_root)
    plot(paired_geomean_summary, paired_geomean_output)
    write_paired_csv(paired_geomean_summary, csv_output_path(paired_geomean_output))
    print(f"Wrote {paired_geomean_output}")
    print(f"Wrote {csv_output_path(paired_geomean_output)}")

    paired_median_output = Path("paper_plots/figures/summary/cbs_variant_metric_ratio_heatmap_paired_median.png")
    paired_median_summary = collect_paired_summary(root, statistic="median", ucycle_results_root=ucycle_root)
    plot(paired_median_summary, paired_median_output)
    write_paired_csv(paired_median_summary, csv_output_path(paired_median_output))
    print(f"Wrote {paired_median_output}")
    print(f"Wrote {csv_output_path(paired_median_output)}")

if __name__ == "__main__":
    main()
