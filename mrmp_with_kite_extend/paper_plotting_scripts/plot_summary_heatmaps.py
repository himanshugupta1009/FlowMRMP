#!/usr/bin/env python3
"""Compact summary heatmaps for main-paper MRMP results."""

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
class PlannerPair:
    baseline_csv: str
    kite_csv: str
    label: str

PARADIGMS = {
    "centralized": PlannerPair("CRRT_results.csv", "K-TI_EB_CRRT_results.csv", "cRRT\n+KiTE"),
    "prrt": PlannerPair("PRRT_results.csv", "KTI_EB_PRRT_results.csv", "pRRT\n+KiTE"),
    "kcbs": PlannerPair("RRT_KCBS_results.csv", "KTI_EB_RRT_KCBS_results.csv", "KCBS\n+KiTE"),
}
SYSTEMS = ("UCYCLE", "SOC", "QUAD")
SYSTEM_LABELS = {"SOC": "SOC", "UCYCLE": "UC", "QUAD": "DI"}
SYSTEM_ENVIRONMENTS = {
    "SOC": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "UCYCLE": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "QUAD": ("large_cluttered_3d_env", "swap_3d_env"),
}
METRIC_COLUMNS = {"time": "Computation Time (s)", "cost": "Total Path Costs"}
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")

def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def success_rate(csv_path: Path) -> tuple[float, int]:
    rows = read_rows(csv_path)
    if not rows:
        return math.nan, 0
    successes = sum(row["Success"].strip().lower() == "true" for row in rows)
    return successes / len(rows), len(rows)

def successful_stat(csv_path: Path, column: str, statistic: str) -> tuple[float | None, int]:
    values = []
    for row in read_rows(csv_path):
        if row["Success"].strip().lower() != "true":
            continue
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            values.append(value)
    if not values:
        return None, 0
    if statistic == "mean":
        return mean(values), len(values)
    if statistic == "median":
        return median(values), len(values)
    raise ValueError(f"Unknown statistic: {statistic}")

def paired_successful_ratio(
    baseline_path: Path,
    kite_path: Path,
    column: str,
    statistic: str,
) -> tuple[float | None, int]:
    baseline_rows = {row["Seed"]: row for row in read_rows(baseline_path)}
    kite_rows = {row["Seed"]: row for row in read_rows(kite_path)}
    baseline_values = []
    method_values = []
    for seed in sorted(set(baseline_rows) & set(kite_rows), key=int):
        baseline_row = baseline_rows[seed]
        method_row = kite_rows[seed]
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

def summarize_cell(root: Path, system: str, paradigm: str, statistic: str = "mean") -> dict[str, float | int]:
    pair = PARADIGMS[paradigm]
    success_deltas, time_ratios, cost_ratios = [], [], []
    success_n = time_n = cost_n = 0
    for environment in SYSTEM_ENVIRONMENTS[system]:
        for run_dir in matched_run_dirs(root, system, environment):
            baseline_path = run_dir / "csvs" / pair.baseline_csv
            kite_path = run_dir / "csvs" / pair.kite_csv
            if not baseline_path.exists() or not kite_path.exists():
                continue
            base_success, _ = success_rate(baseline_path)
            kite_success, _ = success_rate(kite_path)
            if math.isfinite(base_success) and math.isfinite(kite_success):
                success_deltas.append(100.0 * (kite_success - base_success))
                success_n += 1
            base_time, _ = successful_stat(baseline_path, METRIC_COLUMNS["time"], statistic)
            kite_time, _ = successful_stat(kite_path, METRIC_COLUMNS["time"], statistic)
            if base_time and kite_time:
                time_ratios.append(kite_time / base_time)
                time_n += 1
            base_cost, _ = successful_stat(baseline_path, METRIC_COLUMNS["cost"], statistic)
            kite_cost, _ = successful_stat(kite_path, METRIC_COLUMNS["cost"], statistic)
            if base_cost and kite_cost:
                cost_ratios.append(kite_cost / base_cost)
                cost_n += 1
    return {
        "success_delta": mean(success_deltas) if success_deltas else math.nan,
        "time_ratio": aggregate_ratio(time_ratios),
        "cost_ratio": aggregate_ratio(cost_ratios),
        "success_n": success_n,
        "time_n": time_n,
        "cost_n": cost_n,
    }

def summarize_cell_paired(root: Path, system: str, paradigm: str, statistic: str = "mean") -> dict[str, float | int]:
    pair = PARADIGMS[paradigm]
    time_ratios, cost_ratios = [], []
    time_n = cost_n = 0
    time_trials = cost_trials = 0
    for environment in SYSTEM_ENVIRONMENTS[system]:
        for run_dir in matched_run_dirs(root, system, environment):
            baseline_path = run_dir / "csvs" / pair.baseline_csv
            kite_path = run_dir / "csvs" / pair.kite_csv
            if not baseline_path.exists() or not kite_path.exists():
                continue
            time_ratio, shared_time = paired_successful_ratio(
                baseline_path, kite_path, METRIC_COLUMNS["time"], statistic
            )
            if time_ratio:
                time_ratios.append(time_ratio)
                time_n += 1
                time_trials += shared_time
            cost_ratio, shared_cost = paired_successful_ratio(
                baseline_path, kite_path, METRIC_COLUMNS["cost"], statistic
            )
            if cost_ratio:
                cost_ratios.append(cost_ratio)
                cost_n += 1
                cost_trials += shared_cost
    return {
        "time_ratio": aggregate_ratio(time_ratios),
        "cost_ratio": aggregate_ratio(cost_ratios),
        "time_n": time_n,
        "cost_n": cost_n,
        "time_trials": time_trials,
        "cost_trials": cost_trials,
    }

def build_summary(root: Path, statistic: str = "mean", ucycle_results_root: Path | None = None):
    return {
        system: {
            paradigm: summarize_cell(system_results_root(root, system, ucycle_results_root), system, paradigm, statistic)
            for paradigm in PARADIGMS
        }
        for system in SYSTEMS
    }

def build_paired_summary(root: Path, statistic: str = "mean", ucycle_results_root: Path | None = None):
    return {
        system: {
            paradigm: summarize_cell_paired(
                system_results_root(root, system, ucycle_results_root), system, paradigm, statistic
            )
            for paradigm in PARADIGMS
        }
        for system in SYSTEMS
    }

def matrix(summary, key: str) -> np.ndarray:
    return np.array([[summary[system][paradigm][key] for paradigm in PARADIGMS] for system in SYSTEMS], dtype=float)

def annotate_success(ax, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "--" if math.isnan(value) else f"{value:+.0f} pp"
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
    ax.set_xticks(range(len(PARADIGMS)), [PARADIGMS[key].label for key in PARADIGMS])
    ax.set_yticks(range(len(SYSTEMS)), [SYSTEM_LABELS[system] for system in SYSTEMS])
    ax.tick_params(axis="x", length=0, pad=0.15)
    ax.tick_params(axis="y", length=0, pad=3.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(PARADIGMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

def plot_summary(summary, output_path: Path) -> None:
    plt.rcParams.update({
        "figure.figsize": (6.2, 2.0), "figure.dpi": 180, "savefig.dpi": 400,
        "font.family": "serif", "font.size": 8.0, "axes.titlesize": 8.1,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.6,
    })
    success = matrix(summary, "success_delta")
    time = matrix(summary, "time_ratio")
    cost = matrix(summary, "cost_ratio")
    log_time = np.log10(time)
    log_cost = np.log10(cost)
    fig, axes = plt.subplots(1, 3)
    success_lim = max(10.0, np.nanmax(np.abs(success)))
    im0 = axes[0].imshow(success, cmap="RdBu", vmin=-success_lim, vmax=success_lim)
    annotate_success(axes[0], success)
    style_axis(axes[0], "Success gain")
    ratio_lim = math.log10(4.0)
    im1 = axes[1].imshow(log_time, aspect="auto", cmap="RdYlGn_r", vmin=-ratio_lim, vmax=ratio_lim)
    annotate_ratio(axes[1], time)
    style_axis(axes[1], "Computation Time (CT)")
    im2 = axes[2].imshow(log_cost, aspect="auto", cmap="RdYlGn_r", vmin=-ratio_lim, vmax=ratio_lim)
    annotate_ratio(axes[2], cost)
    style_axis(axes[2], "Total Path Time (PT)")
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.ax.tick_params(labelsize=7, length=2)
    cbar1 = fig.colorbar(im2, ax=axes[1:], fraction=0.035, pad=0.03)
    cbar1.set_ticks([math.log10(0.25), math.log10(0.5), 0, math.log10(2.0), math.log10(4.0)])
    cbar1.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    cbar1.ax.tick_params(labelsize=7, length=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1), pad=0.25, w_pad=0.85)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def ratio_colorbar_enabled() -> bool:
    return os.environ.get("SHOW_RATIO_COLORBAR", "0").lower() in {"1", "true", "yes", "on"}


def write_csv(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = ["system", "radius", "paradigm", "success_delta_pp", "time_ratio", "cost_ratio", "success_conditions", "time_conditions", "cost_conditions"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for paradigm in PARADIGMS:
                cell = summary[system][paradigm]
                writer.writerow({
                    "system": system,
                    "radius": summary_radius(system, UCYCLE_RADIUS),
                    "paradigm": paradigm,
                    "success_delta_pp": f"{cell['success_delta']:.6f}",
                    "time_ratio": f"{cell['time_ratio']:.6f}",
                    "cost_ratio": f"{cell['cost_ratio']:.6f}",
                    "success_conditions": cell["success_n"],
                    "time_conditions": cell["time_n"],
                    "cost_conditions": cell["cost_n"],
                })


def plot_metric_summary(summary, output_path: Path, show_colorbar: bool | None = None) -> None:
    selected_paradigms = tuple(PARADIGMS.keys())
    labels = [PARADIGMS[key].label for key in selected_paradigms]

    def selected_matrix(key: str) -> np.ndarray:
        return np.array([[summary[system][paradigm][key] for paradigm in selected_paradigms] for system in SYSTEMS], dtype=float)

    def style_selected_axis(ax, title: str) -> None:
        ax.set_title(title, pad=2.0)
        ax.set_xticks(range(len(selected_paradigms)), labels)
        ax.set_yticks(range(len(SYSTEMS)), [SYSTEM_LABELS[system] for system in SYSTEMS])
        ax.tick_params(axis="x", length=0, pad=0.15)
        ax.tick_params(axis="y", length=0, pad=3.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(selected_paradigms), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    if show_colorbar is None:
        show_colorbar = ratio_colorbar_enabled()

    plt.rcParams.update({
        "figure.figsize": (4.45, 1.12) if show_colorbar else (3.45, 0.98),
        "figure.dpi": 180, "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 8.0 if show_colorbar else 7.3,
        "axes.titlesize": 8.1 if show_colorbar else 7.4,
        "xtick.labelsize": 7.5 if show_colorbar else 6.5,
        "ytick.labelsize": 7.6 if show_colorbar else 6.8,
    })
    time = selected_matrix("time_ratio")
    cost = selected_matrix("cost_ratio")
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
    style_selected_axis(axes[0], "CT Ratio")

    im1 = axes[1].imshow(log_cost, aspect="auto", cmap="RdYlGn_r", vmin=-ratio_lim, vmax=ratio_lim)
    annotate_ratio(axes[1], cost)
    style_selected_axis(axes[1], "PT Ratio")
    axes[1].tick_params(labelleft=False)

    if show_colorbar and cax is not None:
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_ticks([math.log10(0.25), math.log10(0.5), 0, math.log10(2.0), math.log10(4.0)])
        cbar.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
        cbar.ax.tick_params(labelsize=7, length=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if show_colorbar:
        fig.subplots_adjust(left=0.12, right=0.93, bottom=0.085, top=0.910)
    else:
        fig.subplots_adjust(left=0.175, right=0.985, bottom=0.115, top=0.870)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_metric_csv(summary, output_path: Path) -> None:
    selected_paradigms = tuple(PARADIGMS.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "radius",
            "paradigm",
            "time_ratio",
            "cost_ratio",
            "time_conditions",
            "cost_conditions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system in SYSTEMS:
            for paradigm in selected_paradigms:
                cell = summary[system][paradigm]
                writer.writerow({
                    "system": system,
                    "radius": summary_radius(system, UCYCLE_RADIUS),
                    "paradigm": paradigm,
                    "time_ratio": f"{cell['time_ratio']:.6f}",
                    "cost_ratio": f"{cell['cost_ratio']:.6f}",
                    "time_conditions": cell["time_n"],
                    "cost_conditions": cell["cost_n"],
                })

def write_paired_metric_csv(summary, output_path: Path) -> None:
    selected_paradigms = tuple(PARADIGMS.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "radius",
            "paradigm",
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
            for paradigm in selected_paradigms:
                cell = summary[system][paradigm]
                writer.writerow({
                    "system": system,
                    "radius": summary_radius(system, UCYCLE_RADIUS),
                    "paradigm": paradigm,
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

    geomean_output = Path("paper_plots/figures/summary/metric_ratio_heatmap_mean.png")
    geomean_summary = build_summary(root, statistic="mean", ucycle_results_root=ucycle_root)
    plot_metric_summary(geomean_summary, geomean_output)
    write_metric_csv(geomean_summary, csv_output_path(geomean_output))
    print(f"Wrote {geomean_output}")
    print(f"Wrote {csv_output_path(geomean_output)}")

    median_output = Path("paper_plots/figures/summary/metric_ratio_heatmap_median.png")
    median_summary = build_summary(root, statistic="median", ucycle_results_root=ucycle_root)
    plot_metric_summary(median_summary, median_output)
    write_metric_csv(median_summary, csv_output_path(median_output))
    print(f"Wrote {median_output}")
    print(f"Wrote {csv_output_path(median_output)}")

    paired_geomean_output = Path("paper_plots/figures/summary/metric_ratio_heatmap_paired_mean.png")
    paired_geomean_summary = build_paired_summary(root, statistic="mean", ucycle_results_root=ucycle_root)
    plot_metric_summary(paired_geomean_summary, paired_geomean_output)
    write_paired_metric_csv(paired_geomean_summary, csv_output_path(paired_geomean_output))
    print(f"Wrote {paired_geomean_output}")
    print(f"Wrote {csv_output_path(paired_geomean_output)}")

    paired_median_output = Path("paper_plots/figures/summary/metric_ratio_heatmap_paired_median.png")
    paired_median_summary = build_paired_summary(root, statistic="median", ucycle_results_root=ucycle_root)
    plot_metric_summary(paired_median_summary, paired_median_output)
    write_paired_metric_csv(paired_median_summary, csv_output_path(paired_median_output))
    print(f"Wrote {paired_median_output}")
    print(f"Wrote {csv_output_path(paired_median_output)}")

if __name__ == "__main__":
    main()
