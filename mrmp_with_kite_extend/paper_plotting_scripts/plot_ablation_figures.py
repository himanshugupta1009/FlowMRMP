#!/usr/bin/env python3
"""SOC ablation figures for KiTE hyperparameter sensitivity."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from output_paths import csv_output_path


OUTPUT_ROOT = Path("paper_plots/figures/ablations")
ABLATION_RESULTS_ROOT = Path("paper_results/results_23July2026")


@dataclass(frozen=True)
class PlannerSpec:
    key: str
    label: str
    csv_name: str
    baseline_csv_name: str
    baseline_stats_title: str
    color: str
    marker: str


@dataclass(frozen=True)
class AblationRun:
    agent_count: int
    seed: int

    @property
    def label(self) -> str:
        return f"SOC_agents_{self.agent_count}"

    @property
    def ablation_root(self) -> Path:
        return ABLATION_RESULTS_ROOT / "ablations_3Aug2026" / "small_cluttered_env" / (
            f"SOC_a{self.agent_count}_tests100_seed{self.seed}_gr0.5"
        )

    @property
    def baseline_run_dir(self) -> Path:
        return ABLATION_RESULTS_ROOT / "small_cluttered_env" / (
            f"SOC_a{self.agent_count}_tests100_seed{self.seed}_gr0.5_kd0.1"
        )

    @property
    def output_root(self) -> Path:
        return OUTPUT_ROOT / self.label


@dataclass(frozen=True)
class AblationSpec:
    key: str
    title: str
    xlabel: str
    values: tuple[str, ...]
    tick_labels: tuple[str, ...]
    default_value: str


RUNS = (
    AblationRun(agent_count=15, seed=3000),
    AblationRun(agent_count=18, seed=3600),
)


PLANNERS = (
    PlannerSpec("CRRT", "cRRT+KiTE", "K-TI_EB_CRRT_results.csv", "CRRT_results.csv", "CRRT", "#0072B2", "s"),
    PlannerSpec("PRRT", "pRRT+KiTE", "KTI_EB_PRRT_results.csv", "PRRT_results.csv", "PRRT", "#D55E00", "o"),
    PlannerSpec("KCBS", "CBS+KiTE", "KTI_EB_RRT_KCBS_results.csv", "RRT_KCBS_results.csv", "KCBS using RRT", "#009E73", "D"),
)

ABLATIONS = (
    AblationSpec(
        key="kd_delta_radius",
        title="Radius",
        xlabel="δ",
        values=("radius_0p01", "radius_0p05", "radius_0p10", "radius_0p20", "radius_0p50"),
        tick_labels=("0.01", "0.05", "0.1", "0.2", "0.5"),
        default_value="radius_0p10",
    ),
    AblationSpec(
        key="kd_num_edges",
        title="Bundle Size",
        xlabel="Bundle Size",
        values=("edges_10000", "edges_30000", "edges_50000", "edges_75000", "edges_100000"),
        tick_labels=("10k", "30k", "50k", "75k", "100k"),
        default_value="edges_50000",
    ),
    AblationSpec(
        key="num_skip_edges",
        title="Candidate Attempts",
        xlabel="Candidate Attempts",
        values=("skip_1", "skip_5", "skip_10", "skip_20"),
        tick_labels=("1", "5", "10", "20"),
        default_value="skip_10",
    ),
    AblationSpec(
        key="edge_sorting",
        title="Sorting",
        xlabel="Edge Ordering",
        values=("unsorted", "sorted"),
        tick_labels=("Unsorted", "Sorted"),
        default_value="sorted",
    ),
)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def successful_mean(rows: list[dict[str, str]], column: str) -> float | None:
    values = []
    for row in rows:
        if not parse_bool(row.get("Success", "")):
            continue
        raw = row.get(column, "")
        if not raw:
            continue
        value = float(raw)
        if value > 0:
            values.append(value)
    return sum(values) / len(values) if values else None


def summarize_trial_csv(csv_path: Path) -> dict[str, float | int | None]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    successes = sum(1 for row in rows if parse_bool(row.get("Success", "")))
    return {
        "trials": total,
        "successes": successes,
        "success_rate": 100.0 * successes / total if total else None,
        "computation_time": successful_mean(rows, "Computation Time (s)"),
        "total_path_time": successful_mean(rows, "Total Path Costs"),
    }


def parse_stats_value(raw: str) -> float | None:
    raw = raw.strip()
    if raw in {"", "N/A", "--"}:
        return None
    return float(raw)


def summarize_stats_section(stats_path: Path, section_title: str) -> dict[str, float | int | None] | None:
    if not stats_path.exists():
        return None

    lines = stats_path.read_text().splitlines()
    section_header = f"{section_title}:"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == section_header:
            start = idx + 1
            break
    if start is None:
        return None

    values: dict[str, str] = {}
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith("	") and stripped.endswith(":"):
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip()

    trials = parse_int(values.get("Total rounds completed"))
    success_fraction = parse_stats_value(values.get("Percent Success", ""))
    successes = round(trials * success_fraction) if trials and success_fraction is not None else 0
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": 100.0 * success_fraction if success_fraction is not None else None,
        "computation_time": parse_stats_value(values.get("Average Time", "")),
        "total_path_time": parse_stats_value(values.get("Average Total Cost", "")),
    }


def zero_success_baseline(trials: int = 100) -> dict[str, float | int | None]:
    return {
        "trials": trials,
        "successes": 0,
        "success_rate": 0.0,
        "computation_time": None,
        "total_path_time": None,
    }


def load_baselines(run: AblationRun) -> dict[str, dict[str, float | int | None]]:
    baselines: dict[str, dict[str, float | int | None]] = {}
    stats_path = run.baseline_run_dir / "stats.txt"
    for planner in PLANNERS:
        csv_path = run.baseline_run_dir / "csvs" / planner.baseline_csv_name
        if csv_path.exists():
            baselines[planner.key] = summarize_trial_csv(csv_path)
            continue

        stats_summary = summarize_stats_section(stats_path, planner.baseline_stats_title)
        if stats_summary is not None:
            baselines[planner.key] = stats_summary
            continue

        if planner.key == "CRRT":
            baselines[planner.key] = zero_success_baseline()
    return baselines


def collect_rows(run: AblationRun) -> list[dict[str, str | float | int | None]]:
    rows: list[dict[str, str | float | int | None]] = []
    for ablation in ABLATIONS:
        for planner in PLANNERS:
            summary_path = run.ablation_root / ablation.key / planner.key / "summary.csv"
            summary_rows: dict[str, dict[str, str]] = {}
            if summary_path.exists():
                with summary_path.open(newline="") as f:
                    summary_rows = {row["ablation_value"]: row for row in csv.DictReader(f)}
            for value in ablation.values:
                tick_label = ablation.tick_labels[ablation.values.index(value)]
                source_row = summary_rows.get(value)
                if source_row is None:
                    rows.append(
                        {
                            "agent_count": run.agent_count,
                            "ablation": ablation.key,
                            "ablation_label": ablation.title,
                            "value": value,
                            "tick_label": tick_label,
                            "is_default": value == ablation.default_value,
                            "planner": planner.key,
                            "planner_label": planner.label,
                            "trials": 0,
                            "successes": 0,
                            "success_rate": None,
                            "computation_time": None,
                            "total_path_time": None,
                            "csv": "",
                        }
                    )
                    continue

                success_rate = parse_float(source_row.get("success_rate"))
                if success_rate is not None:
                    success_rate *= 100.0
                rows.append(
                    {
                        "agent_count": run.agent_count,
                        "ablation": ablation.key,
                        "ablation_label": ablation.title,
                        "value": value,
                        "tick_label": tick_label,
                        "is_default": value == ablation.default_value,
                        "planner": planner.key,
                        "planner_label": planner.label,
                        "trials": parse_int(source_row.get("total_runs")),
                        "successes": parse_int(source_row.get("successes")),
                        "success_rate": success_rate,
                        "computation_time": parse_float(source_row.get("mean_computation_time_s")),
                        "total_path_time": parse_float(source_row.get("mean_total_path_cost")),
                        "csv": str(summary_path),
                    }
                )
    return rows


def write_summary_csv(rows: list[dict[str, str | float | int | None]], output_root: Path) -> Path:
    output_path = csv_output_path(output_root / "soc_ablation_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent_count",
        "ablation",
        "ablation_label",
        "value",
        "tick_label",
        "is_default",
        "planner",
        "planner_label",
        "trials",
        "successes",
        "success_rate",
        "computation_time",
        "total_path_time",
        "baseline_success_rate",
        "baseline_computation_time",
        "baseline_total_path_time",
        "csv",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def row_lookup(rows: list[dict[str, str | float | int | None]]) -> dict[tuple[str, str, str], dict[str, str | float | int | None]]:
    return {(str(row["ablation"]), str(row["planner"]), str(row["value"])): row for row in rows}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.size": 9.2,
            "axes.labelsize": 9.6,
            "axes.titlesize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "lines.linewidth": 1.35,
        }
    )


def make_legend(fig: plt.Figure) -> None:
    planner_label = Line2D([0], [0], color="none", linewidth=0, label="Planner:")
    planner_handles = [
        Line2D(
            [0],
            [0],
            color=planner.color,
            linestyle="-",
            linewidth=1.6,
            label=planner.label.replace("+KiTE", ""),
        )
        for planner in PLANNERS
    ]
    variant_label = Line2D([0], [0], color="none", linewidth=0, label="Variant:")
    variant_handles = [
        Line2D(
            [0], [0], color="#333333", linestyle="-", marker="s",
            markerfacecolor="none", markeredgewidth=1.05,
            markersize=5.3, linewidth=1.45, label="KiTE"
        ),
        Line2D(
            [0], [0], color="#333333", linestyle="--", marker="o",
            markerfacecolor="none", markeredgewidth=1.05,
            markersize=5.0, linewidth=1.45, label="Rand"
        ),
    ]
    handles = [variant_label] + variant_handles + [planner_label] + planner_handles
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        columnspacing=0.72,
        handlelength=2.55,
        handletextpad=0.34,
    )



def plot_rand_reference(
    ax: plt.Axes,
    x_values: list[int],
    y_value: float,
    color: str,
    markersize: float,
) -> None:
    ax.plot(
        x_values,
        [y_value] * len(x_values),
        color=color,
        linestyle="--",
        linewidth=1.35,
        alpha=0.62,
        zorder=0,
    )
    marker_x_values = [x - 0.045 for x in x_values]
    ax.plot(
        marker_x_values,
        [y_value] * len(x_values),
        color=color,
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgewidth=0.9,
        markersize=markersize,
        alpha=0.72,
        zorder=1,
    )

def add_success_baselines(
    ax: plt.Axes,
    x_values: list[int],
    baselines: dict[str, dict[str, float | int | None]],
    markersize: float,
) -> None:
    for planner in PLANNERS:
        baseline = baselines.get(planner.key, {}).get("success_rate")
        if isinstance(baseline, float):
            plot_rand_reference(ax, x_values, baseline, planner.color, markersize)

def add_ratio_baselines(
    ax: plt.Axes,
    ablation: AblationSpec,
    metric: str,
    lookup: dict[tuple[str, str, str], dict[str, str | float | int | None]],
    baselines: dict[str, dict[str, float | int | None]],
    all_ratios: list[float],
) -> None:
    for planner in PLANNERS:
        default_value = lookup[(ablation.key, planner.key, ablation.default_value)][metric]
        baseline_value = baselines.get(planner.key, {}).get(metric)
        if isinstance(baseline_value, float) and isinstance(default_value, float) and default_value > 0:
            ratio = baseline_value / default_value
            all_ratios.append(ratio)
            ax.axhline(
                ratio,
                color=planner.color,
                linestyle=":",
                linewidth=1.35,
                alpha=0.62,
                zorder=0,
            )


def plot_success(
    rows: list[dict[str, str | float | int | None]],
    baselines: dict[str, dict[str, float | int | None]],
    output_root: Path,
) -> Path:
    lookup = row_lookup(rows)
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.45, 2.08), sharey=True)

    for ax, ablation in zip(axes, ABLATIONS, strict=True):
        x_values = list(range(len(ablation.values)))
        add_success_baselines(ax, x_values, baselines, markersize=4.7)
        for planner in PLANNERS:
            y_values = [lookup[(ablation.key, planner.key, value)]["success_rate"] for value in ablation.values]
            ax.plot(
                x_values,
                y_values,
                color=planner.color,
                linestyle="-",
                marker="s",
                markerfacecolor="none",
                markeredgewidth=1.05,
                markersize=5.2,
                linewidth=1.35,
            )
        default_x = ablation.values.index(ablation.default_value)
        ax.axvline(default_x, color="#222222", linewidth=0.7, alpha=0.18, zorder=0)
        ax.set_title(ablation.title, fontweight="bold", pad=3)
        ax.set_xlabel(ablation.xlabel, fontweight="bold", labelpad=4)
        ax.set_xticks(x_values, ablation.tick_labels)
        if ablation.key == "edge_sorting":
            ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.72)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Success Rate (%)", fontweight="bold")
    make_legend(fig)

    output_path = output_root / "soc_ablation_success.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=0.78, pad=0.2)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_metric(
    rows: list[dict[str, str | float | int | None]],
    baselines: dict[str, dict[str, float | int | None]],
    metric: str,
    ylabel: str,
    output_name: str,
    output_root: Path,
    log_y: bool = False,
) -> Path:
    lookup = row_lookup(rows)
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.45, 2.08), sharey=False)

    all_values: list[float] = []
    for ax, ablation in zip(axes, ABLATIONS, strict=True):
        x_values = list(range(len(ablation.values)))
        for planner in PLANNERS:
            baseline_value = baselines.get(planner.key, {}).get(metric)
            if isinstance(baseline_value, float) and baseline_value > 0:
                all_values.append(baseline_value)
                plot_rand_reference(ax, x_values, baseline_value, planner.color, markersize=4.7)

            y_values: list[float | None] = []
            for value in ablation.values:
                metric_value = lookup[(ablation.key, planner.key, value)][metric]
                if isinstance(metric_value, float) and metric_value > 0:
                    y_values.append(metric_value)
                    all_values.append(metric_value)
                else:
                    y_values.append(None)
            ax.plot(
                x_values,
                y_values,
                color=planner.color,
                linestyle="-",
                marker="s",
                markerfacecolor="none",
                markeredgewidth=1.05,
                markersize=5.2,
                linewidth=1.35,
            )
        default_x = ablation.values.index(ablation.default_value)
        ax.axvline(default_x, color="#222222", linewidth=0.7, alpha=0.18, zorder=0)
        ax.set_title(ablation.title, fontweight="bold", pad=3)
        ax.set_xlabel(ablation.xlabel, fontweight="bold", labelpad=4)
        ax.set_xticks(x_values, ablation.tick_labels)
        if ablation.key == "edge_sorting":
            ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.72)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if log_y:
            ax.set_yscale("log")

    if all_values:
        lower = min(all_values) * (0.75 if log_y else 0.90)
        upper = max(all_values) * (1.35 if log_y else 1.10)
        for ax in axes:
            ax.set_ylim(lower, upper)
    axes[0].set_ylabel(ylabel, fontweight="bold")
    make_legend(fig)

    output_path = output_root / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=0.78, pad=0.2)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def style_axis(ax: plt.Axes, ablation: AblationSpec, show_xlabel: bool = True) -> None:
    x_values = list(range(len(ablation.values)))
    default_x = ablation.values.index(ablation.default_value)
    ax.axvline(default_x, color="#222222", linewidth=0.7, alpha=0.18, zorder=0)
    if show_xlabel:
        ax.set_xlabel(ablation.xlabel, fontweight="bold", labelpad=4)
    ax.set_xticks(x_values, ablation.tick_labels)
    if ablation.key == "edge_sorting":
        ax.set_xlim(-0.35, 1.35)
    if ablation.key == "kd_delta_radius":
        for label in ax.get_xticklabels():
            label.set_fontsize(8.0)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined(
    rows: list[dict[str, str | float | int | None]],
    baselines: dict[str, dict[str, float | int | None]],
    output_root: Path,
) -> Path:
    lookup = row_lookup(rows)
    setup_style()
    fig, axes = plt.subplots(3, 4, figsize=(7.55, 4.75), sharex="col", gridspec_kw={"height_ratios": [0.72, 1.0, 1.0]})

    metric_rows = (
        ("success_rate", "Success (%)", False),
        ("computation_time", "CT (in s)", True),
        ("total_path_time", "PT (in s)", False),
    )

    for col, ablation in enumerate(ABLATIONS):
        x_values = list(range(len(ablation.values)))
        for row_idx, (metric, ylabel, log_y) in enumerate(metric_rows):
            ax = axes[row_idx, col]
            all_values: list[float] = []
            if metric == "success_rate":
                add_success_baselines(ax, x_values, baselines, markersize=4.1)
                for planner in PLANNERS:
                    baseline = baselines.get(planner.key, {}).get(metric)
                    if isinstance(baseline, float):
                        all_values.append(baseline)
            else:
                for planner in PLANNERS:
                    baseline_value = baselines.get(planner.key, {}).get(metric)
                    if isinstance(baseline_value, float) and baseline_value > 0:
                        all_values.append(baseline_value)
                        plot_rand_reference(ax, x_values, baseline_value, planner.color, markersize=4.1)

            for planner in PLANNERS:
                y_values: list[float | None] = []
                for value in ablation.values:
                    metric_value = lookup[(ablation.key, planner.key, value)][metric]
                    if isinstance(metric_value, float) and (metric == "success_rate" or metric_value > 0):
                        y_values.append(metric_value)
                        all_values.append(metric_value)
                    else:
                        y_values.append(None)
                ax.plot(
                    x_values,
                    y_values,
                    color=planner.color,
                    linestyle="-",
                    marker="s",
                    markerfacecolor="none",
                    markeredgewidth=1.05,
                    markersize=4.6,
                    linewidth=1.18,
                )

            style_axis(ax, ablation, show_xlabel=row_idx == 2)
            if log_y:
                ax.set_yscale("log")
            if metric == "success_rate":
                ax.set_ylim(-5, 108)
                ax.set_yticks([0, 50, 100])
            elif all_values:
                lower = min(all_values) * (0.75 if log_y else 0.90)
                upper = max(all_values) * (1.35 if log_y else 1.10)
                ax.set_ylim(lower, upper)
            ax.set_ylabel("")

    make_legend(fig)
    fig.tight_layout(rect=(0.060, 0.025, 1.0, 0.925), w_pad=0.52, h_pad=0.32, pad=0.12)
    fig.canvas.draw()
    for row_idx, (_, ylabel, _) in enumerate(metric_rows):
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            0.024,
            0.5 * (bbox.y0 + bbox.y1),
            ylabel,
            rotation="vertical",
            va="center",
            ha="center",
            fontweight="bold",
            fontsize=plt.rcParams["axes.labelsize"] + 1.1,
        )

    output_path = output_root / "soc_ablation_all_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def main() -> None:
    for run in RUNS:
        rows = collect_rows(run)
        baselines = load_baselines(run)
        for row in rows:
            baseline = baselines.get(str(row["planner"]), {})
            row["baseline_success_rate"] = baseline.get("success_rate")
            row["baseline_computation_time"] = baseline.get("computation_time")
            row["baseline_total_path_time"] = baseline.get("total_path_time")
        summary_csv = write_summary_csv(rows, run.output_root)
        outputs = [
            plot_success(rows, baselines, run.output_root),
            plot_metric(
                rows,
                baselines,
                "computation_time",
                "Computation Time (CT, in s)",
                "soc_ablation_computation_time.png",
                run.output_root,
                log_y=True,
            ),
            plot_metric(
                rows,
                baselines,
                "total_path_time",
                "Total Path Time (PT, in s)",
                "soc_ablation_total_path_time.png",
                run.output_root,
            ),
            plot_combined(rows, baselines, run.output_root),
        ]
        print(f"Wrote {summary_csv}")
        for output in outputs:
            print(f"Wrote {output}")
        missing = [planner.label for planner in PLANNERS if planner.key not in baselines]
        if missing:
            print(f"{run.label}: Missing Rand baseline CSV for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
