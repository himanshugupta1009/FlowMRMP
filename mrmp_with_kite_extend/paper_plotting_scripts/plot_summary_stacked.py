#!/usr/bin/env python3
"""Stacked summary heatmaps for the main paper."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import plot_cbs_comparison as cbs_summary
import plot_summary_heatmaps as kite_summary
from output_paths import DEFAULT_RESULTS_ROOT


OUTPUT_PATH = Path("paper_plots/figures/summary/metric_ratio_heatmap_median_stacked.png")
RAL_OUTPUT_PATH = Path("paper_plots/figures/ral_paper/metric_ratio_heatmap_median_stacked.png")


def ratio_text_color(value: float) -> str:
    if math.isfinite(value) and value > 0 and abs(math.log10(value)) > 0.42:
        return "white"
    return "#111111"


def annotate_ratio(ax: plt.Axes, values: np.ndarray, fontsize: float = 7.7) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "--" if math.isnan(value) else f"{value:.3f}x"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=ratio_text_color(value),
            )


def style_heatmap_axis(
    ax: plt.Axes,
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str | None = None,
    x_labelsize: float = 6.1,
    y_labelsize: float = 5.4,
) -> None:
    if title:
        ax.set_title(title, pad=3.0, fontsize=7.5)
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.tick_params(axis="x", length=0, pad=1.8)
    ax.tick_params(axis="y", length=0, pad=3.2)
    for label in ax.get_xticklabels():
        label.set_fontsize(x_labelsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(y_labelsize)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def add_header_rule(fig: plt.Figure, ax: plt.Axes) -> None:
    bbox = ax.get_position()
    y = bbox.y1 + 0.006
    fig.add_artist(
        plt.Line2D(
            [bbox.x0 + 0.16 * bbox.width, bbox.x1 - 0.16 * bbox.width],
            [y, y],
            transform=fig.transFigure,
            color="#222222",
            linewidth=0.35,
            alpha=0.42,
        )
    )


def matrix_from_kite(summary, key: str) -> np.ndarray:
    return np.array(
        [[summary[system][paradigm][key] for paradigm in kite_summary.PARADIGMS] for system in kite_summary.SYSTEMS],
        dtype=float,
    )


def matrix_from_cbs(summary, key: str) -> np.ndarray:
    return np.array(
        [
            [summary[system][method.key][key] for method in cbs_variant_methods()]
            for system in cbs_summary.SYSTEMS
        ],
        dtype=float,
    )


def cbs_variant_methods() -> tuple[cbs_summary.MethodSpec, ...]:
    return tuple(method for method in cbs_summary.METHODS if method.key != "kite")


def one_line(label: str) -> str:
    return label.replace("\n", "")


def plot_stacked(output_path: Path = OUTPUT_PATH) -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None
    kite = kite_summary.build_summary(root, statistic="median", ucycle_results_root=ucycle_root)
    cbs = cbs_summary.collect_summary(root, statistic="median", ucycle_results_root=ucycle_root)

    kite_time = matrix_from_kite(kite, "time_ratio")
    kite_cost = matrix_from_kite(kite, "cost_ratio")
    cbs_time = matrix_from_cbs(cbs, "time_ratio")
    cbs_cost = matrix_from_cbs(cbs, "cost_ratio")

    ratio_lim = math.log10(4.0)
    cmap = "RdYlGn_r"

    plt.rcParams.update(
        {
            "figure.figsize": (3.55, 1.76),
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 7.2,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
        }
    )

    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[3, 2],
        width_ratios=[1, 1],
        hspace=0.30,
        wspace=0.045,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )

    panels = (
        (axes[0, 0], kite_time, [one_line(kite_summary.PARADIGMS[key].label) for key in kite_summary.PARADIGMS],
         [kite_summary.SYSTEM_LABELS[system] for system in kite_summary.SYSTEMS], "CT compared to baseline", 5.4),
        (axes[0, 1], kite_cost, [one_line(kite_summary.PARADIGMS[key].label) for key in kite_summary.PARADIGMS],
         [kite_summary.SYSTEM_LABELS[system] for system in kite_summary.SYSTEMS], "PT compared to baseline", 5.4),
        (axes[1, 0], cbs_time, [one_line(method.label) for method in cbs_variant_methods()],
         [cbs_summary.SYSTEM_LABELS[system] for system in cbs_summary.SYSTEMS], None, 5.4),
        (axes[1, 1], cbs_cost, [one_line(method.label) for method in cbs_variant_methods()],
         [cbs_summary.SYSTEM_LABELS[system] for system in cbs_summary.SYSTEMS], None, 5.4),
    )

    for ax, values, x_labels, y_labels, title, x_labelsize in panels:
        ax.imshow(np.log10(values), aspect="auto", cmap=cmap, vmin=-ratio_lim, vmax=ratio_lim)
        annotate_ratio(ax, values)
        style_heatmap_axis(ax, x_labels, y_labels, title=title, x_labelsize=x_labelsize, y_labelsize=5.4)

    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.075, top=0.900)
    fig.canvas.draw()
    for ax in axes.flat:
        bbox = ax.get_position()
        fig.add_artist(
            plt.Line2D(
                [bbox.x1, bbox.x1],
                [bbox.y0, bbox.y1],
                transform=fig.transFigure,
                color="white",
                linewidth=1.8,
                zorder=50,
            )
        )
    renderer = fig.canvas.get_renderer()
    upper_label_bottom = min(
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y0
        for ax in axes[0]
        for label in ax.get_xticklabels()
        if label.get_visible()
    )
    lower_axes_top = max(ax.get_position().y1 for ax in axes[1])
    separator_y = 0.5 * (upper_label_bottom + lower_axes_top)
    separator_right = max(ax.get_position().x1 for ax in axes[:, 1])
    separator = plt.Line2D(
        [0.135, separator_right],
        [separator_y, separator_y],
        transform=fig.transFigure,
        color="#777777",
        linewidth=0.35,
        alpha=0.34,
    )
    fig.add_artist(separator)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    plot_stacked(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    RAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_stacked(RAL_OUTPUT_PATH)
    print(f"Wrote {RAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
