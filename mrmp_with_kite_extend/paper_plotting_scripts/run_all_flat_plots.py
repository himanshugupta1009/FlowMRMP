#!/usr/bin/env python3
"""Run the copied plotting scripts with flattened outputs."""

from __future__ import annotations

import subprocess
import sys
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
RAL_PAPER_DIR = ROOT / "paper_plots" / "figures" / "ral_paper"
RAL_PAPER_FIGURES = (
    ROOT / "paper_plots" / "figures" / "success_rate" / "success_rate_all_paradigms.png",
    ROOT / "paper_plots" / "figures" / "success_rate" / "success_rate_all_paradigms_compact.png",
    ROOT / "paper_plots" / "figures" / "summary" / "cbs_variant_metric_ratio_heatmap_median.png",
    ROOT / "paper_plots" / "figures" / "summary" / "metric_ratio_heatmap_median.png",
    ROOT / "paper_plots" / "figures" / "summary" / "metric_ratio_heatmap_median_stacked.png",
    ROOT / "paper_plots" / "figures" / "ablations" / "SOC_agents_15" / "soc_ablation_all_metrics.png",
)


def copy_ral_paper_figures() -> None:
    RAL_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for source in RAL_PAPER_FIGURES:
        if not source.exists():
            print(f"WARNING: missing RAL paper figure: {source.relative_to(ROOT)}", flush=True)
            continue
        target = RAL_PAPER_DIR / source.name
        shutil.copy2(source, target)
        print(f"Wrote {target.relative_to(ROOT)}", flush=True)


def run(args: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(args), flush=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    subprocess.run(args, cwd=ROOT, check=True, env=run_env)


def radius_env(radius: str) -> dict[str, str]:
    return {"UCYCLE_RADIUS": radius}


def radius_args(radius: str) -> list[str]:
    return []


def run_radius_plots(python: str, radius: str) -> None:
    env = radius_env(radius)
    extra_args = radius_args(radius)

    for paradigm in ("centralized", "prrt", "kcbs"):
        run([
            python,
            str(SCRIPTS / "plot_success_vs_agents.py"),
            "--paradigm",
            paradigm,
            "--all-systems",
            "--write-csv",
            *extra_args,
        ], env=env)
        for metric in ("computation_time", "total_path_cost"):
            run([
                python,
                str(SCRIPTS / "plot_metric_vs_agents.py"),
                "--paradigm",
                paradigm,
                "--metric",
                metric,
                "--write-csv",
                *extra_args,
            ], env=env)

    for metric in ("computation_time", "total_path_cost"):
        for paradigm in ("centralized", "prrt", "kcbs"):
            run([
                python,
                str(SCRIPTS / "plot_metric_ratios_vs_agents.py"),
                "--paradigm",
                paradigm,
                "--metric",
                metric,
                "--write-csv",
                *extra_args,
            ], env=env)

    run([python, str(SCRIPTS / "plot_success_combined_paradigms.py")], env=env)
    run([python, str(SCRIPTS / "plot_metric_combined_paradigms.py")], env=env)
    run([python, str(SCRIPTS / "plot_cbs_comparison.py")], env=env)
    run([python, str(SCRIPTS / "plot_summary_heatmaps.py")], env=env)
    run([python, str(SCRIPTS / "plot_summary_stacked.py")], env=env)
    run([python, str(SCRIPTS / "plot_success_summary_bars.py")], env=env)
    run([python, str(SCRIPTS / "plot_success_trends_by_agents.py")], env=env)
    run([
        python,
        str(SCRIPTS / "plot_success_small_multiples.py"),
        *extra_args,
    ], env=env)


def main() -> None:
    python = sys.executable

    run([python, str(SCRIPTS / "extract_dbcbs_results.py")])
    for radius in ("0.3",):
        run_radius_plots(python, radius)

    run([python, str(SCRIPTS / "plot_ablation_figures.py")])
    copy_ral_paper_figures()


if __name__ == "__main__":
    main()
