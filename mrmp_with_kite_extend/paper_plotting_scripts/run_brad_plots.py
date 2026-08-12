#!/usr/bin/env python3
"""Generate the paper plot suite from Brad's 2026-07-15 results.

The plotting scripts use fixed workspace-relative input and output paths.  This
runner builds a temporary compatibility workspace, runs the existing plotting
pipeline there, and copies only generated artifacts into ``brad_plots``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
BRAD_RESULTS = ROOT / "paper_results" / "brad_run_2026-07-15"
OUTPUT = ROOT / "brad_plots"


def run(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def build_compatibility_results(staging_root: Path) -> Path:
    """Expose Brad's tree using the legacy paths expected by the plotters."""
    results = staging_root / "paper_results" / "results_9June2026"
    results.mkdir(parents=True)

    for source in BRAD_RESULTS.iterdir():
        if source.name in {"ablations", "_run_logs", "comparison_vs_paper"}:
            continue
        (results / source.name).symlink_to(source, target_is_directory=source.is_dir())

    # The current ablation plotter predates Brad's grouped ablation layout.
    # Brad ran the full suite at 15 agents, so map that group into its expected
    # legacy location without altering the original result directory.
    legacy = results / "ablations" / "small_cluttered_env"
    legacy.mkdir(parents=True)
    (legacy / "SOC_a15_tests100_seed3000_gr0.5").symlink_to(
        BRAD_RESULTS / "ablations" / "small_cluttered_env" / "SOC",
        target_is_directory=True,
    )
    return results


def copy_generated(staging_root: Path) -> None:
    generated = staging_root / "paper_plots"
    for name in ("figures", "csvs", "data"):
        source = generated / name
        if source.exists():
            shutil.copytree(source, OUTPUT / name, dirs_exist_ok=True)

    # The 18- and 20-agent ablation panels contain no Brad ablation data.
    for name in ("SOC_agents_18", "SOC_agents_20"):
        for parent in (OUTPUT / "figures" / "ablations", OUTPUT / "csvs" / "ablations"):
            path = parent / name
            if path.exists():
                shutil.rmtree(path)


def generate_all(staging_root: Path, env: dict[str, str]) -> None:
    """Run the plot scripts directly so they inherit the isolated cwd."""
    python = sys.executable
    partial = str(SCRIPTS / "run_partial_plot.py")
    run([python, str(SCRIPTS / "extract_dbcbs_results.py")], staging_root, env)

    for paradigm in ("centralized", "prrt", "kcbs"):
        run(
            [python, partial, "plot_success_vs_agents.py", "--paradigm", paradigm,
             "--all-systems", "--write-csv"],
            staging_root,
            env,
        )
        for metric in ("computation_time", "total_path_cost"):
            run(
                [python, partial, "plot_metric_vs_agents.py", "--paradigm", paradigm,
                 "--metric", metric, "--write-csv"],
                staging_root,
                env,
            )

    for metric in ("computation_time", "total_path_cost"):
        for paradigm in ("centralized", "prrt", "kcbs"):
            run(
                [python, partial, "plot_metric_ratios_vs_agents.py", "--paradigm", paradigm,
                 "--metric", metric, "--write-csv"],
                staging_root,
                env,
            )

    for script in (
        "plot_success_combined_paradigms.py",
        "plot_metric_combined_paradigms.py",
        "plot_cbs_comparison.py",
        "plot_summary_heatmaps.py",
        "plot_summary_stacked.py",
        "plot_success_summary_bars.py",
        "plot_success_trends_by_agents.py",
        "plot_success_small_multiples.py",
        "plot_ablation_figures.py",
    ):
        run([python, partial, script], staging_root, env)


def main() -> None:
    if not BRAD_RESULTS.is_dir():
        raise SystemExit(f"Brad result directory does not exist: {BRAD_RESULTS}")

    with tempfile.TemporaryDirectory(prefix="brad-plots-") as temporary:
        staging_root = Path(temporary)
        build_compatibility_results(staging_root)
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(staging_root / ".matplotlib")
        env["UCYCLE_RADIUS"] = "0.3"
        generate_all(staging_root, env)
        copy_generated(staging_root)

    print(f"Brad plots written to {OUTPUT}")


if __name__ == "__main__":
    main()
