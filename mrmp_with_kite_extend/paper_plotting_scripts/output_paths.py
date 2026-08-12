"""Shared output path helpers for the paper plot scripts."""

from __future__ import annotations

import os
import re
import json
from functools import lru_cache
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("paper_results/results_9June2026")
DEFAULT_UCYCLE_RESULTS_ROOT = DEFAULT_RESULTS_ROOT

QUAD_ENVIRONMENT_SUBDIRS = {
    "large_cluttered_3d_env": "MP_6000_free_time",
    "swap_3d_env": "MP_6000_fixed_time",
}

RUN_DIR_RE = re.compile(r"^[A-Z]+_a\d+_tests\d+_seed\d+_gr[\d.]+_kd[\d.]+$")


def csv_output_path(path: Path) -> Path:
    """Store generated CSVs under paper_plots/csvs, mirroring figures."""
    path = Path(path).with_suffix(".csv")
    parts = list(path.parts)
    for idx in range(len(parts) - 1):
        if parts[idx] == "paper_plots" and parts[idx + 1] == "figures":
            parts[idx + 1] = "csvs"
            return Path(*parts)
    return path


def ucycle_radius_for_source(source_path: str) -> str:
    """Infer the UCYCLE radius represented by a source path."""
    if "dbcbs_normalized.csv" in source_path:
        return os.environ.get("UCYCLE_RADIUS", "0.3")
    if "unicycle_results_0p4" in source_path or "radius_0p4" in source_path:
        return "0.4"
    if "results_9June2026" in source_path or "new_final_results" in source_path or "radius_0p3" in source_path or "unicycle_AR_0p3" in source_path:
        return "0.3"
    return ""


def run_dir_for_source(source_path: str) -> Path | None:
    """Return the run directory for a per-run CSV source path."""
    path = Path(source_path)
    if path.name == "dbcbs_normalized.csv":
        return None
    if path.parent.name == "csvs":
        return path.parent.parent
    for parent in (path, *path.parents):
        if RUN_DIR_RE.match(parent.name):
            return parent
    return None


@lru_cache(maxsize=None)
def agent_radius_for_run_dir(run_dir: str) -> str:
    """Read the robot/agent radius from a run manifest."""
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.exists():
        return ""

    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return ""

    builders = data.get("agent_builders") or []
    if not builders:
        return ""

    radius = (builders[0].get("params") or {}).get("radius")
    if radius is None:
        return ""
    return f"{float(radius):.8g}"


def row_radius(system: str, source_path: str) -> str:
    """Return the robot/agent radius represented by a plotted/generated row."""
    run_dir = run_dir_for_source(source_path)
    if run_dir is not None:
        radius = agent_radius_for_run_dir(str(run_dir))
        if radius:
            return radius

    if system == "UCYCLE":
        return ucycle_radius_for_source(source_path)
    if system == "QUAD" and "dbcbs_normalized.csv" in source_path:
        return "0.3"
    return ""


def system_results_root(default_root: Path, system: str, ucycle_root: Path | None = None) -> Path:
    """Use an alternate results root for UCYCLE radius sweeps."""
    if system == "UCYCLE" and ucycle_root is not None:
        return ucycle_root
    return default_root


def environment_results_dir(root: Path, system: str, environment: str) -> Path:
    """Return the directory containing run folders for a system/environment."""
    env_dir = Path(root) / environment
    if system == "QUAD":
        subdir = QUAD_ENVIRONMENT_SUBDIRS.get(environment)
        if subdir:
            return env_dir / subdir
    return env_dir


def summary_radius(system: str, ucycle_radius: str | None = None) -> str:
    """Return the fixed agent radius for aggregate rows without per-run paths."""
    if system == "UCYCLE":
        return ucycle_radius or os.environ.get("UCYCLE_RADIUS", "0.3")
    if system == "SOC":
        return "0.3"
    if system == "QUAD":
        return "0.3"
    return ""


def radius_token(radius: str) -> str:
    """Return a filesystem-friendly UCYCLE radius token."""
    return f"ucycle_r{radius.replace('.', 'p')}"


def with_radius_token(path: Path, radius: str) -> Path:
    """Append the UCYCLE radius token before a file suffix."""
    path = Path(path)
    token = radius_token(radius)
    if path.stem.endswith(token):
        return path
    return path.with_name(f"{path.stem}_{token}{path.suffix}")
