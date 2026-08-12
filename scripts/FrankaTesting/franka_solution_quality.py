"""Shared solution-quality metrics for Franka planning benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def as_bool(value: Any) -> bool:
    """Accept native booleans and CSV boolean strings."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def path_motion_time_from_states(states: np.ndarray, integration_dt: float) -> float:
    """Return physical execution time represented by a dense state path.

    A dense path contains its initial state followed by one state for every
    integration interval.  This is trajectory duration, not planner wall time.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("states must be a non-empty 2D array")
    if not np.isfinite(integration_dt) or integration_dt <= 0.0:
        raise ValueError("integration_dt must be finite and positive")
    return float((states.shape[0] - 1) * integration_dt)


def finite_stats(values: Iterable[float]) -> dict[str, float] | None:
    """Summarize finite values, returning ``None`` for an empty set."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p95": float(np.percentile(array, 95)),
    }


def solution_quality_summary(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """Summarize successful path duration and feasible-witness ratios."""
    successful_rows = [row for row in rows if as_bool(row["success"])]
    path_times = [float(row["path_motion_time_seconds"]) for row in successful_rows]
    witness_ratios = []
    witness_deltas = []
    for row in successful_rows:
        witness = float(row.get("witness_duration_seconds", np.nan))
        path_time = float(row["path_motion_time_seconds"])
        if np.isfinite(witness) and witness > 0.0 and np.isfinite(path_time):
            witness_ratios.append(path_time / witness)
            witness_deltas.append(path_time - witness)
    return {
        "definition": (
            "physical trajectory time from the start state through the first "
            "goal-reaching integration waypoint; planner wall time is excluded"
        ),
        "units": "seconds",
        "lower_is_better": True,
        "successful_paths": finite_stats(path_times),
        "path_time_over_feasible_witness": finite_stats(witness_ratios),
        "path_time_minus_feasible_witness_seconds": finite_stats(witness_deltas),
    }


def both_success_path_quality(
    first_rows: Iterable[Mapping[str, Any]],
    second_rows: Iterable[Mapping[str, Any]],
    *,
    first_name: str,
    second_name: str,
) -> dict[str, object]:
    """Compare path time only on paired trials where both planners succeeded."""
    pairs = [
        (first, second)
        for first, second in zip(first_rows, second_rows)
        if as_bool(first["success"]) and as_bool(second["success"])
    ]
    first_times = [float(first["path_motion_time_seconds"]) for first, _ in pairs]
    second_times = [float(second["path_motion_time_seconds"]) for _, second in pairs]
    differences = [second - first for first, second in zip(first_times, second_times)]
    ratios = [
        second / first
        for first, second in zip(first_times, second_times)
        if first > 0.0
    ]
    return {
        "paired_both_success_count": len(pairs),
        first_name: finite_stats(first_times),
        second_name: finite_stats(second_times),
        f"{second_name}_minus_{first_name}_seconds": finite_stats(differences),
        f"{second_name}_over_{first_name}": finite_stats(ratios),
    }
