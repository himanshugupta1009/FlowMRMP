#!/usr/bin/env python3
"""Extract dbCBS raw trial results into a normalized CSV for plotting."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from output_paths import DEFAULT_RESULTS_ROOT


DBCBS_ROOT = DEFAULT_RESULTS_ROOT / "dbcbs_cpp_results"
UCYCLE_SOURCE = DBCBS_ROOT / "unicycle_AR_0p3_MP_10000_results" / "table.tex"
QUAD_SOURCE = DBCBS_ROOT / "quad_AR_0p3_MP_6000_results" / "table.tex"
OUTPUT = Path("paper_plots/data/dbcbs_normalized.csv")
MAX_PLANNING_TIME = 300.0

UCYCLE_ENVIRONMENTS = {
    "cluttered": "small_cluttered_env",
    "corridor": "narrow_corridor_env",
    "large_ucycle": "large_cluttered_env",
    "swap_ucycle": "swap_env",
}

QUAD_ENVIRONMENTS = {
    "3d": "large_cluttered_3d_env",
    "3d_empty": "swap_3d_env",
}

TABLE_ROW_RE = re.compile(
    r"my\\_examples/(?P<env>[^/]+)/dbcbs\\_env\\_(?P<agents>\d+)\\_agents\\_seed\\_(?P<seed>\d+)"
)


@dataclass(frozen=True)
class Condition:
    system: str
    radius: str
    env_key: str
    environment: str
    agents: int
    seed: int
    source: Path
    tex_success: float | None
    tex_time: float | None
    tex_cost: float | None

    @property
    def raw_dir(self) -> Path:
        return self.source.parent / "my_examples" / self.env_key / f"dbcbs_env_{self.agents}_agents_seed_{self.seed}" / "db-cbs"


def clean_tex(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace(r"\_", "_")
    cleaned = cleaned.replace(r"\;", ";")
    cleaned = re.sub(r"\\(?:bfseries|textbf)\s*", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace(r"\hline", "")
    cleaned = cleaned.replace(r"\\", "")
    return cleaned.strip()


def number_or_none(value: str) -> float | None:
    cleaned = clean_tex(value)
    if cleaned in {"", "-", r"\textemdash", "textemdash"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else None


def finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def parse_table_conditions(
    source: Path,
    system: str,
    radius: str,
    environment_map: dict[str, str],
) -> list[Condition]:
    conditions = []
    if not source.exists():
        return conditions

    for line in source.read_text().splitlines():
        if r"my\_examples/" not in line:
            continue
        match = TABLE_ROW_RE.search(line)
        if not match:
            continue
        env_key = clean_tex(match.group("env"))
        environment = environment_map.get(env_key)
        if environment is None:
            continue
        cells = [clean_tex(cell) for cell in line.split("&")]
        if len(cells) < 5:
            continue
        conditions.append(Condition(
            system=system,
            radius=radius,
            env_key=env_key,
            environment=environment,
            agents=int(match.group("agents")),
            seed=int(match.group("seed")),
            source=source,
            tex_success=number_or_none(cells[2]),
            tex_time=number_or_none(cells[3]),
            tex_cost=number_or_none(cells[4]),
        ))
    return conditions


def load_trial_stat(stats_path: Path) -> tuple[float, float] | None:
    if not stats_path.exists():
        return None
    try:
        data = yaml.safe_load(stats_path.read_text())
    except yaml.YAMLError:
        return None
    entries = data.get("stats") if isinstance(data, dict) else None
    if not entries:
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        time = finite_positive(entry.get("t"))
        cost = finite_positive(entry.get("cost"))
        if time is None or cost is None:
            continue
        if time > MAX_PLANNING_TIME:
            continue
        return time, cost
    return None


def trial_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.iterdir() if path.is_dir() and path.name.isdigit())


def summarize_condition(condition: Condition) -> dict[str, str]:
    trials = trial_dirs(condition.raw_dir)
    successful = []
    for trial_dir in trials:
        stat = load_trial_stat(trial_dir / "stats.yaml")
        if stat is not None:
            successful.append(stat)

    times = [time for time, _ in successful]
    costs = [cost for _, cost in successful]
    trial_count = len(trials)
    success_count = len(successful)
    success_rate = success_count / trial_count if trial_count else 0.0

    return {
        "system": condition.system,
        "radius": condition.radius,
        "environment": condition.environment,
        "agents": str(condition.agents),
        "success_rate": f"{success_rate:.6g}",
        "time": "" if not times else f"{median(times):.6g}",
        "cost": "" if not costs else f"{median(costs):.6g}",
        "trials": str(trial_count),
        "successes": str(success_count),
        "source": str(condition.raw_dir),
    }


def aggregate_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["system"], row["radius"], row["environment"], row["agents"])
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for (system, radius, environment, agents), group in grouped.items():
        success_counts = [int(row["successes"]) for row in group]
        trial_counts = [int(row["trials"]) for row in group]
        time_values = [float(row["time"]) for row in group if row["time"]]
        cost_values = [float(row["cost"]) for row in group if row["cost"]]
        total_successes = sum(success_counts)
        total_trials = sum(trial_counts)
        aggregated.append({
            "system": system,
            "radius": radius,
            "environment": environment,
            "agents": agents,
            "success_rate": "" if total_trials == 0 else f"{total_successes / total_trials:.6g}",
            "time": "" if not time_values else f"{median(time_values):.6g}",
            "cost": "" if not cost_values else f"{median(cost_values):.6g}",
            "trials": str(total_trials),
            "successes": str(total_successes),
            "source": ";".join(sorted({row["source"] for row in group})),
        })

    aggregated.sort(key=lambda row: (
        row["system"],
        row["radius"],
        row["environment"],
        int(row["agents"]),
    ))
    return aggregated


def parse_ucycle() -> list[dict[str, str]]:
    conditions = parse_table_conditions(UCYCLE_SOURCE, "UCYCLE", "0.3", UCYCLE_ENVIRONMENTS)
    return [summarize_condition(condition) for condition in conditions]


def parse_quad() -> list[dict[str, str]]:
    conditions = parse_table_conditions(QUAD_SOURCE, "QUAD", "0.3", QUAD_ENVIRONMENTS)
    return [summarize_condition(condition) for condition in conditions]


def main() -> None:
    rows = aggregate_condition_rows(parse_ucycle() + parse_quad())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "radius",
            "environment",
            "agents",
            "success_rate",
            "time",
            "cost",
            "trials",
            "successes",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    available = sum(1 for row in rows if row["time"] and row["cost"])
    print(f"Wrote {OUTPUT} ({len(rows)} rows, {available} with time/cost)")


if __name__ == "__main__":
    main()
