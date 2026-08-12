#!/usr/bin/env python3
"""Combined success-rate figure with one row per coordination paradigm."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, row_radius, system_results_root

RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)
DBCBS_NORMALIZED_CSV = Path("paper_plots/data/dbcbs_normalized.csv")
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")
SHOW_MARKERS = True


@dataclass(frozen=True)
class PlannerSpec:
    csv_name: str
    label: str
    role: str
    marker: str
    linestyle: str
    external: bool = False


PARADIGMS = {
    "centralized": {
        "row_label": "cRRT",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT", "Baseline", "o", "-"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE (Ours)", "+KiTE (Ours)", "s", "-"),
        ),
    },
    "prrt": {
        "row_label": "pRRT",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT", "Baseline", "o", "-"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE (Ours)", "+KiTE (Ours)", "s", "-"),
        ),
    },
    "kcbs": {
        "row_label": "CBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "CBS", "Baseline", "o", "-"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "CBS+KiTE (Ours)", "+KiTE (Ours)", "s", "-"),
            PlannerSpec("dbRRT_KCBS_results.csv", "CBS+idb-RRT", "+idb-RRT", "X", ":"),
            PlannerSpec("", "dbCBS", "dbCBS", "D", "-.", external=True),
        ),
    },
}
PARADIGM_ORDER = ("centralized", "prrt", "kcbs")
SYSTEMS = ("UCYCLE", "SOC", "QUAD")
SYSTEM_LABELS = {
    "UCYCLE": "UC",
    "SOC": "SOC",
    "QUAD": "DI",
}
SYSTEM_ENVIRONMENTS = {
    "SOC": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "UCYCLE": ("small_cluttered_env", "large_cluttered_env", "swap_env", "narrow_corridor_env"),
    "QUAD": ("large_cluttered_3d_env", "swap_3d_env"),
}
ENVIRONMENT_LABELS = {
    "narrow_corridor_env": "Corridor",
    "swap_env": "Swap",
    "small_cluttered_env": "Small Cluttered",
    "large_cluttered_env": "Large Cluttered",
    "swap_3d_env": "Swap 3D",
    "large_cluttered_3d_env": "Large Cluttered 3D",
}
ENVIRONMENT_COLORS = {
    "small_cluttered_env": "#A50F15",
    "large_cluttered_env": "#E66101",
    "swap_env": "#006B3C",
    "narrow_corridor_env": "#8A8A8A",
    "large_cluttered_3d_env": "#E69F00",
    "swap_3d_env": "#332288",
}
ROLE_STYLES = {
    "Baseline": {"marker": "o", "linestyle": "--", "alpha": 0.88, "linewidth": 1.0, "zorder": 3},
    "+KiTE (Ours)": {"marker": "s", "linestyle": "-", "alpha": 0.88, "linewidth": 1.0, "zorder": 6},
    "+idb-RRT": {"marker": "X", "linestyle": ":", "alpha": 0.88, "linewidth": 1.0, "zorder": 4},
    "dbCBS": {"marker": "D", "linestyle": "-.", "alpha": 0.88, "linewidth": 1.0, "zorder": 3},
}
ROLE_LEGEND_LABELS = {
    "Baseline": "Rand (dashed)",
    "+KiTE (Ours)": r"+KiTE ($\mathbf{Ours}$, solid)",
    "+idb-RRT": "+idb-RRT (dotted)",
    "dbCBS": "dbCBS (dash-dot)",
}


def load_dbcbs_success_rates() -> dict[tuple[str, str, int], float]:
    if not DBCBS_NORMALIZED_CSV.exists():
        return {}
    rates = {}
    with DBCBS_NORMALIZED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] == "UCYCLE" and row.get("radius", "0.3") != UCYCLE_RADIUS:
                continue
            if row["success_rate"]:
                rates[(row["system"], row["environment"], int(row["agents"]))] = float(row["success_rate"])
    return rates


def read_success_rate(csv_path: Path) -> tuple[int, int, float] | None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    successes = sum(row["Success"].strip().lower() == "true" for row in rows)
    return successes, len(rows), successes / len(rows)


def collect(results_root: Path, ucycle_results_root: Path | None = None):
    dbcbs_rates = load_dbcbs_success_rates()
    data = {
        paradigm: {
            system: {
                environment: {planner.label: [] for planner in PARADIGMS[paradigm]["planners"]}
                for environment in SYSTEM_ENVIRONMENTS[system]
            }
            for system in SYSTEMS
        }
        for paradigm in PARADIGM_ORDER
    }

    for paradigm in PARADIGM_ORDER:
        planners = PARADIGMS[paradigm]["planners"]
        for system in SYSTEMS:
            root = system_results_root(results_root, system, ucycle_results_root)
            for environment in SYSTEM_ENVIRONMENTS[system]:
                env_dir = environment_results_dir(root, system, environment)
                if not env_dir.exists():
                    continue
                for run_dir in sorted(env_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    match = RUN_DIR_RE.match(run_dir.name)
                    if not match or match.group("system") != system:
                        continue
                    agents = int(match.group("agents"))
                    for planner in planners:
                        if planner.external:
                            rate = dbcbs_rates.get((system, environment, agents))
                            if rate is None:
                                continue
                            successes = round(rate * int(match.group("tests")))
                            total = int(match.group("tests"))
                            csv_path = DBCBS_NORMALIZED_CSV
                        else:
                            csv_path = run_dir / "csvs" / planner.csv_name
                            if not csv_path.exists():
                                successes = 0
                                total = int(match.group("tests"))
                                rate = 0.0
                            else:
                                result = read_success_rate(csv_path)
                                if result is None:
                                    successes = 0
                                    total = int(match.group("tests"))
                                    rate = 0.0
                                else:
                                    successes, total, rate = result
                        data[paradigm][system][environment][planner.label].append({
                            "agents": agents,
                            "successes": successes,
                            "total": total,
                            "success_rate": 100.0 * rate,
                            "csv_path": str(csv_path),
                        })
    for paradigm_data in data.values():
        for system_data in paradigm_data.values():
            for env_data in system_data.values():
                for points in env_data.values():
                    points.sort(key=lambda row: int(row["agents"]))
    return data


def system_xticks(system: str) -> list[int]:
    if system == "QUAD":
        return [2, 5, 10, 15, 20, 25, 30]
    return [3, 5, 10, 15, 20, 25, 30]


def dodged_marker_positions(
    marker_records: list[dict[str, object]],
    environment_order: tuple[str, ...],
    system: str,
) -> list[dict[str, object]]:
    env_index = {environment: idx for idx, environment in enumerate(environment_order)}
    spread = 0.28 if len(environment_order) >= 4 else 0.18
    if system == "QUAD":
        spread *= 0.85
    y_tolerance = 1.25

    records_by_agent: dict[int, list[dict[str, object]]] = {}
    for record in marker_records:
        records_by_agent.setdefault(int(record["agent"]), []).append(record)

    groups: list[list[dict[str, object]]] = []
    for records in records_by_agent.values():
        current: list[dict[str, object]] = []
        for record in sorted(records, key=lambda item: float(item["rate"])):
            if current and abs(float(record["rate"]) - float(current[-1]["rate"])) > y_tolerance:
                groups.append(current)
                current = []
            current.append(record)
        if current:
            groups.append(current)

    dodged: list[dict[str, object]] = []
    for records in groups:
        if len(records) == 1:
            record = dict(records[0])
            record["x"] = float(record["agent"])
            dodged.append(record)
            continue

        ordered = sorted(records, key=lambda record: (env_index[str(record["environment"])], str(record["role"])))
        center = (len(ordered) - 1) / 2.0
        for idx, record in enumerate(ordered):
            shifted = dict(record)
            shifted["x"] = float(record["agent"]) + (idx - center) * spread / max(center, 1.0)
            dodged.append(shifted)
    return dodged

def plot(data, output_path: Path, compact: bool = False) -> None:
    if compact:
        size = {
            "figure.figsize": (7.65, 3.55),
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.55,
        }
    else:
        size = {
            "figure.figsize": (7.05, 5.65),
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.6,
            "axes.linewidth": 0.65,
        }

    plt.rcParams.update({
        **size,
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "serif",
    })
    fig, axes = plt.subplots(len(SYSTEMS), len(PARADIGM_ORDER), sharey=True)

    for row, system in enumerate(SYSTEMS):
        for col, paradigm in enumerate(PARADIGM_ORDER):
            planners = PARADIGMS[paradigm]["planners"]
            ax = axes[row][col]
            all_agents = set()
            for environment in SYSTEM_ENVIRONMENTS[system]:
                for planner in planners:
                    all_agents.update(
                        int(point["agents"])
                        for point in data[paradigm][system][environment][planner.label]
                    )
            ticks = [tick for tick in system_xticks(system) if all_agents and min(all_agents) <= tick <= max(all_agents)]
            tick_set = set(ticks)

            draw_order = sorted(planners, key=lambda planner: 1 if planner.role == "+KiTE (Ours)" else 0)
            marker_records: list[dict[str, object]] = []
            for environment in SYSTEM_ENVIRONMENTS[system]:
                color = ENVIRONMENT_COLORS[environment]
                for planner in draw_order:
                    points = data[paradigm][system][environment][planner.label]
                    if not points:
                        continue
                    agents = [int(point["agents"]) for point in points]
                    rates = [float(point["success_rate"]) for point in points]
                    style = ROLE_STYLES[planner.role]
                    marker_indices = [idx for idx, agent in enumerate(agents) if agent in tick_set]
                    marker_size = 5.3 if planner.role == "+KiTE (Ours)" else (4.9 if planner.role == "+idb-RRT" else 4.5)
                    ax.plot(
                        agents,
                        rates,
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=style["alpha"],
                        zorder=style["zorder"],
                    )
                    if SHOW_MARKERS:
                        for idx in marker_indices:
                            marker_records.append({
                                "agent": agents[idx],
                                "rate": rates[idx],
                                "role": planner.role,
                                "environment": environment,
                                "color": color,
                                "marker": style["marker"],
                                "markersize": 3.0 if compact else 4.0,
                                "alpha": style["alpha"],
                                "zorder": style["zorder"] + 0.2,
                            })

            if SHOW_MARKERS:
                for marker in dodged_marker_positions(marker_records, SYSTEM_ENVIRONMENTS[system], system):
                    ax.plot(
                        [float(marker["x"])],
                        [float(marker["rate"])],
                        color=str(marker["color"]),
                        marker=str(marker["marker"]),
                        linestyle="none",
                        markersize=float(marker["markersize"]),
                        markerfacecolor="none",
                        markeredgewidth=0.66 if compact else 0.82,
                        alpha=float(marker["alpha"]),
                        zorder=float(marker["zorder"]),
                    )

            if row == 0:
                ax.set_title(PARADIGMS[paradigm]["row_label"], pad=2.2 if compact else 4)
            if col == 0:
                ax.set_ylabel(SYSTEM_LABELS[system], labelpad=2.4 if compact else 5, fontsize=8.8 if compact else 10.2, fontweight="bold")
                if compact:
                    label_y = {"UCYCLE": 0.50, "SOC": 0.60, "QUAD": 0.40}[system]
                    ax.yaxis.set_label_coords(-0.13, label_y)
            else:
                ax.tick_params(labelleft=False)
            if row == len(SYSTEMS) - 1:
                ax.set_xlabel("Robots", labelpad=0.4 if compact else 1.0, fontsize=8.6 if compact else 10.2, fontweight="bold")
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 50, 100] if compact else [0, 25, 50, 75, 100])
            if all_agents:
                ax.set_xlim(min(all_agents) - 0.6, max(all_agents) + 0.6)
                ax.set_xticks(ticks)
            ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.34 if compact else 0.45, zorder=0)
            ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.20 if compact else 0.28, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", pad=0.25 if compact else 0.8)
            ax.tick_params(axis="y", pad=0.25 if compact else 0.8)

    fig.supylabel("Success Rate (%)", x=0.006 if compact else 0.008, fontsize=8.8 if compact else 10.2, fontweight="bold")

    role_handles = [
        Line2D([0, 1, 2], [0, 0, 0], color="#222222", marker=style["marker"],
               linestyle=style["linestyle"], markevery=[0, 1, 2],
               markerfacecolor="none", markeredgewidth=0.92 if compact else 1.05,
               markersize=(4.7 if role == "+KiTE (Ours)" else (4.4 if role == "+idb-RRT" else 4.0)) if compact else (6.0 if role == "+KiTE (Ours)" else (5.7 if role == "+idb-RRT" else 5.2)),
               lw=1.20 if compact else 1.65, label=ROLE_LEGEND_LABELS[role])
        for role, style in ROLE_STYLES.items()
    ]
    env_handles = [
        Line2D([0], [0], color=ENVIRONMENT_COLORS[env], lw=1.25 if compact else 1.8, label=label)
        for env, label in ENVIRONMENT_LABELS.items()
    ]
    variant_label = Line2D([], [], color="none", linestyle="none", label="Variant:")
    environment_label = Line2D([], [], color="none", linestyle="none", label="Environment:")
    fig.legend(
        handles=[variant_label] + role_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.006 if compact else 0.998),
        ncol=len(role_handles) + 1,
        frameon=False,
        columnspacing=0.34 if compact else 0.90,
        handlelength=2.35 if compact else 2.85,
        handletextpad=0.22 if compact else 0.34,
    )
    fig.legend(
        handles=[environment_label] + env_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962 if compact else 0.955),
        ncol=len(env_handles) + 1,
        frameon=False,
        columnspacing=0.18 if compact else 0.38,
        handlelength=1.35 if compact else 0.95,
        handletextpad=0.16 if compact else 0.24,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(
        rect=(0.020, 0, 1, 0.890) if compact else (0.030, 0, 1, 0.875),
        pad=0.05 if compact else 0.16,
        w_pad=0.12 if compact else 0.40,
        h_pad=0.10 if compact else 0.58,
    )
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = ["paradigm", "system", "radius", "environment", "planner", "agents", "successes", "total", "success_rate", "csv_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paradigm in PARADIGM_ORDER:
            for system in SYSTEMS:
                for environment in SYSTEM_ENVIRONMENTS[system]:
                    for planner in PARADIGMS[paradigm]["planners"]:
                        for point in data[paradigm][system][environment][planner.label]:
                            writer.writerow({
                                "paradigm": paradigm,
                                "system": system,
                                "radius": row_radius(system, str(point["csv_path"])),
                                "environment": environment,
                                "planner": planner.label,
                                "agents": point["agents"],
                                "successes": point["successes"],
                                "total": point["total"],
                                "success_rate": f"{point['success_rate']:.6f}",
                                "csv_path": point["csv_path"],
                            })


def main() -> None:
    root = DEFAULT_RESULTS_ROOT
    ucycle_root = Path(os.environ["UCYCLE_RESULTS_ROOT"]) if os.environ.get("UCYCLE_RESULTS_ROOT") else None
    output = Path("paper_plots/figures/success_rate/success_rate_all_paradigms.png")
    compact_output = Path("paper_plots/figures/success_rate/success_rate_all_paradigms_compact.png")
    data = collect(root, ucycle_root)
    plot(data, output)
    plot(data, compact_output, compact=True)
    write_csv(data, csv_output_path(output))
    print(f"Wrote {output}")
    print(f"Wrote {compact_output}")
    print(f"Wrote {csv_output_path(output)}")


if __name__ == "__main__":
    main()
