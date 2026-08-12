#!/usr/bin/env python3
"""Plot success rate vs. number of agents for paper figures."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from output_paths import DEFAULT_RESULTS_ROOT, csv_output_path, environment_results_dir, row_radius, system_results_root


RUN_DIR_RE = re.compile(
    r"^(?P<system>[A-Z]+)_a(?P<agents>\d+)_tests(?P<tests>\d+)_"
    r"seed(?P<seed>\d+)_gr(?P<goal_radius>[\d.]+)_kd(?P<kd_radius>[\d.]+)$"
)
DBCBS_NORMALIZED_CSV = Path("paper_plots/data/dbcbs_normalized.csv")
UCYCLE_RADIUS = os.environ.get("UCYCLE_RADIUS", "0.3")


@dataclass(frozen=True)
class PlannerSpec:
    csv_name: str
    label: str
    color: str = "#222222"
    marker: str = "o"
    linestyle: str = "-"
    external: bool = False


PARADIGMS = {
    "centralized": {
        "title": "centralized planning",
        "planners": (
            PlannerSpec("CRRT_results.csv", "cRRT", "#4C566A", "o", "--"),
            PlannerSpec("K-TI_EB_CRRT_results.csv", "cRRT+KiTE (Ours)", "#0072B2", "s", "-"),
        ),
    },
    "prrt": {
        "title": "prioritized planning",
        "planners": (
            PlannerSpec("PRRT_results.csv", "pRRT", "#4C566A", "o", "--"),
            PlannerSpec("KTI_EB_PRRT_results.csv", "pRRT+KiTE (Ours)", "#0072B2", "s", "-"),
        ),
    },
    "kcbs": {
        "title": "CBS",
        "planners": (
            PlannerSpec("RRT_KCBS_results.csv", "CBS", "#4C566A", "o", "--"),
            PlannerSpec("KTI_EB_RRT_KCBS_results.csv", "CBS+KiTE (Ours)", "#0072B2", "s", "-"),
            PlannerSpec("dbRRT_KCBS_results.csv", "CBS+idb-RRT", "#CC79A7", "X", ":"),
            PlannerSpec("", "dbCBS", "#D55E00", "D", "-.", external=True),
        ),
    },
}

SYSTEM_ENVIRONMENTS = {
    "SOC": (
        "small_cluttered_env",
        "large_cluttered_env",
        "swap_env",
        "narrow_corridor_env",
    ),
    "UCYCLE": (
        "small_cluttered_env",
        "large_cluttered_env",
        "swap_env",
        "narrow_corridor_env",
    ),
    "QUAD": (
        "large_cluttered_3d_env",
        "swap_3d_env",
    ),
}

SYSTEM_LABELS = {
    "SOC": "SOC",
    "UCYCLE": "UC",
    "QUAD": "DI",
}

ENVIRONMENT_LABELS = {
    "small_cluttered_env": "Small",
    "large_cluttered_env": "Large",
    "swap_env": "Swap",
    "narrow_corridor_env": "Corridor",
    "large_cluttered_3d_env": "Large 3D",
    "swap_3d_env": "Swap 3D",
}

ENVIRONMENT_COLORS = {
    "small_cluttered_env": "#A50F15",
    "large_cluttered_env": "#E66101",
    "swap_env": "#006B3C",
    "narrow_corridor_env": "#8A8A8A",
    "large_cluttered_3d_env": "#E69F00",
    "swap_3d_env": "#332288",
}

METHOD_STYLE_SEQUENCE = (
    {"linestyle": "--", "marker": "o"},
    {"linestyle": "-", "marker": "s"},
    {"linestyle": ":", "marker": "X"},
    {"linestyle": "-.", "marker": "D"},
)


def method_styles(planners: tuple[PlannerSpec, ...]) -> dict[str, dict[str, str]]:
    return {planner.label: METHOD_STYLE_SEQUENCE[index] for index, planner in enumerate(planners)}



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
        for record in sorted(records, key=lambda item: float(item["value"])):
            if current and abs(float(record["value"]) - float(current[-1]["value"])) > y_tolerance:
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
        ordered = sorted(records, key=lambda record: (env_index[str(record["environment"])], str(record["planner"])))
        center = (len(ordered) - 1) / 2.0
        for idx, record in enumerate(ordered):
            shifted = dict(record)
            shifted["x"] = float(record["agent"]) + (idx - center) * spread / max(center, 1.0)
            dodged.append(shifted)
    return dodged

def read_success_count(csv_path: Path) -> tuple[int, int]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    successes = sum(row["Success"].strip().lower() == "true" for row in rows)
    return successes, len(rows)


def load_dbcbs_success_rates() -> dict[tuple[str, str, int], float]:
    if not DBCBS_NORMALIZED_CSV.exists():
        return {}
    rates = {}
    with DBCBS_NORMALIZED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] == "UCYCLE" and row.get("radius", "0.3") != UCYCLE_RADIUS:
                continue
            if not row["success_rate"]:
                continue
            rates[(row["system"], row["environment"], int(row["agents"]))] = float(row["success_rate"])
    return rates


def collect_success_rates(
    results_root: Path,
    environment: str,
    system: str,
    planners: tuple[PlannerSpec, ...],
) -> dict[str, list[dict[str, float | int | str]]]:
    env_dir = environment_results_dir(results_root, system, environment)
    if not env_dir.exists():
        raise FileNotFoundError(f"Missing environment results directory: {env_dir}")

    series: dict[str, list[dict[str, float | int | str]]] = {planner.label: [] for planner in planners}
    dbcbs_success_rates = load_dbcbs_success_rates() if any(planner.external for planner in planners) else {}

    for run_dir in sorted(env_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        match = RUN_DIR_RE.match(run_dir.name)
        if not match or match.group("system") != system:
            continue

        agents = int(match.group("agents"))
        for planner in planners:
            if planner.external:
                rate = dbcbs_success_rates.get((system, environment, agents))
                if rate is None:
                    continue
                total = int(match.group("tests"))
                successes = round(rate * total)
                csv_path = DBCBS_NORMALIZED_CSV
            else:
                csv_path = run_dir / "csvs" / planner.csv_name
                if not csv_path.exists():
                    continue

                successes, total = read_success_count(csv_path)
                rate = successes / total if total else 0.0
            series[planner.label].append(
                {
                    "agents": agents,
                    "successes": successes,
                    "total": total,
                    "rate": rate,
                    "csv_path": str(csv_path),
                }
            )

    any_points = False
    for points in series.values():
        points.sort(key=lambda row: int(row["agents"]))
        any_points = any_points or bool(points)
    if not any_points:
        raise ValueError(f"No planner data found in {env_dir} with system={system}")

    return series


def write_summary_csv(
    series: dict[str, list[dict[str, float | int | str]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["radius", "planner", "agents", "successes", "total", "success_rate", "csv_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner, points in series.items():
            for point in points:
                writer.writerow(
                    {
                        "radius": row_radius("UCYCLE" if "UCYCLE_" in str(point["csv_path"]) else "", str(point["csv_path"])),
                        "planner": planner,
                        "agents": point["agents"],
                        "successes": point["successes"],
                        "total": point["total"],
                        "success_rate": f"{float(point['rate']):.6f}",
                        "csv_path": point["csv_path"],
                    }
                )


def write_multi_env_summary_csv(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["radius", "environment", "planner", "agents", "successes", "total", "success_rate", "csv_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for environment, series in data.items():
            for planner, points in series.items():
                for point in points:
                    writer.writerow(
                        {
                            "radius": row_radius("UCYCLE" if "UCYCLE_" in str(point["csv_path"]) else "", str(point["csv_path"])),
                            "environment": environment,
                            "planner": planner,
                            "agents": point["agents"],
                            "successes": point["successes"],
                            "total": point["total"],
                            "success_rate": f"{float(point['rate']):.6f}",
                            "csv_path": point["csv_path"],
                        }
                    )


def write_systems_summary_csv(
    data: dict[str, dict[str, dict[str, list[dict[str, float | int | str]]]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "radius", "environment", "planner", "agents", "successes", "total", "success_rate", "csv_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system, env_data in data.items():
            for environment, series in env_data.items():
                for planner, points in series.items():
                    for point in points:
                        writer.writerow(
                            {
                                "system": system,
                                "radius": row_radius(system, str(point["csv_path"])),
                                "environment": environment,
                                "planner": planner,
                                "agents": point["agents"],
                                "successes": point["successes"],
                                "total": point["total"],
                                "success_rate": f"{float(point['rate']):.6f}",
                                "csv_path": point["csv_path"],
                            }
                        )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (4.9, 3.25),
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.2,
        }
    )


def plot_success_vs_agents(
    series: dict[str, list[dict[str, float | int | str]]],
    planners: tuple[PlannerSpec, ...],
    title: str,
    output_stem: Path,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots()

    all_agents: set[int] = set()
    planner_by_label = {planner.label: planner for planner in planners}

    for label, points in series.items():
        planner = planner_by_label[label]
        agents = [int(point["agents"]) for point in points]
        rates = [100.0 * float(point["rate"]) for point in points]
        all_agents.update(agents)

        ax.plot(
            agents,
            rates,
            label=label,
            color=planner.color,
            marker=planner.marker,
            linestyle=planner.linestyle,
            markerfacecolor="none",
            markeredgewidth=1.4,
            zorder=3,
        )

    ax.set_xlabel("Robots", fontsize=10.2, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=10.2, fontweight="bold")
    ax.set_title(title, pad=7, fontsize=10.5, fontweight="bold")
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(sorted(all_agents))
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.7)
    ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=2.4)

    fig.tight_layout(pad=0.35)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def system_xticks(system: str) -> list[int]:
    if system == "QUAD":
        return [2, 5, 10, 15, 20, 25, 30]
    return [3, 5, 10, 15, 20, 25, 30]


def draw_system_panel(
    ax: plt.Axes,
    env_data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    system: str,
    planners: tuple[PlannerSpec, ...],
    show_ylabel: bool,
) -> None:
    all_agents: set[int] = set()
    styles = method_styles(planners)

    for environment in SYSTEM_ENVIRONMENTS[system]:
        for planner in planners:
            all_agents.update(int(point["agents"]) for point in env_data[environment][planner.label])

    ticks = [tick for tick in system_xticks(system) if min(all_agents) <= tick <= max(all_agents)]
    tick_set = set(ticks)

    draw_order = sorted(planners, key=lambda planner: 1 if "KiTE" in planner.label else 0)
    marker_records: list[dict[str, object]] = []
    for environment in SYSTEM_ENVIRONMENTS[system]:
        series = env_data[environment]
        color = ENVIRONMENT_COLORS[environment]
        for planner in draw_order:
            points = series[planner.label]
            if not points:
                continue
            agents = [int(point["agents"]) for point in points]
            rates = [100.0 * float(point["rate"]) for point in points]
            style = styles[planner.label]
            marker_indices = [index for index, agents_value in enumerate(agents) if agents_value in tick_set]
            zorder = 6 if "KiTE" in planner.label else 3
            ax.plot(
                agents,
                rates,
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.0,
                alpha=0.88,
                zorder=zorder,
            )
            for idx in marker_indices:
                marker_records.append({
                    "agent": agents[idx],
                    "value": rates[idx],
                    "planner": planner.label,
                    "environment": environment,
                    "color": color,
                    "marker": style["marker"],
                    "markersize": 4.0,
                    "alpha": 0.88,
                    "zorder": zorder + 0.2,
                })

    for marker in dodged_marker_positions(marker_records, SYSTEM_ENVIRONMENTS[system], system):
        ax.plot(
            [float(marker["x"])],
            [float(marker["value"])],
            color=str(marker["color"]),
            marker=str(marker["marker"]),
            linestyle="none",
            markersize=float(marker["markersize"]),
            markerfacecolor="none",
            markeredgewidth=0.85,
            alpha=float(marker["alpha"]),
            zorder=float(marker["zorder"]),
        )

    ax.set_title(SYSTEM_LABELS[system], pad=3, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Robots", labelpad=1.0, fontsize=10.2, fontweight="bold")
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    if show_ylabel:
        ax.set_ylabel("Success Rate (%)", fontsize=10.2, fontweight="bold")
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlim(min(all_agents) - 0.6, max(all_agents) + 0.6)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", rotation=0, pad=1.5)
    ax.tick_params(axis="y", pad=1.5)
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.58)
    ax.grid(True, axis="x", color="#ECEFF3", linewidth=0.34)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_multi_env_success_vs_agents(
    data: dict[str, dict[str, list[dict[str, float | int | str]]]],
    system: str,
    planners: tuple[PlannerSpec, ...],
    title: str,
    output_stem: Path,
) -> None:
    configure_matplotlib()
    plt.rcParams.update(
        {
            "figure.figsize": (4.9, 2.85),
            "lines.linewidth": 1.65,
            "lines.markersize": 4.6,
        }
    )

    fig, ax = plt.subplots()
    draw_system_panel(ax, data, system=system, planners=planners, show_ylabel=True)
    ax.set_title(title, pad=7, fontsize=10.5, fontweight="bold")

    fig.tight_layout(pad=0.35)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def legend_handles(planners: tuple[PlannerSpec, ...]):
    from matplotlib.lines import Line2D

    env_handles = [
        Line2D([0], [0], color=ENVIRONMENT_COLORS[environment], lw=2.0, label=ENVIRONMENT_LABELS[environment])
        for environment in (
            "small_cluttered_env",
            "large_cluttered_env",
            "swap_env",
            "narrow_corridor_env",
            "large_cluttered_3d_env",
            "swap_3d_env",
        )
    ]
    styles = method_styles(planners)
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            marker=styles[planner.label]["marker"],
            linestyle=styles[planner.label]["linestyle"],
            markerfacecolor="none",
            markeredgewidth=1.05,
            lw=1.55,
            label=planner.label,
        )
        for planner in planners
    ]
    return method_handles + env_handles


def plot_success_rate_legend(output_path: Path, planners: tuple[PlannerSpec, ...]) -> None:
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (5.35, 0.46), "legend.fontsize": 6.8})
    fig = plt.figure()
    handles = legend_handles(planners)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=min(8, len(handles)),
        frameon=False,
        columnspacing=0.72,
        handlelength=1.45,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def plot_all_systems_success_vs_agents(
    data: dict[str, dict[str, dict[str, list[dict[str, float | int | str]]]]],
    planners: tuple[PlannerSpec, ...],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    plt.rcParams.update(
        {
            "figure.figsize": (5.35, 1.78),
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.75,
        }
    )

    fig, axes = plt.subplots(1, 3, sharey=True)
    for index, system in enumerate(("UCYCLE", "SOC", "QUAD")):
        draw_system_panel(axes[index], data[system], system=system, planners=planners, show_ylabel=index == 0)

    fig.tight_layout(pad=0.16, w_pad=0.42)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate success-rate scaling plots from new_final_results CSVs."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--ucycle-results-root", type=Path, default=None)
    parser.add_argument("--environment", default="small_cluttered_env")
    parser.add_argument("--system", default="SOC")
    parser.add_argument("--paradigm", choices=sorted(PARADIGMS), default="centralized")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_plots/figures/success_rate"))
    parser.add_argument(
        "--all-soc-envs",
        action="store_true",
        help="Plot SOC baseline and KiTE variants across all 2D environments.",
    )
    parser.add_argument(
        "--all-systems",
        action="store_true",
        help="Plot SOC, unicycle, and double-integrator success scaling with one shared legend.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write compact CSV summaries of the plotted values.",
    )
    parser.add_argument(
        "--title",
        default="SOC, small cluttered environment",
        help="Short title printed above the axes.",
    )
    return parser.parse_args()


def collect_system_data(
    results_root: Path,
    system: str,
    planners: tuple[PlannerSpec, ...],
) -> dict[str, dict[str, list[dict[str, float | int | str]]]]:
    return {
        environment: collect_success_rates(
            results_root=results_root,
            environment=environment,
            system=system,
            planners=planners,
        )
        for environment in SYSTEM_ENVIRONMENTS[system]
    }


def main() -> None:
    args = parse_args()
    paradigm = PARADIGMS[args.paradigm]
    planners = paradigm["planners"]
    output_root = args.output_dir

    if args.all_soc_envs:
        data = collect_system_data(args.results_root, "SOC", planners)
        output_stem = output_root / f"success_{args.paradigm}_SOC_all_envs"
        if args.write_csv:
            write_multi_env_summary_csv(data, csv_output_path(output_stem))
        plot_multi_env_success_vs_agents(
            data,
            system="SOC",
            planners=planners,
            title=f"SOC {paradigm['title']} across environments",
            output_stem=output_stem,
        )
        print(f"Wrote {output_stem.with_suffix('.png')}")
        if args.write_csv:
            print(f"Wrote {csv_output_path(output_stem)}")
        return

    if args.all_systems:
        systems_data = {
            system: collect_system_data(
                system_results_root(args.results_root, system, args.ucycle_results_root),
                system,
                planners,
            )
            for system in ("UCYCLE", "SOC", "QUAD")
        }
        combined_stem = output_root / f"success_{args.paradigm}_all_systems"
        if args.write_csv:
            write_systems_summary_csv(systems_data, csv_output_path(combined_stem))
        plot_all_systems_success_vs_agents(systems_data, planners, combined_stem)
        legend_path = output_root / f"success_{args.paradigm}_legend.png"
        plot_success_rate_legend(legend_path, planners)
        print(f"Wrote {combined_stem.with_suffix('.png')}")
        print(f"Wrote {legend_path}")
        if args.write_csv:
            print(f"Wrote {csv_output_path(combined_stem)}")

        for system, data in systems_data.items():
            output_stem = output_root / f"success_{args.paradigm}_{system}_all_envs_no_legend"
            if system == "UCYCLE":
                output_stem = output_stem
            if args.write_csv:
                write_multi_env_summary_csv(data, csv_output_path(output_stem))
            plot_multi_env_success_vs_agents(
                data,
                system=system,
                planners=planners,
                title=f"{SYSTEM_LABELS[system]}",
                output_stem=output_stem,
            )
            print(f"Wrote {output_stem.with_suffix('.png')}")
            if args.write_csv:
                print(f"Wrote {csv_output_path(output_stem)}")
        return

    series = collect_success_rates(
        results_root=system_results_root(args.results_root, args.system, args.ucycle_results_root),
        environment=args.environment,
        system=args.system,
        planners=planners,
    )

    output_name = f"success_{args.paradigm}_{args.environment}_{args.system}"
    output_stem = output_root / output_name
    if args.system == "UCYCLE":
        output_stem = output_stem
    if args.write_csv:
        write_summary_csv(series, csv_output_path(output_stem))
    plot_success_vs_agents(series, planners, args.title, output_stem)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    if args.write_csv:
        print(f"Wrote {csv_output_path(output_stem)}")


if __name__ == "__main__":
    main()
