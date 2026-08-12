import csv
import gc
import os
import sys
from contextlib import redirect_stdout

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent_builders import (
    QuadcopterBuilder,
    SecondOrderCarBuilder,
    UnicycleBuilder,
)
from experiment_manifest import write_pipeline_manifest
from test_classes import (
    KcbsKinoTiEbTestClass,
    KinoTiCRRTEBTestClass,
    PrioritizedKinoTIRRTTestClass,
)
from test_pipeline_small_cluttered import TestPipelineCluttered


def _format_value(value):
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def _clear_edge_bundle_caches():
    # The builder EB caches are class-level and do not include every ablation
    # parameter in their cache keys, so clear them between variants.
    UnicycleBuilder.edge_bundles.clear()
    UnicycleBuilder.kino_ti_edge_bundles.clear()
    SecondOrderCarBuilder.kino_ti_edge_bundles.clear()
    QuadcopterBuilder.kino_ti_edge_bundles.clear()


def _make_agent_builder(builder_class, ablation):
    kwargs = {
        "num_skip_edges": ablation["num_skip_edges"],
        "sort_edges": ablation["sort_edges"],
    }
    if ablation["kd_num_edges"] is not None:
        kwargs["kd_num_edges"] = ablation["kd_num_edges"]

    return builder_class(**kwargs)


def _make_test_class(planner_key, ablation, planning_time):
    common_kwargs = {
        "max_planning_time": planning_time,
        "obs_buffers": False,
        "kd_delta_radius": ablation["kd_delta_radius"],
        "max_num_edges_per_node": ablation["max_num_edges_per_node"],
        "num_extension_trials": ablation["num_extension_trials"],
        "epsilon_random": ablation["epsilon_random"],
        "dynamic_agent_clearance": ablation["dynamic_agent_clearance"],
        # Edge ordering and traversal schedule are independent ablations.
        # Keep the golden-rank schedule for both sorted and unsorted edges.
        "use_geometric_candidate_schedule": True,
    }

    if planner_key == "kcbs":
        return KcbsKinoTiEbTestClass(**common_kwargs)
    if planner_key == "prrt":
        return PrioritizedKinoTIRRTTestClass(**common_kwargs)
    if planner_key == "crrt":
        return KinoTiCRRTEBTestClass(
            **common_kwargs,
            branch_goal_parking=True,
            num_edge_candidates_per_agent=ablation[
                "num_edge_candidates_per_agent"],
            # Use the shared candidate-attempt ablation value as Kite-cRRT's
            # joint candidate/repair budget.
            max_joint_edge_trials=ablation["num_skip_edges"],
            fallback_to_random_control=ablation["fallback_to_random_control"],
        )
    raise ValueError("Unknown planner key: " + planner_key)


def _agent_count_label(agent_type, agent_count, test_rounds, master_seed,
                       goal_radius):
    return (
        agent_type
        + "_a" + str(agent_count)
        + "_tests" + str(test_rounds)
        + "_seed" + str(master_seed)
        + "_gr" + str(goal_radius)
    )


def _ablation_run_label(ablation):
    return (
        "kd" + _format_value(ablation["kd_delta_radius"])
        + "_edges" + _format_value(ablation["kd_num_edges"] or "default")
        + "_skip" + _format_value(ablation["num_skip_edges"])
        + "_sort" + str(ablation["sort_edges"])
    )


def _effective_run_key(
        agent_builder,
        test_class,
        planner_key,
        agent_count,
        test_rounds,
        master_seed,
        goal_radius,
        planning_time):
    """Identify runs that give a planner the same effective configuration."""
    return (
        agent_builder.__class__.__name__,
        planner_key,
        agent_count,
        test_rounds,
        master_seed,
        goal_radius,
        planning_time,
        agent_builder.kd_num_edges,
        agent_builder.num_skip_edges,
        agent_builder.sort_edges,
        test_class.kd_delta_radius,
        test_class.max_num_edges_per_node,
        test_class.num_extension_trials,
        test_class.epsilon_random,
        test_class.dynamic_agent_clearance,
        test_class.use_geometric_candidate_schedule,
        getattr(test_class, "num_edge_candidates_per_agent", None),
        getattr(test_class, "max_joint_edge_trials", None),
        getattr(test_class, "fallback_to_random_control", None),
    )


SUMMARY_FIELDS = [
    "agent_type",
    "planner",
    "planner_key",
    "ablation_group",
    "ablation_value",
    "agent_count",
    "test_rounds",
    "master_seed",
    "goal_radius",
    "kd_num_edges",
    "kd_delta_radius",
    "sort_edges",
    "rank_candidates",
    "num_skip_edges",
    "max_num_edges_per_node",
    "num_extension_trials",
    "epsilon_random",
    "dynamic_agent_clearance",
    "use_geometric_candidate_schedule",
    "num_edge_candidates_per_agent",
    "max_joint_edge_trials",
    "fallback_to_random_control",
    "total_runs",
    "successes",
    "success_rate",
    "mean_computation_time_s",
    "mean_total_path_cost",
    "mean_average_agent_path_time_s",
    "mean_max_agent_path_time_s",
    "run_label",
    "result_dir",
]


def _mean(values):
    if len(values) == 0:
        return ""
    return sum(values) / len(values)


def _make_summary_row(
        *,
        agent_type,
        planner_key,
        planner_label,
        ablation,
        agent_count,
        test_rounds,
        master_seed,
        goal_radius,
        agent_builder,
        test_class,
        run_label,
        result_dir):
    sorted_success = dict(sorted(test_class.success.items()))
    success_values = [bool(value) for value in sorted_success.values()]
    successes = sum(success_values)
    total_runs = len(success_values)

    sorted_times = list(dict(sorted(test_class.times.items())).values())
    sorted_costs = list(dict(sorted(test_class.costs.items())).values())
    sorted_path_times = list(dict(sorted(test_class.path_times.items())).values())
    sorted_max_times = list(dict(sorted(test_class.max_times.items())).values())

    successful_times = [
        value for value, success in zip(sorted_times, success_values)
        if success
    ]
    successful_costs = [
        value for value, success in zip(sorted_costs, success_values)
        if success
    ]
    successful_path_times = [
        value for value, success in zip(sorted_path_times, success_values)
        if success
    ]
    successful_max_times = [
        value for value, success in zip(sorted_max_times, success_values)
        if success
    ]

    success_rate = ""
    if total_runs > 0:
        success_rate = successes / total_runs

    return {
        "agent_type": agent_type,
        "planner": planner_label,
        "planner_key": planner_key,
        "ablation_group": ablation["group"],
        "ablation_value": ablation["value_label"],
        "agent_count": agent_count,
        "test_rounds": test_rounds,
        "master_seed": master_seed,
        "goal_radius": goal_radius,
        "kd_num_edges": agent_builder.kd_num_edges,
        "kd_delta_radius": test_class.kd_delta_radius,
        "sort_edges": agent_builder.sort_edges,
        "rank_candidates": agent_builder.sort_edges is True,
        "num_skip_edges": agent_builder.num_skip_edges,
        "max_num_edges_per_node": test_class.max_num_edges_per_node,
        "num_extension_trials": test_class.num_extension_trials,
        "epsilon_random": test_class.epsilon_random,
        "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
        "use_geometric_candidate_schedule": (
            test_class.use_geometric_candidate_schedule),
        "num_edge_candidates_per_agent": getattr(
            test_class, "num_edge_candidates_per_agent", ""),
        "max_joint_edge_trials": getattr(
            test_class, "max_joint_edge_trials", ""),
        "fallback_to_random_control": getattr(
            test_class, "fallback_to_random_control", ""),
        "total_runs": total_runs,
        "successes": successes,
        "success_rate": success_rate,
        "mean_computation_time_s": _mean(successful_times),
        "mean_total_path_cost": _mean(successful_costs),
        "mean_average_agent_path_time_s": _mean(successful_path_times),
        "mean_max_agent_path_time_s": _mean(successful_max_times),
        "run_label": run_label,
        "result_dir": result_dir,
    }


def _write_summary_row(summary_path, row):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [
        existing for existing in rows
        if not (
            existing.get("ablation_value") == str(row["ablation_value"])
            and existing.get("agent_count") == str(row["agent_count"])
            and existing.get("test_rounds") == str(row["test_rounds"])
            and existing.get("master_seed") == str(row["master_seed"])
        )
    ]
    rows.append({field: row.get(field, "") for field in SUMMARY_FIELDS})

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    planning_time = 300.0
    # save_root = "paper_results/final_results/ablations/small_cluttered_env"
    # save_root = "paper_results/results_9June2026/ablations/small_cluttered_env"
    save_root = (
        "paper_results/results_23July2026/"
        "ablations_3Aug2026/small_cluttered_env"
    )
    test_rounds = 100
    goal_radius = 0.5
    seed_multiplier = 200
    num_processes = 20

    # agent_counts = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]
    agent_counts = [15, 18]
    agent_builder_classes = [
        SecondOrderCarBuilder,
        # UnicycleBuilder,
    ]
    planner_specs = [
        ("kcbs", "KCBS"),
        ("prrt", "PRRT"),
        ("crrt", "CRRT"),
    ]

    baseline = {
        "kd_num_edges": None,
        "kd_delta_radius": 0.10,
        "sort_edges": True,
        "num_skip_edges": 10,
        "max_num_edges_per_node": 1000,
        "num_extension_trials": 1,
        "epsilon_random": 0.01,
        "dynamic_agent_clearance": 0.0,
        "num_edge_candidates_per_agent": 10,
        "fallback_to_random_control": True,
    }

    ablations = [
        {
            **baseline,
            "group": "kd_num_edges",
            "value_label": "edges_10000",
            "kd_num_edges": 10000,
        },
        {
            **baseline,
            "group": "kd_num_edges",
            "value_label": "edges_30000",
            "kd_num_edges": 30000,
        },
        {
            **baseline,
            "group": "kd_num_edges",
            "value_label": "edges_50000",
            "kd_num_edges": 50000,
        },
        {
            **baseline,
            "group": "kd_num_edges",
            "value_label": "edges_75000",
            "kd_num_edges": 75000,
        },
        {
            **baseline,
            "group": "kd_num_edges",
            "value_label": "edges_100000",
            "kd_num_edges": 100000,
        },
        {
            **baseline,
            "group": "kd_delta_radius",
            "value_label": "radius_0p01",
            "kd_delta_radius": 0.01,
        },        
        {
            **baseline,
            "group": "kd_delta_radius",
            "value_label": "radius_0p05",
            "kd_delta_radius": 0.05,
        },
        {
            **baseline,
            "group": "kd_delta_radius",
            "value_label": "radius_0p10",
            "kd_delta_radius": 0.10,
        },
        {
            **baseline,
            "group": "kd_delta_radius",
            "value_label": "radius_0p20",
            "kd_delta_radius": 0.20,
        },
        {
            **baseline,
            "group": "kd_delta_radius",
            "value_label": "radius_0p50",
            "kd_delta_radius": 0.5,
        },        
        {
            **baseline,
            "group": "edge_sorting",
            "value_label": "sorted",
            "sort_edges": True,
        },
        {
            **baseline,
            "group": "edge_sorting",
            "value_label": "unsorted",
            "sort_edges": False,
        },
        {
            **baseline,
            "group": "num_skip_edges",
            "value_label": "skip_1",
            "num_skip_edges": 1,
        },
        {
            **baseline,
            "group": "num_skip_edges",
            "value_label": "skip_5",
            "num_skip_edges": 5,
        },
        {
            **baseline,
            "group": "num_skip_edges",
            "value_label": "skip_10",
            "num_skip_edges": 10,
        },
        {
            **baseline,
            "group": "num_skip_edges",
            "value_label": "skip_20",
            "num_skip_edges": 20,
        },
    ]

    # Separate ablation groups contain the same effective baseline. Keep the
    # first completed result and reuse its metrics for later aliases.
    completed_runs = {}

    for ablation in ablations:
        for agent_count in agent_counts:
            master_seed = agent_count * seed_multiplier

            for builder_class in agent_builder_classes:
                for planner_key, planner_label in planner_specs:
                    _clear_edge_bundle_caches()
                    agent_builder = _make_agent_builder(builder_class, ablation)
                    test_class = _make_test_class(
                        planner_key,
                        ablation,
                        planning_time,
                    )

                    agent_count_label = _agent_count_label(
                        agent_builder.name,
                        agent_count,
                        test_rounds,
                        master_seed,
                        goal_radius,
                    )
                    run_label = _ablation_run_label(ablation)

                    savepath = os.path.join(
                        save_root,
                        agent_count_label,
                        ablation["group"],
                        planner_label,
                        ablation["value_label"],
                        run_label,
                    )

                    effective_run_key = _effective_run_key(
                        agent_builder,
                        test_class,
                        planner_key,
                        agent_count,
                        test_rounds,
                        master_seed,
                        goal_radius,
                        planning_time,
                    )
                    completed_run = completed_runs.get(effective_run_key)

                    if completed_run is None:
                        os.makedirs(savepath, exist_ok=True)
                        print(
                            "Running small cluttered ablation",
                            ablation["group"],
                            ablation["value_label"],
                            "for",
                            agent_builder.name,
                            planner_label,
                            "with",
                            agent_count,
                            "agents",
                        )

                        pipeline = TestPipelineCluttered(
                            [test_class],
                            [agent_builder],
                            test_rounds=test_rounds,
                            num_agents=agent_count,
                            master_seed=master_seed,
                            savepath=savepath,
                            goal_radius=goal_radius,
                            processes=num_processes,
                        )
                        write_pipeline_manifest(
                            pipeline=pipeline,
                            savepath=savepath,
                            pipeline_file=__file__,
                            environment_name="small_cluttered_env",
                            extra_experiment_config={
                                "agent_type": agent_builder.name,
                                "planner_key": planner_key,
                                "planner_label": planner_label,
                                "planning_time": planning_time,
                                "seed_multiplier": seed_multiplier,
                                "ablation_group": ablation["group"],
                                "ablation_value": ablation["value_label"],
                                "ablation": ablation,
                            },
                        )

                        with open(savepath + "/log.txt", "w") as f:
                            with redirect_stdout(f):
                                pipeline.run()

                        failed_classes = pipeline.print_stats(
                            filename=savepath,
                            plots=True,
                        )
                        completed_run = {
                            "test_class": test_class,
                            "run_label": run_label,
                            "result_dir": savepath,
                        }
                        completed_runs[effective_run_key] = completed_run
                        print("Failed Classes:", failed_classes)
                    else:
                        print(
                            "Reusing equivalent results for",
                            ablation["group"],
                            ablation["value_label"],
                            "from",
                            completed_run["result_dir"],
                        )

                    summary_path = os.path.join(
                        save_root,
                        agent_count_label,
                        ablation["group"],
                        planner_label,
                        "summary.csv",
                    )
                    _write_summary_row(
                        summary_path,
                        _make_summary_row(
                            agent_type=agent_builder.name,
                            planner_key=planner_key,
                            planner_label=planner_label,
                            ablation=ablation,
                            agent_count=agent_count,
                            test_rounds=test_rounds,
                            master_seed=master_seed,
                            goal_radius=goal_radius,
                            agent_builder=agent_builder,
                            test_class=completed_run["test_class"],
                            run_label=completed_run["run_label"],
                            result_dir=completed_run["result_dir"],
                        ),
                    )

                    gc.collect()
