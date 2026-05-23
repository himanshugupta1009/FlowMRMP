import datetime
import json
import math
import os
import subprocess
from contextlib import redirect_stdout

import numpy as np


def _git_output(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _git_status_short():
    status = _git_output(["status", "--short"])
    if status is None:
        return None
    return status.splitlines()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _obstacle_config(obstacle):
    if all(hasattr(obstacle, attr) for attr in ("x", "y", "z", "l", "w", "h")):
        return {
            "type": obstacle.__class__.__name__,
            "center": [obstacle.x, obstacle.y, obstacle.z],
            "size": [obstacle.l, obstacle.w, obstacle.h],
        }
    if all(hasattr(obstacle, attr) for attr in ("x", "y", "w", "h")):
        return {
            "type": obstacle.__class__.__name__,
            "center": [obstacle.x, obstacle.y],
            "size": [obstacle.w, obstacle.h],
        }
    if all(hasattr(obstacle, attr) for attr in ("x", "y", "r")):
        return {
            "type": obstacle.__class__.__name__,
            "center": [obstacle.x, obstacle.y],
            "radius": obstacle.r,
        }
    return {
        "type": obstacle.__class__.__name__,
        "repr": repr(obstacle),
    }


def _agent_builder_config(agent_builder):
    if hasattr(agent_builder, "get_manifest_config"):
        return agent_builder.get_manifest_config()
    return {
        "class": agent_builder.__class__.__name__,
        "name": getattr(agent_builder, "name", None),
    }


def _single_runtime_value(values):
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    if len(unique_values) == 1:
        return unique_values[0]
    return {
        "varies_by_agent": True,
        "unique_values": unique_values,
    }


def _runtime_values_from_agents(agents):
    sampling_time_steps = [
        agent.sampling_time_step
        for agent in agents
        if hasattr(agent, "sampling_time_step")
    ]
    num_skip_edges = [
        agent.num_skip_edges
        for agent in agents
        if hasattr(agent, "num_skip_edges")
    ]

    values = {}
    if sampling_time_steps:
        values["min_sampling_time_step"] = min(sampling_time_steps)
        values["agent_sampling_time_step"] = _single_runtime_value(
            sampling_time_steps)
    if num_skip_edges:
        values["min_num_skip_edges"] = min(num_skip_edges)
        values["agent_num_skip_edges"] = _single_runtime_value(num_skip_edges)
    return values


def _test_class_config(test_class, runtime_values):
    if hasattr(test_class, "get_manifest_config"):
        return test_class.get_manifest_config(runtime_values=runtime_values)
    return {
        "class": test_class.__class__.__name__,
        "name": getattr(test_class, "name", None),
        "short_name": getattr(test_class, "short_name", None),
    }


def build_pipeline_manifest(
        *,
        pipeline,
        savepath,
        pipeline_file,
        environment_name,
        extra_experiment_config=None):
    seeds = list(range(pipeline.master_seed,
                       pipeline.master_seed + pipeline.test_rounds))
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        sample_agents, sample_starts, sample_obstacles, sample_goals, sample_goal_radii = (
            pipeline.get_env_parms(pipeline.master_seed)
        )
    runtime_values = _runtime_values_from_agents(sample_agents)

    manifest = {
        "manifest_version": 1,
        "run": {
            "result_dir": savepath,
            "pipeline_file": pipeline_file,
            "pipeline_class": pipeline.__class__.__name__,
            "environment_name": environment_name,
            "created_at_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_status_short()),
            "git_status_short": _git_status_short(),
        },
        "experiment": {
            "num_agents": pipeline.num_agents,
            "test_rounds": pipeline.test_rounds,
            "master_seed": pipeline.master_seed,
            "seed_range": {
                "first_seed": seeds[0] if seeds else None,
                "last_seed": seeds[-1] if seeds else None,
                "num_seeds": len(seeds),
            },
            "goal_radius": pipeline.goal_radius,
            "num_processes": pipeline.processes,
        },
        "environment": {
            "width": pipeline.env_width,
            "breadth": pipeline.env_bredth,
            "height": pipeline.env_height,
            "obstacles": [_obstacle_config(obs) for obs in sample_obstacles],
            "sample_seed": pipeline.master_seed,
            "sample_starts": sample_starts,
            "sample_goals": sample_goals,
            "sample_goal_radii": sample_goal_radii,
        },
        "agent_builders": [
            _agent_builder_config(agent_builder)
            for agent_builder in pipeline.agent_builders
        ],
        "planners": [
            _test_class_config(test_class, runtime_values)
            for test_class in pipeline.test_classes
        ],
        "output_files": {
            "stats": "stats.txt",
            "log": "log.txt",
            "csv_dir": "csvs/",
            "plots": [
                "successes_barplot.png",
                "Computation_Time_(s)_boxen.png",
                "Total_Path_Costs_boxen.png",
                "Average_Agent_Path_Time_(s)_boxen.png",
                "Max_Agent_Path_Time_(s)_boxen.png",
            ],
        },
    }

    if extra_experiment_config:
        manifest["experiment"].update(extra_experiment_config)

    return _json_ready(manifest)


def write_pipeline_manifest(*, savepath, **kwargs):
    os.makedirs(savepath, exist_ok=True)
    manifest = build_pipeline_manifest(savepath=savepath, **kwargs)
    manifest_path = os.path.join(savepath, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path
