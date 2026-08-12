"""
Dynoplan/db-CBS C++ optimizer bridge for constrained unicycle Db-RRT paths.

This module keeps the Python low-level search in charge of constraints, then
uses the C++ Dynoplan optimizer as a repair backend. The optimized trajectory is
accepted only after the Python static and KCBS dynamic validators pass.
"""

from __future__ import annotations

import math
import itertools
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from constrainedX import constraint_satisfaction_numba
from db.constrained_db_optimize_unicycle import (
    OptimizedTrajectoryView,
    UnicycleTrajOptResult,
    _extract_obstacle_arrays,
    _static_boundary_feasibility_numba,
)


_CPP_OPT_CALL_COUNTER = itertools.count(1)
_CPP_OPT_TIMING_STATS = {
    "calls": 0,
    "successes": 0,
    "yaml_write_time": 0.0,
    "yaml_read_time": 0.0,
    "subprocess_wall_time": 0.0,
    "python_validation_time": 0.0,
    "total_bridge_time": 0.0,
}


def reset_cpp_dynoplan_optimizer_timing_stats():
    for key in _CPP_OPT_TIMING_STATS:
        _CPP_OPT_TIMING_STATS[key] = 0 if key in ("calls", "successes") else 0.0


def get_cpp_dynoplan_optimizer_timing_stats():
    return dict(_CPP_OPT_TIMING_STATS)


@dataclass
class CppDynoplanUnicycleOptimizerOptions:
    dbcbs_root: Path = Path("/home/himanshu/Documents/Research/dbcbs_stuff/db-CBS")
    robot_type: str = "unicycle1_sphere_v0"
    # Dynoplan solver ids: 0 = fixed-time trajectory optimization,
    # 1 = free-time/time-optimal optimization. Keep both static and
    # KCBS-constrained repairs fixed-time so optimized trajectories preserve
    # the warm-start time horizon used by the Python planner.
    solver_id_static: int = 0
    solver_id_constrained: int = 0
    smooth_traj: bool = True
    weight_goal: float = 200.0
    collision_weight: float = 100.0
    max_iter: int = 50
    timeout_s: float = 300.0
    include_static_in_moving_obstacles: bool = True
    validate_python: bool = True
    keep_files: bool = False
    scratch_dir: Optional[Path] = None
    extra_cfg: dict = field(default_factory=dict)

    @property
    def main_optimization(self) -> Path:
        return self.dbcbs_root / "buildRelease/dynoplan/main_optimization"

    @property
    def models_base_path(self) -> Path:
        return self.dbcbs_root / "dynoplan/dynobench/models"


def _float_list(values):
    return [float(v) for v in values]


def _has_constraints(planner) -> bool:
    constraints = getattr(planner, "constraints", None)
    return constraints is not None and len(constraints) > 0


def _dynoplan_static_obstacles(env):
    obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
    obstacles = []
    for obs in getattr(env, "obstacles", []):
        if hasattr(obs, "w") and hasattr(obs, "h"):
            obstacles.append(
                {
                    "type": "box",
                    "center": [float(obs.x), float(obs.y)],
                    "size": [
                        float(obs.w) + 2.0 * obstacle_buffer,
                        float(obs.h) + 2.0 * obstacle_buffer,
                    ],
                }
            )
        elif hasattr(obs, "r"):
            obstacles.append(
                {
                    "type": "sphere",
                    "center": [float(obs.x), float(obs.y)],
                    "size": [float(obs.r) + obstacle_buffer],
                }
            )
    return obstacles


def _constraint_position_at_time(
    collision_keys,
    collision_agent_position_array,
    t: float,
    dt: float,
    roundoff_digits: int,
):
    start_conflict_time = float(collision_keys[0])
    end_conflict_time = float(collision_keys[-1])
    if t < start_conflict_time or t > end_conflict_time:
        return None

    pos_index = int(round((t - start_conflict_time) / dt, roundoff_digits))
    pos_index = max(0, min(pos_index, collision_agent_position_array.shape[0] - 1))
    return collision_agent_position_array[pos_index]


def _dynoplan_moving_obstacles_from_constraints(planner, horizon: int):
    constraints = getattr(planner, "constraints", None)
    if constraints is None or len(constraints) == 0:
        return None

    dt = float(planner.minimum_time_step)
    roundoff_digits = int(getattr(planner, "roundoff_digits", 1))
    clearance = float(getattr(planner, "dynamic_agent_clearance", 0.0))
    moving = [[] for _ in range(horizon)]

    for step in range(horizon):
        t = round(step * dt, roundoff_digits)
        for i in range(len(constraints)):
            collision_keys, positions, other_radius = constraints[i]
            if len(collision_keys) == 0 or positions.shape[0] == 0:
                continue
            pos = _constraint_position_at_time(
                collision_keys,
                positions,
                t,
                dt,
                roundoff_digits,
            )
            if pos is None:
                continue
            moving[step].append(
                {
                    "type": "sphere",
                    "center": [float(pos[0]), float(pos[1])],
                    "size": [float(other_radius) + clearance],
                }
            )

    return moving


def _yaml_number_list(values):
    return "[" + ", ".join(f"{float(value):.17g}" for value in values) + "]"


def _append_obstacle_yaml(lines, obstacle, indent: int):
    prefix = " " * indent
    child_prefix = " " * (indent + 2)
    lines.append(f"{prefix}- type: {obstacle['type']}\n")
    lines.append(f"{child_prefix}center: {_yaml_number_list(obstacle['center'])}\n")
    lines.append(f"{child_prefix}size: {_yaml_number_list(obstacle['size'])}\n")


def _render_obstacles_yaml(obstacles, indent: int):
    lines = []
    for obstacle in obstacles:
        _append_obstacle_yaml(lines, obstacle, indent)
    return "".join(lines)


def _write_problem_yaml(path: Path, planner, options: CppDynoplanUnicycleOptimizerOptions, moving_obstacles):
    env_size = np.asarray(planner.env.size, dtype=np.float64)
    start = np.asarray(planner.start, dtype=np.float64)
    goal = np.asarray(planner.goal, dtype=np.float64)
    if goal.shape[0] == 2:
        goal = np.array([goal[0], goal[1], 0.0], dtype=np.float64)

    static_obstacles = _dynoplan_static_obstacles(planner.env)
    lines = [
        "environment:\n",
        "  min: [0, 0]\n",
        f"  max: {_yaml_number_list(env_size[:2])}\n",
    ]
    if static_obstacles:
        lines.append("  obstacles:\n")
        lines.append(_render_obstacles_yaml(static_obstacles, 4))
    else:
        lines.append("  obstacles: []\n")

    if moving_obstacles is not None:
        lines.append("  moving_obstacles:\n")
        static_moving_obstacle_text = (
            _render_obstacles_yaml(static_obstacles, 6)
            if options.include_static_in_moving_obstacles
            else ""
        )
        for obs_at_t in moving_obstacles:
            if not static_moving_obstacle_text and not obs_at_t:
                lines.append("    - []\n")
                continue
            lines.append("    -\n")
            if static_moving_obstacle_text:
                lines.append(static_moving_obstacle_text)
            for obstacle in obs_at_t:
                _append_obstacle_yaml(lines, obstacle, 6)

    lines.extend(
        [
            "robots:\n",
            f"  - type: {options.robot_type}\n",
            f"    start: {_yaml_number_list(start)}\n",
            f"    goal: {_yaml_number_list(goal)}\n",
        ]
    )

    path.write_text("".join(lines), encoding="utf-8")


def _write_trajectory_yaml(path: Path, xs: np.ndarray, us: np.ndarray):
    lines = [
        f"num_states: {int(xs.shape[0])}\n",
        f"num_actions: {int(us.shape[0])}\n",
        "states:\n",
    ]
    for row in xs:
        lines.append("  - [" + ", ".join(f"{float(value):.17g}" for value in row) + "]\n")

    lines.append("actions:\n")
    for row in us:
        lines.append("  - [" + ", ".join(f"{float(value):.17g}" for value in row) + "]\n")

    path.write_text("".join(lines), encoding="utf-8")


def _write_cfg_yaml(path: Path, options: CppDynoplanUnicycleOptimizerOptions, solver_id: int):
    data = {
        "solver_id": int(solver_id),
        "smooth_traj": bool(options.smooth_traj),
        "weight_goal": float(options.weight_goal),
        "collision_weight": float(options.collision_weight),
        "max_iter": int(options.max_iter),
    }
    data.update(options.extra_cfg)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run_cpp_optimizer(
    problem_yaml: Path,
    init_yaml: Path,
    cfg_yaml: Path,
    result_yaml: Path,
    options: CppDynoplanUnicycleOptimizerOptions,
):
    cmd = [
        str(options.main_optimization),
        "--env_file",
        str(problem_yaml),
        "--init_file",
        str(init_yaml),
        "--cfg_file",
        str(cfg_yaml),
        "--results_file",
        str(result_yaml),
        "--models_base_path",
        str(options.models_base_path) + "/",
    ]
    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(options.timeout_s),
        check=False,
    )
    return completed, time.perf_counter() - t0


def _parse_cpp_scalar(value: str):
    value = value.strip().strip('"')
    if value in ("", "~", "null", "None"):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _scan_result_metadata(path: Path):
    summary = {"feasible": False, "success": False, "cost": math.inf, "solver_iters": -1}
    if not path.exists():
        return summary

    in_info = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("result:") or stripped.startswith("trajs_opt:"):
                break
            if stripped == "info:":
                in_info = True
                continue
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            parsed = _parse_cpp_scalar(value)
            if key == "feasible":
                summary["feasible"] = bool(parsed)
            elif key == "success":
                summary["success"] = bool(parsed)
            elif key == "cost" and parsed is not None:
                summary["cost"] = float(parsed)
            elif in_info and key == "ddp_iterations" and parsed is not None:
                summary["solver_iters"] = int(parsed)
    return summary


def _parse_info_solver_iters(value: str):
    value = value.strip().strip('"')
    for part in value.split(";"):
        if part.startswith("ddp_iterations="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return -1
    return -1


def _parse_numeric_list_line(line: str):
    left = line.find("[")
    right = line.rfind("]")
    if left < 0 or right <= left:
        return None
    return np.fromstring(line[left + 1:right], sep=",", dtype=np.float64)


def _read_trajopt_trajectory(path: Path):
    if not path.exists():
        return {}, np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    metadata = {}
    states = []
    actions = []
    section = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped == "states:":
                section = "states"
                continue
            if stripped == "actions:":
                section = "actions"
                continue
            if stripped.startswith("- ["):
                row = _parse_numeric_list_line(stripped)
                if row is None:
                    continue
                if section == "states":
                    states.append(row)
                elif section == "actions":
                    actions.append(row)
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            if key == "info":
                metadata["solver_iters"] = _parse_info_solver_iters(value)
            elif key in ("feasible", "success", "cost"):
                parsed = _parse_cpp_scalar(value)
                if key == "cost" and parsed is not None:
                    metadata[key] = float(parsed)
                elif parsed is not None:
                    metadata[key] = bool(parsed)

    xs = np.asarray(states, dtype=np.float64) if states else np.empty((0, 3), dtype=np.float64)
    us = np.asarray(actions, dtype=np.float64) if actions else np.empty((0, 2), dtype=np.float64)
    return metadata, xs, us


def _read_cpp_result(path: Path):
    if not path.exists():
        return {}, np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    summary = _scan_result_metadata(path)
    traj_path = Path(str(path) + ".trajopt.yaml")
    traj_summary, xs, us = _read_trajopt_trajectory(traj_path)

    if xs.shape[0] == 0:
        # Fallback for older binaries that do not write the trajectory-only file.
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        traj = data
        if isinstance(data.get("trajs_opt"), list) and data["trajs_opt"]:
            traj = data["trajs_opt"][0]
        elif isinstance(data.get("result"), list) and data["result"]:
            traj = data["result"][0]
        xs = np.asarray(traj.get("states") or [], dtype=np.float64)
        us = np.asarray(traj.get("actions") or [], dtype=np.float64)
        if xs.ndim == 1 and xs.size == 0:
            xs = np.empty((0, 3), dtype=np.float64)
        if us.ndim == 1 and us.size == 0:
            us = np.empty((0, 2), dtype=np.float64)
        summary.update(
            {
                "feasible": bool(traj.get("feasible", data.get("feasible", summary["feasible"]))),
                "success": bool(data.get("success", traj.get("success", summary["success"]))),
                "cost": float(traj.get("cost", data.get("cost", summary["cost"]))),
                "solver_iters": int(data.get("ddp_iterations", traj.get("ddp_iterations", summary["solver_iters"])) or -1),
            }
        )
    else:
        summary.update({k: v for k, v in traj_summary.items() if v is not None})

    summary["raw"] = None
    return summary, xs, us

def _python_validate(planner, xs: np.ndarray, us: np.ndarray, feasibility_tolerance: float = 1e-3) -> bool:
    if xs.shape[0] == 0 or us.shape[0] != xs.shape[0] - 1:
        return False

    circles, rects = _extract_obstacle_arrays(planner.env)
    feasible, _, _, _ = _static_boundary_feasibility_numba(
        xs,
        circles,
        rects,
        np.asarray(planner.env.size, dtype=np.float64),
        float(planner.agent.radius),
        float(getattr(planner.env, "obstacle_buffer", 0.0)),
        float(getattr(planner.env, "boundary_buffer", 0.0)),
        float(feasibility_tolerance),
    )
    if not feasible:
        return False

    goal_dist = np.linalg.norm(xs[-1, :2] - np.asarray(planner.goal[:2], dtype=np.float64))
    if goal_dist > float(planner.goal_radius):
        return False

    constraints = getattr(planner, "constraints", None)
    if constraints is not None and len(constraints) > 0:
        return bool(
            constraint_satisfaction_numba(
                constraints,
                0,
                len(constraints),
                xs[1:],
                0.0,
                float(planner.agent.radius),
                float(planner.minimum_time_step),
                int(planner.agent.distance_metric_state_size),
                float(getattr(planner, "dynamic_agent_clearance", 0.0)),
                int(getattr(planner, "roundoff_digits", 1)),
            )
        )

    return True


def _failure_result(planner, *, source: str, optimizer_output_feasible: bool = False):
    highres_states, us_init, _ = planner.get_high_resolution_path_and_actions()
    return UnicycleTrajOptResult(
        success=False,
        feasible=False,
        optimizer_output_feasible=optimizer_output_feasible,
        source=source,
        xs=np.asarray(highres_states, dtype=np.float64),
        us=np.asarray(us_init, dtype=np.float64),
        cost=float("inf"),
        solver_iters=-1,
        path_view=None,
    )


def optimize_dbrrt_unicycle_path_with_cpp_dynoplan(
    planner,
    *,
    options: Optional[CppDynoplanUnicycleOptimizerOptions] = None,
) -> UnicycleTrajOptResult:
    if options is None:
        options = CppDynoplanUnicycleOptimizerOptions()

    if not options.main_optimization.exists():
        raise FileNotFoundError(f"C++ optimizer binary not found: {options.main_optimization}")
    if not options.models_base_path.exists():
        raise FileNotFoundError(f"Dynobench models path not found: {options.models_base_path}")

    highres_states, us_init, _ = planner.get_high_resolution_path_and_actions()
    if highres_states.shape[0] == 0:
        return _failure_result(planner, source="cpp_dynoplan_no_raw_path")
    if highres_states.shape[0] != us_init.shape[0] + 1:
        raise ValueError(
            f"Warm start length mismatch: states={highres_states.shape[0]}, controls={us_init.shape[0]}"
        )

    has_constraints = _has_constraints(planner)
    solver_id = options.solver_id_constrained if has_constraints else options.solver_id_static
    moving_obstacles = _dynoplan_moving_obstacles_from_constraints(planner, highres_states.shape[0])

    tmp_ctx = None
    if options.scratch_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="mrmp_cpp_dynoplan_")
        scratch = Path(tmp_ctx.name)
    else:
        call_id = next(_CPP_OPT_CALL_COUNTER)
        # Keep this identifier simple for now. KCBS node/agent labels can be
        # appended later if we need easier debugging across CBS expansions.
        scratch = Path(options.scratch_dir) / f"call_{call_id:06d}"
        scratch.mkdir(parents=True, exist_ok=True)

    try:
        total_t0 = time.perf_counter()
        problem_yaml = scratch / "problem.yaml"
        init_yaml = scratch / "init_traj.yaml"
        cfg_yaml = scratch / "optimizer_cfg.yaml"
        result_yaml = scratch / "result.yaml"

        write_t0 = time.perf_counter()
        _write_problem_yaml(problem_yaml, planner, options, moving_obstacles)
        _write_trajectory_yaml(init_yaml, highres_states, us_init)
        _write_cfg_yaml(cfg_yaml, options, solver_id)
        yaml_write_time = time.perf_counter() - write_t0

        completed, wall_time = _run_cpp_optimizer(
            problem_yaml,
            init_yaml,
            cfg_yaml,
            result_yaml,
            options,
        )
        read_t0 = time.perf_counter()
        summary, xs_out, us_out = _read_cpp_result(result_yaml)
        yaml_read_time = time.perf_counter() - read_t0
        optimizer_output_feasible = bool(completed.returncode == 0 and summary.get("feasible", False))

        feasible = optimizer_output_feasible
        python_validation_time = 0.0
        if feasible and options.validate_python:
            validate_t0 = time.perf_counter()
            feasible = _python_validate(planner, xs_out, us_out)
            python_validation_time = time.perf_counter() - validate_t0

        _CPP_OPT_TIMING_STATS["calls"] += 1
        if feasible:
            _CPP_OPT_TIMING_STATS["successes"] += 1
        _CPP_OPT_TIMING_STATS["yaml_write_time"] += yaml_write_time
        _CPP_OPT_TIMING_STATS["yaml_read_time"] += yaml_read_time
        _CPP_OPT_TIMING_STATS["subprocess_wall_time"] += wall_time
        _CPP_OPT_TIMING_STATS["python_validation_time"] += python_validation_time
        _CPP_OPT_TIMING_STATS["total_bridge_time"] += time.perf_counter() - total_t0

        if getattr(planner, "debug_flag", False):
            print(
                "C++ Dynoplan optimizer result: "
                f"returncode={completed.returncode}, solver_id={solver_id}, "
                f"cpp_feasible={optimizer_output_feasible}, python_feasible={feasible}, "
                f"states={xs_out.shape[0]}, controls={us_out.shape[0]}, wall={wall_time:.3f}s"
            )
            if completed.returncode != 0:
                print("\n".join(completed.stderr.splitlines()[-12:]))

        result = UnicycleTrajOptResult(
            success=bool(completed.returncode == 0 and summary.get("success", optimizer_output_feasible)),
            feasible=bool(feasible),
            optimizer_output_feasible=optimizer_output_feasible,
            source="cpp_dynoplan_optimized" if feasible else "cpp_dynoplan_failed",
            xs=xs_out if xs_out.shape[0] else highres_states,
            us=us_out if us_out.shape[0] else us_init,
            cost=float(summary.get("cost", math.inf)),
            solver_iters=int(summary.get("solver_iters", -1)),
        )
        if result.feasible:
            result.path_view = OptimizedTrajectoryView(planner, result.xs, result.us, result.cost)
        return result
    finally:
        if tmp_ctx is not None and not options.keep_files:
            tmp_ctx.cleanup()
