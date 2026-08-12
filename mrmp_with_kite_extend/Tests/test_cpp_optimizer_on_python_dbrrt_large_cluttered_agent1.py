"""
Benchmark bridge: feed Python db-RRT raw trajectories to the db-CBS C++
Dynoplan optimizer for Large Cluttered unicycle agent 1, and compare that with
running the full C++ db-RRT executable on the same single-agent problem.

This is intentionally a probe, not a KCBS integration.

Run from the repo root:
    python3 Tests/test_cpp_optimizer_on_python_dbrrt_large_cluttered_agent1.py --runs 5
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "Tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from Environments import SquareEnvironment
from mapf_env_square_agent_unicycle import (
    get_constrained_db_rrt_planner_unicycle,
    get_unicycle_agent,
)
from test_db_rrt_unicycle_large_cluttered_agent1 import GOALS, OBSTACLES, STARTS


AGENT_ID = 1
GOAL_RADIUS = 0.5

DBCBS_ROOT = Path("/home/himanshu/Documents/Research/dbcbs_stuff/db-CBS")
MAIN_OPTIMIZATION = DBCBS_ROOT / "buildRelease/dynoplan/main_optimization"
MAIN_DBRRT = DBCBS_ROOT / "buildRelease/dynoplan/main_dbrrt"
MODELS_BASE = DBCBS_ROOT / "dynoplan/dynobench/models"
MOTIONS_FILE = (
    DBCBS_ROOT
    / "dynoplan/dynomotions/"
    "unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.small5000.msgpack"
)

SCRATCH_DIR = Path("/tmp/mrmp_cpp_optimizer_bridge_large_cluttered_agent1")
PROBLEM_YAML = SCRATCH_DIR / "large_cluttered_agent1_problem.yaml"
OPT_CFG_YAML = SCRATCH_DIR / "cpp_optimizer_free_time_cfg.yaml"
DBRRT_CFG_YAML = SCRATCH_DIR / "cpp_dbrrt_free_time_cfg.yaml"


def _float_list(values):
    return [float(v) for v in values]


def write_problem_yaml(path, moving_obstacles=None):
    data = {
        "environment": {
            "min": [0.0, 0.0],
            "max": [40.0, 40.0],
            "obstacles": [
                {
                    "type": "box",
                    "center": [float(obs.x), float(obs.y)],
                    "size": [float(obs.w), float(obs.h)],
                }
                for obs in OBSTACLES
            ],
        },
        "robots": [
            {
                "type": "unicycle1_sphere_v0",
                "start": _float_list(STARTS[AGENT_ID]),
                "goal": _float_list([GOALS[AGENT_ID][0], GOALS[AGENT_ID][1], 0.0]),
            }
        ],
    }
    if moving_obstacles is not None:
        data["environment"]["moving_obstacles"] = moving_obstacles
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def generate_moving_obstacles(
    states,
    count,
    mode,
    seed,
    radius,
    duration_min,
    duration_max,
    horizon_scale,
):
    """Return Dynobench environment.moving_obstacles.

    Format: moving_obstacles[t] is the list of obstacle objects at optimizer
    time index t. Empty lists mean no dynamic obstacle at that step.
    """
    if count <= 0:
        return None

    rng = np.random.default_rng(seed)
    ref_dt = 0.1
    raw_len = int(states.shape[0])
    horizon = max(raw_len + 1, int(np.ceil(raw_len * horizon_scale)))
    moving = [[] for _ in range(horizon)]

    for _ in range(count):
        duration = float(rng.uniform(duration_min, duration_max))
        duration_steps = max(1, int(round(duration / ref_dt)))
        t0 = int(rng.integers(1, max(2, min(raw_len - duration_steps - 1, horizon - duration_steps - 1))))
        t1 = min(horizon - 1, t0 + duration_steps)

        if mode == "random":
            p0 = rng.uniform([0.5, 0.5], [39.5, 39.5])
            p1 = rng.uniform([0.5, 0.5], [39.5, 39.5])
        elif mode == "far-corner":
            p0 = np.array([38.5, 1.0])
            p1 = np.array([39.0, 1.5])
        elif mode == "near-path":
            center_idx = min(raw_len - 2, max(1, t0 + duration_steps // 2))
            center = np.asarray(states[center_idx, :2], dtype=np.float64)
            tangent = np.asarray(states[min(raw_len - 1, center_idx + 1), :2] - states[max(0, center_idx - 1), :2])
            norm = np.linalg.norm(tangent)
            if norm < 1e-9:
                tangent = np.array([1.0, 0.0])
            else:
                tangent = tangent / norm
            normal = np.array([-tangent[1], tangent[0]])
            crossing_half_width = rng.uniform(0.6, 1.2)
            along_jitter = tangent * rng.uniform(-0.2, 0.2)
            p0 = center + along_jitter - normal * crossing_half_width
            p1 = center + along_jitter + normal * crossing_half_width
            p0 = np.clip(p0, [0.5, 0.5], [39.5, 39.5])
            p1 = np.clip(p1, [0.5, 0.5], [39.5, 39.5])
        else:
            raise ValueError(f"unknown moving obstacle mode: {mode}")

        for t in range(t0, t1 + 1):
            alpha = 0.0 if t1 == t0 else (t - t0) / float(t1 - t0)
            pos = (1.0 - alpha) * p0 + alpha * p1
            moving[t].append(
                {
                    "type": "sphere",
                    "center": [float(pos[0]), float(pos[1])],
                    "size": [float(radius)],
                }
            )

    return moving


def add_static_obstacles_to_moving_obstacles(moving_obstacles):
    static_obs = [
        {
            "type": "box",
            "center": [float(obs.x), float(obs.y)],
            "size": [float(obs.w), float(obs.h)],
        }
        for obs in OBSTACLES
    ]
    return [static_obs + list(obs_at_t) for obs_at_t in moving_obstacles]


def write_raw_trajectory_yaml(path, states, controls):
    data = {
        "num_states": int(states.shape[0]),
        "num_actions": int(controls.shape[0]),
        "states": [_float_list(row) for row in states],
        "actions": [_float_list(row) for row in controls],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_free_time_optimizer_cfg(path, solver_id=1):
    data = {
        "solver_id": int(solver_id),
        "smooth_traj": True,
        "weight_goal": 200.0,
        "collision_weight": 100.0,
        "max_iter": 50,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_cpp_dbrrt_cfg(path):
    data = {
        "do_optimization": True,
        "solver_id": 1,
        "smooth_traj": True,
        "use_connect": False,
        "choose_first_motion_valid": True,
        "timelimit": 120,
        "max_expands": 10000,
        "max_motions": 5000,
        "cost_bound": 1000000,
        "delta": 0.3,
        "goal_region": GOAL_RADIUS,
        "goal_bias": 0.1,
        "use_nigh_nn": True,
        "seed": 0,
        "motionsFile": str(MOTIONS_FILE),
        "weight_goal": 200.0,
        "collision_weight": 100.0,
        "max_iter": 50,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_python_raw_dbrrt():
    env = SquareEnvironment(40.0, 40.0, OBSTACLES, obs_buffers=False)
    agent = get_unicycle_agent(AGENT_ID)
    planner = get_constrained_db_rrt_planner_unicycle(
        STARTS[AGENT_ID],
        GOALS[AGENT_ID],
        GOAL_RADIUS,
        agent,
        env,
        use_optimizer=False,
    )
    planner.print_logs = False
    planner.debug_flag = False

    t0 = time.perf_counter()
    planner.plan_path()
    elapsed = time.perf_counter() - t0
    states, controls, timesteps = planner.get_high_resolution_path_and_actions()

    return planner, states, controls, timesteps, elapsed


def run_cpp_optimizer(raw_traj_yaml, result_yaml):
    cmd = [
        str(MAIN_OPTIMIZATION),
        "--env_file",
        str(PROBLEM_YAML),
        "--init_file",
        str(raw_traj_yaml),
        "--cfg_file",
        str(OPT_CFG_YAML),
        "--results_file",
        str(result_yaml),
        "--models_base_path",
        str(MODELS_BASE) + "/",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def run_cpp_dbrrt(result_yaml):
    cmd = [
        str(MAIN_DBRRT),
        "--env_file",
        str(PROBLEM_YAML),
        "--cfg_file",
        str(DBRRT_CFG_YAML),
        "--results_file",
        str(result_yaml),
        "--models_base_path",
        str(MODELS_BASE) + "/",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def dynamic_sphere_collision_stats(traj_yaml_path, moving_obstacles, agent_radius=0.3):
    if moving_obstacles is None or not traj_yaml_path.exists():
        return None
    data = yaml.safe_load(traj_yaml_path.read_text(encoding="utf-8")) or {}
    traj = data
    if isinstance(data.get("result"), list) and data["result"]:
        traj = data["result"][0]
    elif isinstance(data.get("trajs_opt"), list) and data["trajs_opt"]:
        traj = data["trajs_opt"][0]
    states = traj.get("states") or []
    collisions = 0
    min_margin = float("inf")
    for t, state in enumerate(states):
        if t >= len(moving_obstacles):
            break
        px, py = float(state[0]), float(state[1])
        for obs in moving_obstacles[t]:
            if obs.get("type") != "sphere":
                continue
            ox, oy = obs["center"][:2]
            radius = float(obs["size"][0])
            margin = float(np.hypot(px - ox, py - oy) - (agent_radius + radius))
            min_margin = min(min_margin, margin)
            if margin < 0.0:
                collisions += 1
    if min_margin == float("inf"):
        min_margin = None
    return {"dynamic_sphere_collisions": collisions, "dynamic_sphere_min_margin": min_margin}


def summarize_result_yaml(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    traj = data
    if isinstance(data.get("trajs_opt"), list) and data["trajs_opt"]:
        traj = data["trajs_opt"][0]
    return {
        "solved": data.get("solved"),
        "solved_raw": data.get("solved_raw"),
        "success": data.get("success"),
        "feasible": traj.get("feasible", data.get("feasible")),
        "cost": traj.get("cost", data.get("cost")),
        "num_states": traj.get("num_states", data.get("num_states")),
        "goal_distance": traj.get("goal_distance", data.get("goal_distance")),
        "max_collision": traj.get("max_collision", data.get("max_collision")),
        "max_jump": traj.get("max_jump", data.get("max_jump")),
        "ddp_iterations": data.get("ddp_iterations", traj.get("ddp_iterations")),
        "ddp_time": data.get("ddp_time", traj.get("ddp_time")),
        "info_ddp_time": (data.get("info") or {}).get("ddp_time"),
        "info_time_ddp_total": (data.get("info") or {}).get("time_ddp_total"),
        "info_time_raw": (data.get("info") or {}).get("time_raw"),
    }


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _std(values):
    return float(np.std(values, ddof=0)) if values else float("nan")


def print_table(title, rows):
    print(f"\n===== {title} =====")
    for name, values in rows:
        if not values:
            print(f"{name}: no samples")
            continue
        print(
            f"{name}: n={len(values)} mean={_mean(values):.6f}s "
            f"std={_std(values):.6f}s min={min(values):.6f}s max={max(values):.6f}s"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--skip-full-cpp", action="store_true")
    parser.add_argument("--moving-obstacles", type=int, default=0)
    parser.add_argument("--moving-mode", choices=["random", "near-path", "far-corner"], default="near-path")
    parser.add_argument("--moving-seed", type=int, default=12345)
    parser.add_argument("--moving-radius", type=float, default=0.3)
    parser.add_argument("--moving-duration-min", type=float, default=1.0)
    parser.add_argument("--moving-duration-max", type=float, default=2.0)
    parser.add_argument("--moving-horizon-scale", type=float, default=2.0)
    parser.add_argument("--optimizer-solver-id", type=int, default=1)
    parser.add_argument("--moving-include-static", action="store_true")
    args = parser.parse_args(argv)

    if not MAIN_OPTIMIZATION.exists():
        raise FileNotFoundError(f"C++ optimizer binary not found: {MAIN_OPTIMIZATION}")
    if not MAIN_DBRRT.exists():
        raise FileNotFoundError(f"C++ db-RRT binary not found: {MAIN_DBRRT}")
    if not MODELS_BASE.exists():
        raise FileNotFoundError(f"Dynobench models path not found: {MODELS_BASE}")
    if not MOTIONS_FILE.exists():
        raise FileNotFoundError(f"C++ motions file not found: {MOTIONS_FILE}")

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    write_problem_yaml(PROBLEM_YAML)
    write_free_time_optimizer_cfg(OPT_CFG_YAML, solver_id=args.optimizer_solver_id)
    write_cpp_dbrrt_cfg(DBRRT_CFG_YAML)

    print("===== Python raw db-RRT -> C++ Dynoplan optimizer benchmark =====")
    print("agent_id:", AGENT_ID)
    print("start:", STARTS[AGENT_ID])
    print("goal:", GOALS[AGENT_ID])
    print("scratch_dir:", SCRATCH_DIR)
    print("runs:", args.runs)
    print("moving_obstacles:", args.moving_obstacles)
    print("moving_mode:", args.moving_mode)
    print("optimizer_solver_id:", args.optimizer_solver_id)
    print("moving_include_static:", args.moving_include_static)

    py_raw_times = []
    yaml_write_times = []
    cpp_opt_wall_times = []
    cpp_opt_successes = 0

    for run_idx in range(args.runs):
        raw_traj_yaml = SCRATCH_DIR / f"python_dbrrt_raw_traj_run{run_idx}.yaml"
        opt_result_yaml = SCRATCH_DIR / f"cpp_optimizer_free_time_result_run{run_idx}.yaml"

        planner, states, controls, timesteps, raw_elapsed = run_python_raw_dbrrt()
        py_raw_times.append(raw_elapsed)
        print(f"\n--- Bridge run {run_idx + 1}/{args.runs} ---")
        print("python_path_found:", planner.path_found)
        print("python_raw_wall_time:", raw_elapsed)
        print("python_raw_path_cost:", planner.path_cost)
        print("states:", states.shape, "controls:", controls.shape, "timesteps:", timesteps.shape)
        if not planner.path_found:
            continue

        final_dist = np.linalg.norm(states[-1, :2] - np.asarray(GOALS[AGENT_ID], dtype=np.float64))
        print("python_final_goal_distance:", final_dist)

        moving_obstacles = None
        if args.moving_obstacles > 0:
            moving_obstacles = generate_moving_obstacles(
                states,
                count=args.moving_obstacles,
                mode=args.moving_mode,
                seed=args.moving_seed + run_idx,
                radius=args.moving_radius,
                duration_min=args.moving_duration_min,
                duration_max=args.moving_duration_max,
                horizon_scale=args.moving_horizon_scale,
            )
            if args.moving_include_static:
                moving_obstacles = add_static_obstacles_to_moving_obstacles(moving_obstacles)
            active_steps = sum(1 for obs_at_t in moving_obstacles if obs_at_t)
            total_obs_instances = sum(len(obs_at_t) for obs_at_t in moving_obstacles)
            print("moving_obstacle_horizon:", len(moving_obstacles))
            print("moving_obstacle_active_steps:", active_steps)
            print("moving_obstacle_instances:", total_obs_instances)

        t0 = time.perf_counter()
        write_problem_yaml(PROBLEM_YAML, moving_obstacles=moving_obstacles)
        write_raw_trajectory_yaml(raw_traj_yaml, states, controls)
        yaml_write_elapsed = time.perf_counter() - t0
        yaml_write_times.append(yaml_write_elapsed)

        t0 = time.perf_counter()
        completed = run_cpp_optimizer(raw_traj_yaml, opt_result_yaml)
        cpp_elapsed = time.perf_counter() - t0
        cpp_opt_wall_times.append(cpp_elapsed)
        summary = summarize_result_yaml(opt_result_yaml)
        if completed.returncode == 0 and bool(summary.get("feasible")):
            cpp_opt_successes += 1

        print("yaml_write_time:", yaml_write_elapsed)
        print("cpp_optimizer_returncode:", completed.returncode)
        print("cpp_optimizer_wall_time:", cpp_elapsed)
        print("cpp_optimizer_feasible:", summary.get("feasible"))
        print("cpp_optimizer_cost:", summary.get("cost"))
        print("cpp_optimizer_num_states:", summary.get("num_states"))
        print("cpp_optimizer_goal_distance:", summary.get("goal_distance"))
        print("cpp_optimizer_max_collision:", summary.get("max_collision"))
        print("cpp_optimizer_info_time_ddp_total_ms:", summary.get("info_time_ddp_total"))
        if moving_obstacles is not None:
            raw_dyn = dynamic_sphere_collision_stats(raw_traj_yaml, moving_obstacles)
            opt_dyn = dynamic_sphere_collision_stats(opt_result_yaml, moving_obstacles)
            print("raw_dynamic_sphere_collisions:", raw_dyn.get("dynamic_sphere_collisions") if raw_dyn else None)
            print("raw_dynamic_sphere_min_margin:", raw_dyn.get("dynamic_sphere_min_margin") if raw_dyn else None)
            print("opt_dynamic_sphere_collisions:", opt_dyn.get("dynamic_sphere_collisions") if opt_dyn else None)
            print("opt_dynamic_sphere_min_margin:", opt_dyn.get("dynamic_sphere_min_margin") if opt_dyn else None)
        if completed.returncode != 0:
            print("cpp_optimizer_stderr_tail:")
            print("\n".join(completed.stderr.splitlines()[-12:]))

    print_table(
        "Bridge Timing Summary",
        [
            ("python_raw_dbrrt_wall", py_raw_times),
            ("yaml_raw_trajectory_write", yaml_write_times),
            ("cpp_optimizer_subprocess_wall", cpp_opt_wall_times),
            (
                "bridge_total_after_raw_path",
                [a + b for a, b in zip(yaml_write_times, cpp_opt_wall_times)],
            ),
        ],
    )
    print("cpp_optimizer_successes:", f"{cpp_opt_successes}/{args.runs}")

    if args.skip_full_cpp or args.moving_obstacles > 0:
        if args.moving_obstacles > 0:
            print("\nSkipping full C++ db-RRT comparison for moving-obstacle runs; the synthetic moving env is regenerated per bridge run.")
        return cpp_opt_successes == args.runs

    cpp_dbrrt_wall_times = []
    cpp_dbrrt_successes = 0
    for run_idx in range(args.runs):
        dbrrt_result_yaml = SCRATCH_DIR / f"cpp_dbrrt_free_time_result_run{run_idx}.yaml"
        t0 = time.perf_counter()
        completed = run_cpp_dbrrt(dbrrt_result_yaml)
        cpp_elapsed = time.perf_counter() - t0
        cpp_dbrrt_wall_times.append(cpp_elapsed)
        summary = summarize_result_yaml(dbrrt_result_yaml)
        if completed.returncode == 0 and bool(summary.get("solved")):
            cpp_dbrrt_successes += 1

        print(f"\n--- Full C++ db-RRT run {run_idx + 1}/{args.runs} ---")
        print("cpp_dbrrt_returncode:", completed.returncode)
        print("cpp_dbrrt_wall_time:", cpp_elapsed)
        print("cpp_dbrrt_solved:", summary.get("solved"))
        print("cpp_dbrrt_solved_raw:", summary.get("solved_raw"))
        print("cpp_dbrrt_cost:", summary.get("cost"))
        print("cpp_dbrrt_num_states:", summary.get("num_states"))
        print("cpp_dbrrt_info_time_raw_ms:", summary.get("info_time_raw"))
        print("cpp_dbrrt_info_time_ddp_total_ms:", summary.get("info_time_ddp_total"))
        if completed.returncode != 0:
            print("cpp_dbrrt_stderr_tail:")
            print("\n".join(completed.stderr.splitlines()[-12:]))

    print_table(
        "Full C++ db-RRT Timing Summary",
        [("cpp_dbrrt_subprocess_wall", cpp_dbrrt_wall_times)],
    )
    print("cpp_dbrrt_successes:", f"{cpp_dbrrt_successes}/{args.runs}")
    return cpp_opt_successes == args.runs and cpp_dbrrt_successes == args.runs


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
