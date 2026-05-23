"""
Single-agent iDb-RRT smoke test scaffold for QuadCopter6D.

This uses the generic outer-loop `IdbRRTPlanner` together with the real
`DbRRTPlanner` search stage and the quadcopter edge-bundle primitives.

Run from the repo root:
    python Tests/test_idb_rrt_quadcopter6d.py
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Environments import CuboidEnvironment, SphericalObstacle3D
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent
from db_rrt import DbRRTPlanner
from db_optimize_quadcopter6d import (
    optimize_dbrrt_quadcopter6d_path,
    Quadcopter6DTrajOptOptions,
)
from idb_rrt import IdbRRTPlanner
from motion_primitives import (
    load_quadcopter6d_motion_primitives,
    transform_quadcopter6d_trajectory_numba,
)
from quadcopter_viz_3d import (
    save_quadcopter_rrt_tree_3d,
    save_quadcopter_path_3d,
    save_quadcopter_path_overlay_3d,
)


def make_agent(agent_id=0):
    return get_quadcopter_agent(agent_id)


def make_planner(*, debug=False, delta=0.4, max_motions=30000):
    obstacles = [
        SphericalObstacle3D(x=2.5, y=2.5, z=0.8, r=0.4),
        SphericalObstacle3D(x=5.5, y=3.5, z=1.6, r=0.6),
        SphericalObstacle3D(x=3.5, y=6.0, z=1.0, r=0.5),
    ]
    env = CuboidEnvironment(length=8.0, breadth=8.0, height=3.0, obs=obstacles)
    agent = make_agent(0)
    motion_primitives, kd_tree = load_quadcopter6d_motion_primitives(
        num_edges=max_motions, repo_root=REPO_ROOT
    )

    start = np.array((0.8, 0.8, 0.6, 0.0, 0.0, 0.0), dtype=np.float64)
    goal = np.array((7.0, 7.0, 1.8), dtype=np.float64)

    return DbRRTPlanner(
        start=start,
        goal=goal,
        goal_radius=0.35,
        env=env,
        agent=agent,
        motion_primitives=motion_primitives,
        alpha=0.5,
        delta=delta,
        minimum_time_step=0.1,
        max_iter=15000,
        planning_time=60.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        transform_trajectory_function=transform_quadcopter6d_trajectory_numba,
        motion_primitive_kd_tree=kd_tree,
        get_motion_primitive_kd_tree_query=agent.get_eb_kd_tree_query,
        max_candidate_motions_per_expand=12,
        allow_intermediate_goal=True,
        cost_delta_factor=0.0,
        goal_bias=0.15,
        goal_expand_mode="focused",
        random_expand_mode="randomized",
        udf_seed=77,
        debug_flag=debug,
        print_logs=True,
    )


def make_idb_planner(*, debug=False):
    def planner_factory(*, delta, max_motions):
        return make_planner(debug=debug, delta=delta, max_motions=max_motions)

    def optimizer_function(planner):
        return optimize_dbrrt_quadcopter6d_path(
            planner,
            options=Quadcopter6DTrajOptOptions(),
        )

    return IdbRRTPlanner(
        planner_factory=planner_factory,
        optimizer_function=optimizer_function,
        delta_0=0.4,
        delta_rate=0.9,
        num_primitives_0=150,
        num_primitives_rate=1.5,
        max_outer_iters=3,
        planning_time=90.0,
        print_logs=True,
    )


def verify_rollout_consistency(path_view, *, atol=1e-4, rtol=1e-4):
    _, states, controls, timesteps = path_view.get_path()
    reported_highres = path_view.get_high_resolution_path_numpy_array()

    assert states.shape == reported_highres.shape, (
        f"Path/state shape mismatch: get_path states={states.shape}, "
        f"highres={reported_highres.shape}"
    )
    assert np.allclose(states, reported_highres, atol=atol, rtol=rtol), (
        "Reported states from get_path() and get_high_resolution_path_numpy_array() differ"
    )

    rollout = np.empty_like(states)
    rollout[0] = np.asarray(path_view.start, dtype=np.float64)
    curr = rollout[0].copy()

    for i, (u, dt) in enumerate(zip(controls, timesteps)):
        next_state, _ = path_view.agent.get_next_state(curr, u, float(dt), num_steps=1)
        rollout[i + 1] = next_state
        curr = next_state

    max_abs_err = float(np.max(np.abs(rollout - states)))
    print("\n===== Rollout consistency =====")
    print("rollout shape:", rollout.shape)
    print("max_abs_err:", max_abs_err)
    print("rolled final state:", rollout[-1])
    print("reported final state:", states[-1])

    assert np.allclose(rollout, states, atol=atol, rtol=rtol), (
        f"Returned path is not rollout-consistent; max_abs_err={max_abs_err}"
    )
    print(f"[OK] returned states are rollout-consistent, max_abs_err={max_abs_err:.3e}")


def run_search(idb_planner):
    t0 = time.time()
    result = idb_planner.plan_path()
    elapsed = time.time() - t0

    print("\n===== QuadCopter6D iDb-RRT result =====")
    print("solved:", result.solved)
    print("solved_raw:", result.solved_raw)
    print("outer_iters:", result.outer_iters)
    print("best_raw_cost:", result.best_raw_cost)
    print("best_opt_cost:", result.best_opt_cost)
    print("delta_history:", result.delta_history)
    print("primitive_history:", result.primitive_history)
    print("wall_time:", elapsed)

    assert result.outer_iters >= 1, "iDb-RRT did not run any outer iterations."
    assert result.solved_raw, "iDb-RRT did not find even a raw Db-RRT path for QuadCopter6D."
    assert result.raw_planner is not None, "iDb-RRT returned no raw planner."

    raw_planner = result.raw_planner
    highres, _, primitive_timesteps = raw_planner.get_high_resolution_path_and_actions()
    print("\n===== Raw QuadCopter6D Db-RRT path =====")
    print("highres shape:", highres.shape)
    print("final highres state:", highres[-1])
    print("primitive count:", len(primitive_timesteps))

    reached, dist = raw_planner.reached_goal(
        highres[-1],
        raw_planner.goal,
        raw_planner.goal_radius,
        raw_planner.agent,
    )
    relaxed_dist = np.linalg.norm(highres[-1][:3] - raw_planner.goal[:3])
    assert reached or relaxed_dist <= raw_planner.delta, (
        "Raw QuadCopter6D path is neither inside the goal nor within delta "
        f"of it; dist={dist}, relaxed_dist={relaxed_dist}"
    )
    print(f"[OK] raw final state reaches relaxed goal condition, dist={relaxed_dist:.3f}")

    if result.optimized_result is not None:
        opt = result.optimized_result
        print("\n===== QuadCopter6D optimization result =====")
        print("success:", opt.success)
        print("feasible:", opt.feasible)
        print("solver_iters:", opt.solver_iters)
        print("cost:", opt.cost)
        print("final optimized state:", opt.xs[-1])

    if result.solved:
        opt_states = result.optimized_path_view.get_high_resolution_path_numpy_array()
        reached, dist = raw_planner.reached_goal(
            opt_states[-1],
            raw_planner.goal,
            raw_planner.goal_radius,
            raw_planner.agent,
        )
        assert reached, f"Optimized final state does not reach the goal; dist={dist}"
        print(f"[OK] optimized final state reaches goal, dist={dist:.3f}")
        verify_rollout_consistency(result.optimized_path_view)

    return result


if __name__ == "__main__":
    idb = make_idb_planner(debug=False)
    result = run_search(idb)

    if result is not None and result.raw_planner is not None:
        raw_planner = result.raw_planner
        save_quadcopter_rrt_tree_3d(
            raw_planner,
            "media/idb_rrt_quadcopter6d_raw_tree_3d.png",
            print_tree=True,
        )
        print("\nSaved 3D raw tree to media/idb_rrt_quadcopter6d_raw_tree_3d.png")

        raw_states, _, _ = raw_planner.get_high_resolution_path_and_actions()
        save_quadcopter_path_3d(
            env=raw_planner.env,
            start=raw_planner.start,
            goal=raw_planner.goal,
            goal_radius=raw_planner.goal_radius,
            highres_states=raw_states,
            filename="media/idb_rrt_quadcopter6d_raw_path_3d.png",
            agent_radius=raw_planner.agent.radius,
            path_color="xkcd:orange",
            path_label="Raw Db-RRT path",
        )
        print("Saved 3D raw path to media/idb_rrt_quadcopter6d_raw_path_3d.png")

        if result.optimized_path_view is not None:
            save_quadcopter_path_overlay_3d(
                raw_planner,
                result.optimized_path_view,
                "media/idb_rrt_quadcopter6d_optimized_overlay_3d.png",
            )
            print("Saved 3D optimized overlay to media/idb_rrt_quadcopter6d_optimized_overlay_3d.png")
