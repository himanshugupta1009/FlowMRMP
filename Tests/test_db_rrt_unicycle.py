"""
Single-agent Db-RRT smoke test using the Dynoplan unicycle .msgpack primitives.

Run from the repo root:
    python Tests/test_db_rrt_unicycle.py
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Environments import RectangleObstacle2D, SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from db.db_rrt import DbRRTPlanner
from db.db_optimize_unicycle import optimize_dbrrt_unicycle_path, UnicycleTrajOptOptions
from db.idb_rrt import IdbRRTPlanner
from motion_primitives import (
    load_unicycle_motion_primitives,
    transform_unicycle_trajectory_numba,
)
from printer import RRTPrinter
from utils import verify_rollout_consistency

def make_agent(agent_id=0):
    return get_unicycle_agent(agent_id)


def make_planner(*, debug=False, delta=0.3, max_motions=30000,
                 max_candidate_motions_per_expand=100, udf_seed=11):
    
    env = SquareEnvironment(10, 10, [CircularObstacle2D(3.0, 5.0, 1.5), CircularObstacle2D(8.0, 5.0, 1.5)], obs_buffers=True)
    start = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    goal = np.array([8.0, 8.0], dtype=np.float64)
    goal_radius = 0.25

    env = SquareEnvironment(20, 20, [CircularObstacle2D(9.0, 9.0, 1.5)], obs_buffers=True)
    start = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    goal = np.array([18.0, 18.0], dtype=np.float64)
    goal_radius = 0.25

    # obstacles = [
    #     RectangleObstacle2D(4.5, 3.0, 0.2, 3.2),
    #     RectangleObstacle2D(3.0, 1.5, 3.2, 0.2),
    #     RectangleObstacle2D(3.0, 4.5, 3.2, 0.2),
    #     RectangleObstacle2D(1.5, 4.05, 0.2, 1.1),
    #     RectangleObstacle2D(1.5, 1.95, 0.2, 1.1),
    # ]
    # env = SquareEnvironment(10.0, 10.0, obstacles, obs_buffers=False)
    # start = np.array((3.8, 3.0, 0.0), dtype=np.float64)   # x, y, theta
    # goal = np.array((9.2, 9.2), dtype=np.float64)         # x, y
    # goal_radius = 0.5


    agent = make_agent(0)
    motion_primitives, kd_tree = load_unicycle_motion_primitives(
        num_edges=max_motions, dt=0.1, repo_root=REPO_ROOT
    )


    planner = DbRRTPlanner(
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        motion_primitives=motion_primitives,
        alpha=0.5,
        delta=delta,
        minimum_time_step=0.1,
        max_iter=10000,
        planning_time=120.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        transform_trajectory_function=transform_unicycle_trajectory_numba,
        motion_primitive_kd_tree=kd_tree,
        get_motion_primitive_kd_tree_query=agent.get_eb_kd_tree_query,
        max_candidate_motions_per_expand=max_candidate_motions_per_expand,
        allow_intermediate_goal=True,
        cost_delta_factor=0.0,
        goal_bias=0.1,
        goal_expand_mode="focused",
        random_expand_mode="randomized",
        udf_seed=udf_seed,
        debug_flag=debug,
        print_logs=True,
    )
    return planner


def make_idb_planner(*, debug=False):
    delta_0 = 0.5
    delta_rate = 0.9
    num_primitives_0 = 5000
    num_primitives_rate = 1.5
    max_outer_iters = 5
    max_motion_budget = num_primitives_0
    curr_budget = num_primitives_0
    for _ in range(1, max_outer_iters):
        next_budget = int(np.ceil(curr_budget * num_primitives_rate))
        curr_budget = max(curr_budget + 1, next_budget)
        max_motion_budget = max(max_motion_budget, curr_budget)

    def planner_factory(*, delta, max_motions):
        return make_planner(debug=debug, delta=delta, max_motions=max_motion_budget)

    def optimizer_function(planner):
        return optimize_dbrrt_unicycle_path(
            planner,
            options=UnicycleTrajOptOptions(),
        )

    return IdbRRTPlanner(
        planner_factory=planner_factory,
        optimizer_function=optimizer_function,
        delta_0=delta_0,
        delta_rate=delta_rate,
        num_primitives_0=num_primitives_0,
        num_primitives_rate=num_primitives_rate,
        max_outer_iters=max_outer_iters,
        planning_time=120.0,
        print_logs=True,
    )


def run_search(planner):
    t0 = time.time()
    planner.plan_path()
    elapsed = time.time() - t0

    print("\n===== Db-RRT result =====")
    print("path_found:", planner.path_found)
    print("num_nodes:", len(planner.tree.nodes))
    print("path_cost:", planner.path_cost)
    print("path_time:", planner.path_time)
    print("wall_time:", elapsed)

    assert len(planner.tree.nodes) > 1, "Planner did not add any nodes beyond the start."

    if not planner.path_found:
        print("\nNo path found in this smoke test with the Dynoplan primitive file.")
        print("Things to try:")
        print("  - increase delta, e.g., 1.2")
        print("  - increase max_candidate_motions_per_expand")
        print("  - increase num_edges in load_unicycle_primitives")
        print("  - make the goal easier / larger goal_radius")
        return False

    ids, states, action_sequences, timesteps = planner.get_path_to_node_id(planner.goal_node_id)
    highres, highres_controls, highres_timesteps = planner.get_high_resolution_path_and_actions()

    print("\n===== Path summary =====")
    print("coarse node ids:", ids)
    print("coarse states shape:", states.shape)
    print("first action sequence shape:", action_sequences[0].shape)
    print("timesteps:", timesteps)
    print("highres shape:", highres.shape)
    print("highres controls shape:", highres_controls.shape)
    print("final highres state:", highres[-1])

    assert highres.shape[0] > 1, "High-resolution path is empty or has only start."
    reached, dist = planner.reached_goal(highres[-1], planner.goal, planner.goal_radius, planner.agent)
    relaxed_dist = np.linalg.norm(highres[-1][:2] - planner.goal[:2])
    assert reached or relaxed_dist <= planner.delta, (
        f"Final state is neither inside goal nor within delta; dist={dist}, relaxed_dist={relaxed_dist}"
    )
    print(f"[OK] final state reaches relaxed goal condition, dist={relaxed_dist:.3f}")
    assert highres.shape[0] == highres_controls.shape[0] + 1
    assert highres_controls.shape[0] == highres_timesteps.shape[0]
    verify_rollout_consistency(planner)
    return True


def maybe_optimize(planner):
    try:
        t0 = time.time()
        result = optimize_dbrrt_unicycle_path(
            planner,
            options=UnicycleTrajOptOptions(),
        )
        elapsed = time.time() - t0
    except ImportError as exc:
        print(f"\nSkipping Crocoddyl repair stage: {exc}")
        print("If Crocoddyl is installed, this test can also optimize the Db-RRT warm start.")
        return

    print("\n===== Crocoddyl repair result =====")
    print("success:", result.success)
    print("feasible:", result.feasible)
    print("optimizer_output_feasible:", result.optimizer_output_feasible)
    print("source:", result.source)
    print("solver_iters:", result.solver_iters)
    print("cost:", result.cost)
    print("wall_time:", elapsed)
    print("final state:", result.xs[-1])
    if result.feasible:
        verify_rollout_consistency(result.path_view)
    else:
        print("optimized path is not feasible; keeping raw Db-RRT path")
    return result


def print_optimized_overlay(planner, opt_result, filename):
    raw_states, _, _ = planner.get_high_resolution_path_and_actions()
    opt_states = opt_result.path_view.get_high_resolution_path_numpy_array()

    fig, ax = plt.subplots()
    ax.set_xlim(0, planner.env.size[0])
    ax.set_ylim(0, planner.env.size[1])
    ax.set_aspect("equal", adjustable="box")

    RRTPrinter.print_obs(ax, planner.env.obstacles, planner.env.obstacle_buffer)
    RRTPrinter.print_goal(ax, planner.goal, planner.goal_radius)
    RRTPrinter.print_start(ax, planner.start, planner.agent, pcol="black")

    raw_xy = np.asarray(raw_states[:, :2], dtype=np.float64)
    opt_xy = np.asarray(opt_states[:, :2], dtype=np.float64)

    ax.plot(raw_xy[:, 0], raw_xy[:, 1], color="xkcd:light grey", linewidth=2.0, label="Db-RRT warm start")
    ax.plot(opt_xy[:, 0], opt_xy[:, 1], color="xkcd:bright blue", linewidth=2.0, label="Optimized trajectory")

    ax.scatter(raw_xy[0, 0], raw_xy[0, 1], color="black", s=20)
    ax.scatter(opt_xy[-1, 0], opt_xy[-1, 1], color="xkcd:bright blue", s=20)

    ax.legend(loc="best")
    fig.savefig(filename)
    plt.close(fig)
    print(f"\nSaved optimized overlay to {filename}")


if __name__ == "__main__":
    planner = make_planner(debug=False, udf_seed=np.random.randint(1e3))
    ok = run_search(planner)
    if ok:
        opt_result = maybe_optimize(planner)
        print("\nSmoke test passed.")
        path_ids, _, _, _ = planner.get_path()
        viz = RRTPrinter(planner.env, planner, path_ids)
        viz.print_rrt("media/db_rrt_graph.png", print_tree=True)
        print("\nSaved visualization to media/db_rrt_graph.png")
        if opt_result is not None and opt_result.feasible:
            opt_dict = opt_result.path_view.get_high_resolution_path()
            print("optimized path samples:", len(opt_dict))
            print_optimized_overlay(planner, opt_result, "media/db_rrt_optimized_overlay.png")
