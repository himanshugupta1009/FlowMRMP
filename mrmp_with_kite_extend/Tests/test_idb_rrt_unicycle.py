"""
Single-agent iDb-RRT smoke test using the Dynoplan unicycle .msgpack primitives.

Run from the repo root:
    python Tests/test_idb_rrt_unicycle.py
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from printer import RRTPrinter
from test_db_rrt_unicycle import make_idb_planner, verify_rollout_consistency


def print_optimized_overlay(result, filename):
    planner = result.raw_planner
    raw_states, _, _ = planner.get_high_resolution_path_and_actions()
    opt_states = result.optimized_path_view.get_high_resolution_path_numpy_array()

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
    ax.plot(opt_xy[:, 0], opt_xy[:, 1], color="xkcd:bright blue", linewidth=2.0, label="iDb-RRT optimized")
    ax.legend(loc="best")
    fig.savefig(filename)
    plt.close(fig)
    print(f"\nSaved iDb-RRT optimized overlay to {filename}")


def run_search(idb_planner):
    t0 = time.time()
    result = idb_planner.plan_path()
    elapsed = time.time() - t0

    print("\n===== iDb-RRT result =====")
    print("solved:", result.solved)
    print("solved_raw:", result.solved_raw)
    print("outer_iters:", result.outer_iters)
    print("best_raw_cost:", result.best_raw_cost)
    print("best_opt_cost:", result.best_opt_cost)
    print("delta_history:", result.delta_history)
    print("primitive_history:", result.primitive_history)
    print("wall_time:", elapsed)

    assert result.outer_iters >= 1, "iDb-RRT did not run any outer iterations."
    assert result.solved_raw, "iDb-RRT did not find even a raw Db-RRT path."
    assert result.raw_planner is not None, "iDb-RRT returned no raw planner."

    if not result.solved:
        print("\nOptimization did not yield a feasible repaired path.")
        return None

    opt_states = result.optimized_path_view.get_high_resolution_path_numpy_array()
    assert opt_states.shape[0] > 1, "Optimized high-resolution path is empty."

    planner = result.raw_planner
    reached, dist = planner.reached_goal(opt_states[-1], planner.goal, planner.goal_radius, planner.agent)
    assert reached, f"Optimized final state does not reach the goal; dist={dist}"
    print(f"[OK] optimized final state reaches goal, dist={dist:.3f}")
    verify_rollout_consistency(result.optimized_path_view)

    return result


if __name__ == "__main__":
    idb = make_idb_planner(debug=False)
    result = run_search(idb)

    if result is not None:
        print("\nSmoke test passed.")

        raw_planner = result.raw_planner
        path_ids, _, _, _ = raw_planner.get_path()
        viz = RRTPrinter(raw_planner.env, raw_planner, path_ids)
        viz.print_rrt("media/idb_rrt_raw_graph.png", print_tree=True)
        print("\nSaved raw iDb-RRT search graph to media/idb_rrt_raw_graph.png")

        print_optimized_overlay(result, "media/idb_rrt_optimized_overlay.png")
