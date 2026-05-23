"""
Convenience 3D visualization helpers for QuadCopter6D planners and paths.

These wrappers sit on top of `printer_3d.py` so quadcopter planners can be
visualized from tests and trial scripts with a small, consistent API.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapf_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from printer_3d import RRTPrinter3d


def save_quadcopter_rrt_tree_3d(planner, filename: str, *, print_tree: bool = True):
    """
    Save a 3D rendering of a planner's tree and selected raw path.
    """
    if not getattr(planner, "path_found", False):
        raise ValueError("Planner has no path to visualize")

    path_ids, _, _, _ = planner.get_path()
    viz = RRTPrinter3d(planner.env, planner, path_ids)
    viz.print_rrt(filename, print_tree=print_tree)


def save_quadcopter_path_3d(
    *,
    env,
    start,
    goal,
    goal_radius,
    highres_states: np.ndarray,
    filename: str,
    agent_radius: float = 0.25,
    path_color: str = "xkcd:bright blue",
    path_label: str = "Path",
):
    """
    Save a standalone 3D path rendering without requiring a tree structure.
    """
    highres_states = np.asarray(highres_states, dtype=np.float64)
    if highres_states.ndim != 2 or highres_states.shape[1] < 3:
        raise ValueError(f"Expected highres_states shape (N, >=3), got {highres_states.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_zlim(0, env.size[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    RRTPrinter3d.print_obs(ax, env.obstacles, env.obstacle_buffer)
    RRTPrinter3d.print_start(ax, start, type("Tmp", (), {"radius": agent_radius})(), pcol="black")
    RRTPrinter3d.print_goal(ax, goal, goal_radius)

    ax.plot(
        highres_states[:, 0],
        highres_states[:, 1],
        highres_states[:, 2],
        color=path_color,
        linewidth=2.0,
        label=path_label,
    )
    ax.scatter(
        [highres_states[-1, 0]],
        [highres_states[-1, 1]],
        [highres_states[-1, 2]],
        color=path_color,
        s=24,
    )
    ax.legend(loc="best")
    fig.savefig(filename)
    plt.close(fig)


def save_quadcopter_path_overlay_3d(planner, optimized_path_view, filename: str):
    """
    Save a 3D overlay comparing the raw planner warm start and the optimized path.
    """
    raw_states, _, _ = planner.get_high_resolution_path_and_actions()
    opt_states = optimized_path_view.get_high_resolution_path_numpy_array()

    raw_states = np.asarray(raw_states, dtype=np.float64)
    opt_states = np.asarray(opt_states, dtype=np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, planner.env.size[0])
    ax.set_ylim(0, planner.env.size[1])
    ax.set_zlim(0, planner.env.size[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    RRTPrinter3d.print_obs(ax, planner.env.obstacles, planner.env.obstacle_buffer)
    RRTPrinter3d.print_start(ax, planner.start, planner.agent, pcol="black")
    RRTPrinter3d.print_goal(ax, planner.goal, planner.goal_radius)

    ax.plot(
        raw_states[:, 0],
        raw_states[:, 1],
        raw_states[:, 2],
        color="xkcd:light grey",
        linewidth=1.8,
        label="Db-RRT warm start",
    )
    ax.plot(
        opt_states[:, 0],
        opt_states[:, 1],
        opt_states[:, 2],
        color="xkcd:bright blue",
        linewidth=2.0,
        label="Optimized trajectory",
    )
    ax.scatter([raw_states[0, 0]], [raw_states[0, 1]], [raw_states[0, 2]], color="black", s=24)
    ax.scatter([opt_states[-1, 0]], [opt_states[-1, 1]], [opt_states[-1, 2]], color="xkcd:bright blue", s=24)

    ax.legend(loc="best")
    fig.savefig(filename)
    plt.close(fig)
