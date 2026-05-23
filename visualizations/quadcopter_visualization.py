"""
Quadcopter path visualization helper.
Uses QuadVisualizer (PyBullet) to replay a high-resolution path dict
produced by RRT.get_high_resolution_path().
"""
import numpy as np

from Environments import QuadVisualizer


def visualize_quadcopter_path(path_dict, env, agent_radius=0.25, goal=None):
    """
    Visualize a quadcopter path in PyBullet.

    Args:
        path_dict (dict[float, sequence]): time->state mapping; state must have x,y,z in first 3 entries.
        env: Environment instance (CuboidEnvironment with obstacles).
        agent_radius (float): radius for drawing the quad body.
        goal (sequence, optional): explicit goal position (x, y, z). If None, uses final state in path.
    """
    viz = QuadVisualizer(env, agent_radius=agent_radius)
    final_goal = goal if goal is not None else list(path_dict.values())[-1][:3] if path_dict else None
    viz.visualize_path(path_dict, goal_position=final_goal)
