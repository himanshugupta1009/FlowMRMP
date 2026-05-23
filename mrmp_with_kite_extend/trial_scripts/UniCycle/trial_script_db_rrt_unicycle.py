"""
Single-agent Db-RRT smoke test using the Dynoplan unicycle .msgpack primitives.

"""

import os
import sys
import time
sys.path.append('./src')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from Environments import RectangleObstacle2D, SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import (
    get_unicycle_agent,
    get_rrt_planner,
    get_kino_TI_eb_rrt_planner_unicycle,
    get_constrained_db_rrt_planner_unicycle)
from db.db_rrt import DbRRTPlanner
from db.db_optimize_unicycle import optimize_dbrrt_unicycle_path, UnicycleTrajOptOptions
from db.idb_rrt import IdbRRTPlanner
from motion_primitives import (
    load_unicycle_motion_primitives,
    transform_unicycle_trajectory_numba,
    )
from printer import RRTPrinter, print_optimized_overlay
from utils import verify_rollout_consistency


#Option1: Smaller env
obstacles = [CircularObstacle2D(3.0, 5.0, 1.5), CircularObstacle2D(8.0, 5.0, 1.5)]
env = SquareEnvironment(10, 10, obstacles, obs_buffers=True)
start = np.array([1.0, 1.0, 0.0], dtype=np.float64)
goal = np.array([8.0, 8.0], dtype=np.float64)
goal_radius = 0.25


#Option2: Bigger env
obstacles = [CircularObstacle2D(9.0, 9.0, 1.5)]
env = SquareEnvironment(20, 20, obstacles, obs_buffers=True)
start = np.array([1.0, 1.0, 0.0], dtype=np.float64)
goal = np.array([18.0, 18.0], dtype=np.float64)
goal_radius = 0.25

#Option3: Even bigger env with more obstacles
obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 4),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 5),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]   
# obstacles = []
env = SquareEnvironment(40, 40, obstacles)
start = np.array([5.0, 5.0, 0.0], dtype=np.float64)
goal = np.array([25.0, 25.0], dtype=np.float64)
goal_radius = 0.5

#Option4: BugTrap environment
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


agent = get_unicycle_agent(1)

s = np.random.randint(0, 1000)
print("Seed: ", s)
db_rrt_planner = get_constrained_db_rrt_planner_unicycle(start, goal, 
                                goal_radius, agent, env, use_optimizer=True)
db_rrt_planner.rng = np.random.default_rng(s)
db_rrt_planner.print_logs = True
db_rrt_planner.plan_path()

node_ids, states, actions, timesteps = db_rrt_planner.get_path()

v = RRTPrinter(env, db_rrt_planner, node_ids)
v.print_rrt("media/db_rrt_unicycle.png", print_tree=True)

verify_rollout_consistency(db_rrt_planner)


#Make the path feasible using Crocoddyl-based trajectory optimization, if possible.
#This is not strictly necessary for the Db-RRT smoke test, 

# print("\nAttempting to optimize Db-RRT warm start with Crocoddyl...")
# try:
#     t0 = time.time()
#     opt_result = optimize_dbrrt_unicycle_path(
#         db_rrt_planner,
#         options=UnicycleTrajOptOptions(),
#     )
#     elapsed = time.time() - t0
# except ImportError as exc:
#     print(f"\nSkipping Crocoddyl repair stage: {exc}")
#     print("If Crocoddyl is installed, this test can also optimize the Db-RRT warm start.")

# print("\n===== Crocoddyl Repair Result =====")
# print("Success:", opt_result.success)
# print("Feasible:", opt_result.feasible)
# print("Time Taken:", elapsed)

# verify_rollout_consistency(opt_result.path_view)

# print_optimized_overlay(db_rrt_planner, opt_result, 
#                         "media/db_rrt_unicycle_optimized_overlay.png")
