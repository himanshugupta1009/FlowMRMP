import sys
sys.path.append('./src')

import numpy as np
from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from Agents import FlowCar
from rrt import RRT
from printer import RRTPrinter


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

obstacles = [
    CircularObstacle2D(10, 10, 2),
    CircularObstacle2D(16, 25, 3),
    CircularObstacle2D(20, 5,  2),
    CircularObstacle2D(25, 15, 4),
    RectangleObstacle2D(30, 20, 6, 4),
]

env = SquareEnvironment(40, 40, obstacles)

# ---------------------------------------------------------------------------
# Agent
# State : [x, y, psi, v, D, delta]
# Action: [dD, dDelta]
# All defaults match car_env.py exactly.
# ---------------------------------------------------------------------------

agent = FlowCar(
    agent_id=1,
    max_speed=10.0,
    max_D=1.0,
    max_delta=0.40,
    max_dD=10.0,
    max_dDelta=2.0,
    radius=0.3,
    rng_seed=42,
)

# ---------------------------------------------------------------------------
# Start / goal
# Start at rest (v=0, D=0, delta=0), pointing in the +x direction (psi=0).
# Goal is an (x, y) point; agent_reached_goal checks Euclidean distance to it.
# ---------------------------------------------------------------------------

start       = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
goal        = np.array([35.0, 35.0], dtype=np.float64)
goal_radius = 0.5

# ---------------------------------------------------------------------------
# RRT
# sampling_time_step: maximum edge duration [s].
# minimum_time_step : record interval [s] — controls path resolution and sets
#                     the RK4 step size via get_next_state(num_steps=round(T/dt)).
# The car's native dt is 0.02 s; 0.1 s per record step is 5× coarser but
# sufficient for verifying correctness.
# ---------------------------------------------------------------------------

s = np.random.randint(0, 1000)
print(f"RNG seed: {s}")

rrt = RRT(
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    env=env,
    agent=agent,
    sampling_time_step=1.0,
    use_fixed_sampling_time=False,
    minimum_time_step=0.1,
    max_iter=10000,
    num_extension_trials=10,
    planning_time=300.0,
    isvalid_function=agent.is_new_node_valid,
    cost_function=agent.get_cost,
    random_point_function=agent.get_random_point,
    reached_goal_function=agent.agent_reached_goal,
    udf_seed=s,
    print_logs=True,
)

rrt.plan_path()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

rrt_node_ids, states, actions, timesteps = rrt.get_path()

if rrt.path_found:
    print(f"\nPath found!")
    print(f"  Nodes in tree : {rrt._node_matrix.count}")
    print(f"  Path length   : {len(states)} states")
    print(f"  Path cost     : {rrt.path_cost:.3f} s")
    print(f"  Path time     : {rrt.path_time:.3f} s")
else:
    print("\nNo path found within the time/iteration budget.")
    print(f"  Nodes in tree : {rrt._node_matrix.count}")

v = RRTPrinter(env, rrt, rrt_node_ids)
v.print_rrt('media/rrt_graph_flowcar.png')
print("Tree plot saved to media/rrt_graph_flowcar.png")
