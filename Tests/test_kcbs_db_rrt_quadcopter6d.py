import sys
sys.path.append('./src')
import time
import numpy as np
from Environments import CuboidEnvironment, SphericalObstacle3D
from kcbs import KCBS
from mapf_env_cuboid_agent_quadcopter6d import (
    get_quadcopter_agent,
    get_constrained_db_rrt_planner_quadcopter6d,
)


obstacles = [
    SphericalObstacle3D(x=2.5, y=2.5, z=0.8, r=0.4),
    SphericalObstacle3D(x=5.5, y=3.5, z=1.6, r=0.6),
    SphericalObstacle3D(x=3.5, y=6.0, z=1.0, r=0.5),
]
env = CuboidEnvironment(length=8.0, breadth=8.0, height=3.0, obs=obstacles)
goal_radius = 0.35

starts = [
    np.array((0.8, 0.8, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((6.5, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((1.0, 6.0, 0.8, 0.0, 0.0, 0.0), dtype=np.float64),
]
goals = [
    np.array((7.0, 7.0, 1.8), dtype=np.float64),
    np.array((1.5, 6.5, 1.2), dtype=np.float64),
    np.array((6.2, 1.5, 1.5), dtype=np.float64),
]

num_agents = len(starts)
agents = [get_quadcopter_agent(agent_id) for agent_id in range(num_agents)]


def build_planners(*, use_optimizer=True):
    planners = []
    for i in range(num_agents):
        planners.append(
            get_constrained_db_rrt_planner_quadcopter6d(
                starts[i],
                goals[i],
                goal_radius,
                agents[i],
                env,
                use_optimizer=use_optimizer,
            )
        )
    return planners


planners = build_planners(use_optimizer=True)
s = 93
print("RNG Seed:", s)
kcbs_planner = KCBS(
    env=env,
    agents=agents,
    low_level_planners=planners,
    max_trials=50,
    planning_time=300.0,
    rng_seed=s,
    print_logs=True,
    debug_flag=True,
)
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()

print("Path found:", path_found)
print("Cost:", cost)
print("Delta t:", delta_t)


seeds = [93]
costs = []
time_array = []

print("Starting KCBS Db-RRT Quadcopter Experiments")
print("#######################################################")

for seed_index, s in enumerate(seeds):
    planners = build_planners(use_optimizer=True)

    kcbs_planner = KCBS(
        env=env,
        agents=agents,
        low_level_planners=planners,
        max_trials=200,
        planning_time=300.0,
        rng_seed=s,
        print_logs=False,
        debug_flag=False,
    )
    t = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print("Iteration:", seed_index)
    print("RNG Seed:", s)
    print("Time taken for planning:", t)
    time_array.append(t)
    print("Cost:", cost)
    costs.append(cost)
    print("#######################################################")
