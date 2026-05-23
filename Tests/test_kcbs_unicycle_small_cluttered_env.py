import sys
sys.path.append('./src')
from Environments import *
from Agents import UniCycle
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from mapf_env_square_agent_unicycle import (
        get_unicycle_agent,
        get_rrt_planner, get_eb_rrt_planner, get_kino_TI_eb_rrt_planner_unicycle,
        get_constrained_db_rrt_planner_unicycle)
from printer import *

starts = [
            (1.0, 1.0, 0.0),    # red
            (6.0, 5.0, 0.0),    # teal
            (12.0, 5.0, 0.0),   # blue
            (14.0, 4.0, 0.0),   # green
            (9.0, 10.0, 0.0),   # magenta
            (3.0, 11.0, 0.0),   # red/orange
            (6.0, 14.0, 0.0),   # green (top)
            (8.0, 5.0, 0.0),    # purple
            (10.0, 1.0, 0.0),   # yellow
            (11.0, 14.0, 0.0),  # blue (top)
        ]

goals = [
            (6.0, 3.0),     # red
            (2.0, 11.0),    # teal
            (5.0, 12.0),    # blue
            (9.0, 1.0),     # green
            (14.0, 3.0),    # magenta
            (12.0, 12.0),   # red/orange
            (3.0, 7.0),     # green (top)
            (13.0, 6.0),    # purple
            (11.0, 8.0),    # yellow
            (3.0, 14.0),    # blue (top)
        ]


obstacles = [
            # bottom row
            RectangleObstacle2D(4.5, 1.5, 3, 1),
            RectangleObstacle2D(12.5, 1.5, 1, 1),

            # lower-middle
            RectangleObstacle2D(2.0, 4.5, 2, 1),
            RectangleObstacle2D(10.0, 3.5, 2, 1),

            # middle
            RectangleObstacle2D(.5, 7.5, 1, 1),
            RectangleObstacle2D(1.5, 9.5, 1, 1),
            RectangleObstacle2D(13.5, 8.0, 1, 2),

            # vertical center column
            RectangleObstacle2D(7.5, 8.0, 1, 4),

            # upper-middle
            RectangleObstacle2D(2.5, 12.5, 3, 1),
            RectangleObstacle2D(10.5, 10.5, 1, 1),

            # top
            RectangleObstacle2D(7.5, 14.5, 1, 1),
            RectangleObstacle2D(13.0, 14.0, 2, 2),
        ]

env = SquareEnvironment(15.0, 15.0, obstacles)

num_agents = 5
goal_radius = 1.0


agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

planners = []
planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 10000,
                    planning_time = 600.0,
                    rng_seed = s,
                    print_logs=False,
                    debug_flag=False
                    )
t = time.time()
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Planning Time: ", t)


"""

seeds = np.random.randint(0, 10000, size=10)
seeds = [5431, 5930, 5126,  983, 2899, 6983, 6879, 1363, 9582, 7075]
for s in seeds:
    print("\nRNG Seed: ", s)

    planners = []
    planner_function = get_rrt_planner
    # planner_function = get_eb_rrt_planner
    # planner_function = get_kino_TI_eb_rrt_planner_unicycle
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))
    
    kcbs_planner = KCBS(
                        env = env,
                        agents = agents,
                        low_level_planners = planners,
                        max_trials = 10000,
                        planning_time = 600.0,
                        rng_seed = s,
                        print_logs=False,
                        debug_flag=False
                        )
    t = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print("Planning Time: ", t)
    print("Paths Cost: ", cost)

"""


"""
import gc
seeds = np.random.randint(0, 10000, size=25)
costs = []
time_array = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

# seeds = np.random.randint(0, 10000, size=1)
for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    planner_function = get_kino_TI_eb_rrt_planner_unicycle
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

    kcbs_planner = KCBS(
                env = env,
                agents = agents,
                low_level_planners = planners,
                max_trials = 1000,
                planning_time = 300.0,
                rng_seed = s,
                print_logs=False,
                debug_flag=False,
                )
    t = time.time()  
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print("Iteration: ", seed_index)
    print("RNG Seed: ", s)
    print("Time taken for planning: ", t)
    time_array.append(t)
    print("Cost: ", cost)
    costs.append(cost)
    print("#######################################################")
    gc.collect()


# seeds = [93, 228, 828, 760, 472, 701, 881, 140, 365, 160]
costs2 = []
time_array2 = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

# seeds = np.random.randint(0, 10000, size=1)
for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    planner_function = get_rrt_planner
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

    kcbs_planner = KCBS(
                env = env,
                agents = agents,
                low_level_planners = planners,
                max_trials = 1000,
                planning_time = 300.0,
                rng_seed = s,
                print_logs=False,
                debug_flag=False,
                )
    t = time.time()  
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print("Iteration: ", seed_index)
    print("RNG Seed: ", s)
    print("Time taken for planning: ", t)
    time_array2.append(t)
    print("Cost: ", cost)
    costs2.append(cost)
    print("#######################################################")
    gc.collect()

print("Kino-TI EB RRT KCBS Results:")
print("Mean Cost: ", np.mean(costs))
print("Mean Time: ", np.mean(time_array))

print("RRT KCBS Results:")
print("Mean Cost: ", np.mean(costs2))
print("Mean Time: ", np.mean(time_array2))
"""