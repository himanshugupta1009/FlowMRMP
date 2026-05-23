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

obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 2),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 4),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]                      
env = SquareEnvironment(40, 40, obstacles)
goal_radius = 0.5

num_agents = 5

start1 = np.array((7.0, 5.0, 0))
goal1 = np.array((24.0, 37.0))

start2 = np.array((2.0, 26.0, 0))
goal2 = np.array((37.0, 30.0))

start3 = np.array((28.0, 5.0, 0))
goal3 = np.array((5.0, 29.0))

start4 = np.array((32.0, 18.0, 0))
goal4 = np.array((2.0, 10.0))

start5 = np.array((16.0, 37.0, 0))
goal5 = np.array((36.0, 10.0))

starts = [start1, start2, start3, start4, start5]
goals = [goal1, goal2, goal3, goal4, goal5]

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

planners = []
planner_function = get_rrt_planner
planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

# s = np.random.randint(0, 1000)
# print("RNG Seed: ", s)
s = 42  
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 50,
                    planning_time = 600.0,
                    rng_seed = s,
                    print_logs=True,
                    debug_flag=True
                    )
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
# c, rrt = kcbs_planner.plan_multi_agent_paths()

# rrt.debug_flag = True
# rrt.print_logs = True
# rrt.reset_tree()
# rrt.max_iter = 17
# rrt.plan_path()


seeds = [42, 595, 272, 275, 613, 350, 65, 662, 940, 815]
costs = []
time_array = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    # planner_function = get_rrt_planner
    # planner_function = get_eb_rrt_planner
    planner_function = get_kino_TI_eb_rrt_planner_unicycle
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],0.5,agents[i],env))

    kcbs_planner = KCBS(
                env = env,
                agents = agents,
                low_level_planners = planners,
                max_trials = 1000,
                planning_time = 600.0,
                rng_seed = s,
                print_logs=False
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


