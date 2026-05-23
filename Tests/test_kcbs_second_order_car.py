import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from printer import *
from mapf_env_square_agent_second_order_car import get_second_order_car_agent, \
        get_rrt_planner, get_kino_TI_eb_rrt_planner_SOC


# num_agents = 6
# obstacles = []                      
# env = SquareEnvironment(10, 10, obstacles)

# pos1 = (2.0, 2.0, 0, 0.0, 0.0)
# pos2 = (2.0, 5.0, 0, 0.0, 0.0)
# pos3 = (2.0, 8.0, 0, 0.0, 0.0)
# pos4 = (8.0, 2.0, 0, 0.0, 0.0)
# pos5 = (8.0, 5.0, 0, 0.0, 0.0)
# pos6 = (8.0, 8.0, 0, 0.0, 0.0)

# starts = [pos1, pos2, pos3, pos4, pos5, pos6]
# goals = [pos6[0:2], pos4[0:2], pos5[0:2], pos3[0:2], pos2[0:2], pos1[0:2]]
# goal_radii = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# agent_ids = List()
# agents = []
# for agent_id in range(num_agents):
#     agent_ids.append(agent_id)
#     agents.append(get_second_order_car_agent(agent_id))

# planners = []
# planner_function = get_rrt_planner
# for i in range(num_agents):
#     planners.append(planner_function(starts[i],goals[i],goal_radii[i],agents[i],env))


# # s = np.random.randint(0, 1000)
# # print("RNG Seed: ", s)
# s = 610  
# kcbs_planner = KCBS(
#                     env = env,
#                     agents = agents,
#                     low_level_planners = planners,
#                     max_trials = 10000,
#                     planning_time = 600.0,
#                     rng_seed = s,
#                     print_logs=True
#                     )

# path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
# c, rrt = kcbs_planner.plan_multi_agent_paths()

# rrt.debug_flag = True
# rrt.print_logs = True
# rrt.reset_tree()
# rrt.max_iter = 17
# rrt.plan_path()


obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 2),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            ]
env = SquareEnvironment(40, 40, obstacles)

start1 = np.array((7.0, 5.0, 0, 0.0, 0.0))
goal1 = np.array((24.0, 37.0))

start2 = np.array((2.0, 26.0, 0, 0.0, 0.0))
goal2 = np.array((37.0, 30.0))

start3 = np.array((28.0, 5.0, 0, 0.0, 0.0))
goal3 = np.array((5.0, 29.0))

start4 = np.array((32.0, 18.0, 0, 0.0, 0.0))
goal4 = np.array((2.0, 10.0))

start5 = np.array((16.0, 37.0, 0, 0.0, 0.0))
goal5 = np.array((36.0, 10.0))

starts = [start1, start2, start3, start4, start5]
goals = [goal1, goal2, goal3, goal4, goal5]

num_agents= 5
agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_second_order_car_agent(agent_id))


seeds = [913, 2328, 8328, 7660, 4772, 6701, 3881, 2140, 5365, 1160]
# seeds = np.random.randint(0, 10000, size=100)
costs = []
time_array = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    planner_function = get_rrt_planner
    planner_function = get_kino_TI_eb_rrt_planner_SOC
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


"""

t1 = copy.deepcopy(time_array)
c1 = copy.deepcopy(costs)

"""
