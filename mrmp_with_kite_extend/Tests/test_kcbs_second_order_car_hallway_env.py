import sys
sys.path.append('./src')
from Environments import SquareEnvironment, RectangleObstacle2D
from Agents import SecondOrderCar
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from printer import *
from mapf_env_square_agent_second_order_car import get_second_order_car_agent, \
        get_rrt_planner, get_kino_TI_eb_rrt_planner_SOC


goal_radius = 0.25
starts = [(1.5,5.0,0.0,0.0,0.0),
        #   (8.5,5.0,0.0,0.0,0.0),
          ]
goals = [(8.5, 2.5), 
        #  (1.5, 2.5)
         ]

num_agents= len(starts)

obstacles = [
            RectangleObstacle2D(x = 5.0, y=1.75, w=4, h=3.5),
            RectangleObstacle2D(x = 5.0, y=7.5, w=4, h=5),        
            ]

env = SquareEnvironment(10, 10, obstacles)

goal_radii = [goal_radius for _ in range(num_agents)]

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_second_order_car_agent(agent_id))

pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
MultiRRTPrinter.print_rrt_env('./media/env_hallway_' + str(num_agents) + 'agents' + '.png',
                            env, agents, starts, goals, goal_radii, pcol)



seeds = [93, 228, 828, 760, 472, 701, 881, 140, 365, 160]
import random
random.seed(4)
seeds = [random.randint(0, 10000) for _ in range(7)]
seeds = [2539]
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


planner_list = [planners[i] for i in range(num_agents)]

path_node = kcbs_planner.path_cbs_node
dict_paths = {}
ids_list = []
for i in range(num_agents):
    t = path_node.agent_trees[i]
    planners[i].tree = t[0]
    planners[i]._node_matrix = t[1]
    goal_id = next(reversed(t[0]._node))
    planners[i].goal_node_id = goal_id
    planners[i].path_found = True
    dict_paths[i] = planners[i].get_high_resolution_path()
    ids, states, controls, timesteps = planners[i].get_path()
    ids_list.append(ids)


# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
mprint = MultiRRTPrinter(env, planner_list, ids_list, tcol, pcol)
mprint.print_rrt('./media/hallway_env_kcbs_paths.png', print_tree=False)
# mprint.print_highres_simulation(dict_paths, "./media/kcbs_hallway_env_SOC.gif", 
                            # animation_speed=50)


"""

In [16]: costs
Out[16]: 
[inf,
 25.618095370262516,
 31.81021381473338,
 41.93135005313057,
 31.736541600164315,
 41.14799737084203,
 39.67608584698317]

In [17]: time_array
Out[17]: 
[602.6972618103027,
 49.61966252326965,
 71.51467871665955,
 190.35194659233093,
 455.2608675956726,
 48.897682428359985,
 265.7974302768707]

In [18]: seeds
Out[18]: [3867, 4969, 1690, 6489, 7845, 2539, 1476]

"""