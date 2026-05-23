import sys

sys.path.append('./src')
from Environments import *
from Agents import SecondOrderCar
from utils import euclidean_distance
from constrainedX import *
from kcbs import *
from mapf_env_square_agent_unicycle import (
            get_unicycle_agent,
            get_rrt_planner, get_kino_TI_eb_rrt_planner_unicycle,
            get_constrained_db_rrt_planner_unicycle)
from printer import *

goal_radius = 0.25
starts = [(3.5,3.5,np.pi),
          (1.0,1.5,0.0),
          (4.0,1.5,np.pi)
          ]
goals = [(0.5, 2.5), (4.5, 1.5),(2.5, 1.5)]

obstacles = [
            RectangleObstacle2D(x = 1.5, y=0.5, w=3.0, h=0.7),
            RectangleObstacle2D(x = 3.4, y=2.5, w=3.0, h=0.7),        
            ]

env = SquareEnvironment(5, 4, obstacles)

num_agents = len(starts)
agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

"""
s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
i = 0
rrt  = RRT( 
            start=starts[i], goal=goals[i],
            goal_radius=goal_radius, 
            env = env, agent=agents[i],
            sampling_time_step=1.5,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            max_iter = 10_000,
            planning_time=300.0,         
            isvalid_function=agents[i].is_new_node_valid,
            cost_function=agents[i].get_cost,
            random_point_function=agents[i].get_random_point, 
            reached_goal_function = agents[i].agent_reached_goal,
            udf_seed = s,
            print_logs=True
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_second_order_car_corridor.png')
"""

seeds = [93, 228, 828, 760, 472, 701, 881, 140, 365, 160]
seeds = [93, 228, 828, 760, 472]
#Seeds that failed to give paths: 472
# seeds = [93]
costs = []
time_array = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

# seeds = np.random.randint(0, 10000, size=1)
for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    planner_function = get_rrt_planner
    planner_function = get_kino_TI_eb_rrt_planner_unicycle
    # planner_function = get_constrained_db_rrt_planner_unicycle
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

    kcbs_planner = KCBS(
                env = env,
                agents = agents,
                low_level_planners = planners,
                max_trials = 100,
                planning_time = 300.0,
                rng_seed = s,
                print_logs=True,
                debug_flag=False,
                prune_tree=True,
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
#Visualize the paths

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
mprint.print_rrt('./media/kcbs_corridor_env_second_order_car.png', print_tree=False)

mprint.print_highres_simulation(dict_paths, "./media/kcbs_corridor_env_second_order_car.gif", 
                            animation_speed=50)

"""