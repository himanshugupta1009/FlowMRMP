import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt

from Environments import SquareEnvironment, CircularObstacle2D
from kinodynamic_TI_eb_rrt import * 
from edge_bundle import EdgeBundle
from printer import *
from kd_tree_unicycle import *
from mapf_env_square_agent_unicycle import get_unicycle_agent


start = np.array([5.0, 5.0, 0.0])
goal = (25.0, 25.0)
goal_radius = 0.5
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
agent = get_unicycle_agent(1)


edge_bundle_file_location = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)
edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
kd_tree_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)


seeds = np.random.randint(0, 1000, size=25)
planning_time_kino_TI_eb_rrt = []
time_kino_TI_eb_rrt = []
cost_kino_TI_eb_rrt = []
planning_time_rrt = []
time_rrt = []
cost_rrt = []

T_max = 1.0

for i in range(len(seeds)):
    print("\n------------------ Trial ", i+1, " ------------------")
    s = seeds[i]
    print("Seed: ", seeds[i])
    kino_eb_rrt  = KinoTIEBRRT( 
                start=start, goal=goal,
                goal_radius=goal_radius,
                env = env, agent=agent, 
                edge_bundle = kino_TI_eb_unicycle,
                use_fixed_sampling_time=False,
                sampling_time_step=T_max,
                minimum_time_step=0.1,
                max_iter = 10000,
                planning_time=600.0,
                isvalid_function=agent.is_new_node_valid,
                cost_function=agent.get_cost,
                random_point_function=agent.get_random_point,
                reached_goal_function = agent.agent_reached_goal,
                translate_function = agent.kd_tree_point_translate_function,
                sort_edges_function=agent.sort_kd_tree_edges,
                max_num_edges_per_node=1000,
                num_skip_edges= 10,
                num_random_edges= 0,
                eb_kd_tree=kd_tree_TI_eb_unicycle,
                get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                kd_tree_delta_radius=0.1,
                udf_seed = s,
                debug_flag=False,
                print_logs=True,
                )
    
    t = time.time()
    kino_eb_rrt.plan_path()
    t = time.time() - t
    planning_time_kino_TI_eb_rrt.append(t)
    time_kino_TI_eb_rrt.append(kino_eb_rrt.path_time)
    cost_kino_TI_eb_rrt.append(kino_eb_rrt.path_cost)

    rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=T_max,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,
            num_extension_trials=10,      
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True
           )
    
    t = time.time()
    rrt.plan_path()
    t = time.time() - t
    planning_time_rrt.append(t)
    time_rrt.append(rrt.path_time)
    cost_rrt.append(rrt.path_cost)

print("\nKino TI EB RRT Results:")
print("Average Planning Time: {:.2f} s".format(np.mean(planning_time_kino_TI_eb_rrt)))
print("Average Time: {:.2f} s".format(np.mean(time_kino_TI_eb_rrt)))
print("Average Cost: {:.2f}".format(np.mean(cost_kino_TI_eb_rrt)))
print("\nRRT Results:")
print("Average Planning Time: {:.2f} s".format(np.mean(planning_time_rrt)))
print("Average Time: {:.2f} s".format(np.mean(time_rrt)))
print("Average Cost: {:.2f}".format(np.mean(cost_rrt)))