import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt

from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from utils import euclidean_distance, euclidean_distance_satisfaction_numba
from kinodynamic_TI_eb_sst import * 
from edge_bundle import EdgeBundle, GenerateEdgeBundle
from sst_printer import *
from kd_tree_unicycle import CircularAngleIndexNumba


start = (5.0, 5.0, 0.0)
goal = (25.0, 25.0)
goal_radius = 1.0
obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            # CircularObstacle2D(35, 15, 4),
            # CircularObstacle2D(30, 34, 4),
            # CircularObstacle2D(25, 15, 4),
            # CircularObstacle2D(7, 19, 5),
            # CircularObstacle2D(16, 16, 2),
            # CircularObstacle2D(33, 4, 2),
            # CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]   
# obstacles = []
env = SquareEnvironment(40, 40, obstacles)
# agent = UniCycle(agent_id = 1, 
#                  max_speed = 2.0,
#                  max_omega= np.pi/2,
#                  radius = 1.0,
#                  rng_seed= 77)
agent = get_unicycle_agent(1)

edge_bundle_file_location = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)
edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
kd_tree_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)


s = np.random.randint(0, 1000)
print("Seed: ", s)
kino_eb_sst  = EB_SST( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_unicycle,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            num_random_edges= 10,
            num_skip_edges= 100,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            eb_kd_tree=kd_tree_TI_eb_unicycle,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            max_num_edges_per_node=1000,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
            best_near_radius=5.0,
            prune_radius=0.1
            )

kino_eb_sst.plan_path()
node_ids, states, actions, timesteps = kino_eb_sst.get_path()

printer = SSTPrinter(env, kino_eb_sst)
printer.print_sst("media/kino_TI_eb_sst_unicycle.png",
                  show_tree=True, show_path=True,
                  show_active=True, show_inactive=False,
                  show_witness=True, show_prune_circles=False)



"""
#Check if the returned states and actions to find the path are correct

node_ids, states, actions, timesteps = kino_eb_sst.get_path_to_node_id(kino_eb_sst.goal_node_id)
for i in range(len(node_ids)-1):
    parent_state = states[i]
    action = actions[i]
    duration = timesteps[i]
    num_steps = round(duration/sst.minimum_time_step)
    next_state, _ = agent.get_next_state(parent_state, action, duration, num_steps=num_steps)
    print("Next State from propagation: ", next_state)
    print("Stored State in SST: ", states[i+1])

"""


from pyinstrument import Profiler

s = np.random.randint(0, 1000)
print("Seed: ", s)
kino_eb_sst  = EB_SST( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_unicycle,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            num_random_edges= 10,
            num_skip_edges= 100,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            eb_kd_tree=kd_tree_TI_eb_unicycle,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            max_num_edges_per_node=1000,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
            best_near_radius=5.0,
            prune_radius=0.1
            )

p = Profiler()
p.start()
kino_eb_sst.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  