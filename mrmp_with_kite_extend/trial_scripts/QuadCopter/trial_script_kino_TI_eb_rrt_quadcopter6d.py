import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt

from Environments import CuboidEnvironment, SphericalObstacle3D
from Agents import QuadCopter6D
from kinodynamic_TI_eb_rrt import * 
from edge_bundle import EdgeBundle
from printer import *
from kd_tree_quadcopter6d import *
from kd_tree_grid_quadcopter6d import *
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent


obstacles = [
            SphericalObstacle3D(x=2.0, y=2.0, z=0.8, r=0.4),
            SphericalObstacle3D(x=4.0, y=3.0, z=1.4, r=0.5),
            ]
# obstacles = []
env = CuboidEnvironment(length=14.0, breadth=14.0, height=6.0, obs=obstacles)
start = np.array((13.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64)
goal = np.array((1.0, 13.0, 4.9), dtype=np.float64)
goal_radius = 0.3


agent = get_quadcopter_agent(1)

edge_bundle_file_location = 'edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_quadcopter6d = EdgeBundle(data, fix_num_edges=100000, use_all_edges=False)
edge_ids = np.arange(kino_TI_eb_quadcopter6d.num_edges, dtype=np.int64)
vx = kino_TI_eb_quadcopter6d.start_states[:, 3]  # vx
vy = kino_TI_eb_quadcopter6d.start_states[:, 4]   # vy
vz = kino_TI_eb_quadcopter6d.start_states[:, 5]   # vz
v_scale = 1.0
delta_radius = 0.1
kd_tree_TI_eb_quadcopter6d = VxyzTree(vx, vy, vz, ids=edge_ids, 
                            scales=(v_scale,v_scale,v_scale))
kd_tree_TI_eb_quadcopter6d = VxyzGridTree(vx, vy, vz, ids=edge_ids, 
                            scales=(v_scale,v_scale,v_scale),
                            vmin=-agent.max_speed,
                            vmax= agent.max_speed,
                            cell_size=delta_radius/2,
                            initial_out_capacity=2048,
                            return_ids=False
                            )

s = np.random.randint(0, 1000)
print("Seed: ", s)
#s = 755
kino_eb_rrt  = KinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_quadcopter6d,
            use_fixed_sampling_time=False,
            sampling_time_step=1.0,
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
            num_random_edges= 10,
            eb_kd_tree=kd_tree_TI_eb_quadcopter6d,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
            )

kino_eb_rrt.plan_path()
node_ids, states, actions, timesteps = kino_eb_rrt.get_path()

v = RRTPrinter(env,kino_eb_rrt,node_ids)
v.print_rrt('media/kino_TI_eb_rrt_quadcopter6d.png')

i = 138
kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_indices
kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_mask

"""
#Check if the returned states and actions to find the path are correct

node_ids, states, actions, timesteps = kino_eb_rrt.get_path_to_node_id(kino_eb_rrt.goal_node_id)
for i in range(len(node_ids)-1):
    parent_state = states[i]
    action = actions[i]
    duration = timesteps[i]
    num_steps = round(duration/kino_eb_rrt.minimum_time_step)
    next_state, _ = agent.get_next_state(parent_state, action, duration, num_steps=num_steps)
    print("Next State from propagation: ", next_state)
    print("Stored State in the Tree: ", states[i+1])

"""


from pyinstrument import Profiler

s = np.random.randint(0, 1000)
# s = 868
print("Seed: ", s)
kino_eb_rrt  = KinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_quadcopter6d,
            use_fixed_sampling_time=False,
            sampling_time_step=1.0,
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
            num_random_edges= 10,
            eb_kd_tree=kd_tree_TI_eb_quadcopter6d,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
            )

p = Profiler()
p.start()
kino_eb_rrt.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  
