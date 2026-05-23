import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt

from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_second_order_car import get_second_order_car_agent
from utils import euclidean_distance, euclidean_distance_satisfaction_numba
from kinodynamic_TI_eb_rrt import * 
from edge_bundle import EdgeBundle
from printer import *
from kd_tree_second_order_car import *


start = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
goal = np.array([24.0, 37.0])
goal_radius = 1.0
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
obstacles = []
env = SquareEnvironment(40, 40, obstacles)
# agent = SecondOrderCar(agent_id = 1, 
#                        max_speed = 1.0,
#                        max_acceleration = 2.0,
#                        max_phi = np.pi/3,
#                        max_steering_rate = 0.5,
#                        radius = 0.3,
#                        wheelbase = 0.7,
#                        rng_seed=42
#                        )

agent = get_second_order_car_agent(1)
edge_bundle_file_location = 'edge_bundles_unclamped/eb_second_order_car_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_SOC = EdgeBundle(data, fix_num_edges=50000, use_all_edges=False)
edge_ids = np.arange(kino_TI_eb_SOC.num_edges, dtype=np.int64)
speeds = kino_TI_eb_SOC.start_states[:, 3]  # v
phis = kino_TI_eb_SOC.start_states[:, 4]   # phi
v_scale = agent.max_speed
phi_scale = agent.max_phi
kd_tree_TI_eb_SOC = VPhiTree(speeds, phis, ids=edge_ids, 
                    v_scale=v_scale, phi_scale=phi_scale)


s = np.random.randint(0, 1000)
print("Seed: ", s)
#s = 755
kino_eb_rrt  = KinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_SOC,
            use_fixed_sampling_time=True,
            sampling_time_step=2.0,
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
            num_skip_edges= 100,
            num_random_edges= 10,
            eb_kd_tree=kd_tree_TI_eb_SOC,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
            )

kino_eb_rrt.plan_path()
node_ids, states, actions, timesteps = kino_eb_rrt.get_path()

v = RRTPrinter(env,kino_eb_rrt,node_ids)
v.print_rrt('media/kino_TI_eb_rrt_second_order_car.png')

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
    print("Stored State in SST: ", states[i+1])

"""


from pyinstrument import Profiler

s = np.random.randint(0, 1000)
# s = 868
print("Seed: ", s)
kino_eb_rrt  = KinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_SOC,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 50000,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            max_num_edges_per_node=1000,
            num_skip_edges= 100,
            num_random_edges= 10,
            eb_kd_tree=kd_tree_TI_eb_SOC,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.5,
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




#Plotting Edge Bundles
plot_edge_ids = kd_tree_TI_eb_SOC.radius_query((-0.3, 0.3), 0.1)
kino_TI_eb_SOC.plot_edges_by_ids(plot_edge_ids)

for i in range(len(plot_edge_ids)):
    idx = plot_edge_ids[i]
    print(f"ID: {idx}, phi: {kd_tree_TI_eb_SOC.phi[idx]:.4f}, \
                        v: {kd_tree_TI_eb_SOC.v[idx]:.4f}")
    print("Start State: ", kino_TI_eb_SOC.start_states[idx])
    print("Final State: ", kino_TI_eb_SOC.final_states[idx])
    print("Action: ", kino_TI_eb_SOC.actions[idx])
    print("Timestep: ", kino_TI_eb_SOC.timesteps[idx])
    print("######################################")



plot_edge_ids = kino_eb_rrt.tree.nodes[2]['value'].edge_bundle_indices
kino_TI_eb_SOC.plot_edges_by_ids(plot_edge_ids)

#Plotting Edge Bundles
test_state = kino_eb_rrt.tree.nodes[2]['value'].state
print("Test State: ", test_state)
plot_edge_ids = kd_tree_TI_eb_SOC.radius_query(
                (test_state[3], test_state[4]), 0.1)
kino_TI_eb_SOC.plot_edges_by_ids(plot_edge_ids)


for i in range(len(kino_eb_rrt.tree.nodes)):
    print( "i : ", i, " State : ", kino_eb_rrt.tree.nodes[i]['value'].state )
    print("######################################")




for i in range(len(kino_eb_rrt.tree.nodes)):
    if kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_indices is not None:
        if sum( kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_mask ) > 0:
            print( len( kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_indices ) )
            print( sum( kino_eb_rrt.tree.nodes[i]['value'].edge_bundle_mask ) )
            print("######################################")


c = kino_eb_rrt.compute_children_counts()
for k in c.keys():
    if c[k] > 1:
        print(f"Node id: {k}, Children count: {c[k]}")

sum(kino_eb_rrt.tree.nodes[4]['value'].edge_bundle_mask)


c = kino_eb_rrt.compute_children_counts()
for k in c.keys():
    if c[k] > 1:
        print(f"Node id: {k}, Children count: {c[k]}")
        print(sum(kino_eb_rrt.tree.nodes[k]['value'].edge_bundle_mask))
        if sum(kino_eb_rrt.tree.nodes[k]['value'].edge_bundle_mask) != c[k]:
            print("Mismatch found!")



c = kino_eb_rrt.compute_children_counts()
for k in c.keys():
    if c[k] > 1:
        print(f"Node id: {k}, Children count: {c[k]}")
        print(sum(kino_eb_rrt.tree.nodes[k]['value'].edge_bundle_mask))
        if sum(kino_eb_rrt.tree.nodes[k]['value'].edge_bundle_mask) < c[k]:
            print("Mismatch found!")



i = 13
st = kino_eb_rrt.tree.nodes[i]['value'].state
nearby_ed_ids = kd_tree_TI_eb_unicycle.radius_query( kino_eb_rrt.get_eb_kd_tree_query(st), 0.1)
x = nearby_ed_ids[273]
eb_st = kino_TI_eb_unicycle.start_states[x]
a = kino_TI_eb_unicycle.actions[x]
t = kino_TI_eb_unicycle.timesteps[x]
traj = kino_TI_eb_unicycle.trajectories[x]
f = kino_TI_eb_unicycle.final_states[x]

new_state, path = agent.get_next_state(st, a, t, num_steps=round(t/0.1))

new_eb_st = np.array([st[0], st[1], eb_st[2]])
new_state_eb, path_eb = agent.get_next_state(new_eb_st, a, t, num_steps=round(t/0.1))



samples = np.random.choice(nearby_ed_ids, 10, replace=False)
kino_TI_eb_unicycle.plot_edges_by_ids(samples)

dist_arr = np.full(10000,-1, dtype=np.float64)
unicycle_sort_kd_tree_edges_numba(st, goal, 
    kino_TI_eb_unicycle.start_states, 
    kino_TI_eb_unicycle.final_states, 
    samples, 
    np.zeros(len(samples), dtype=np.bool_),
    dist_arr)

sample_idx = 9
end_point = unicycle_point_translate_function_kd_tree_numba(st,
        kino_TI_eb_unicycle.start_states[samples[sample_idx]],
        kino_TI_eb_unicycle.final_states[samples[sample_idx]])
print("End Point from KD-Tree translation: ", end_point)



"""
#Verify if the edge bundle is actually correct.

for i in range(kino_TI_eb_unicycle.num_edges):
    start_state = kino_TI_eb_unicycle.start_states[i]
    action = kino_TI_eb_unicycle.actions[i]
    duration = kino_TI_eb_unicycle.timesteps[i]
    num_steps = round(duration/0.1)
    end_state, _ = agent.get_next_state(start_state, action, duration, num_steps=num_steps)
    if not np.allclose(end_state, kino_TI_eb_unicycle.final_states[i]):
        print("Mismatch found at edge id: ", i)
        print("Computed end state: ", end_state)
        print("Stored final state: ", kino_TI_eb_unicycle.final_states[i])

"""

seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
times = []
costs = []
for s in seeds:
    kino_eb_rrt  = KinoTIEBRRT( 
                            start=start, goal=goal,
                            goal_radius=goal_radius,
                            env = env, agent=agent, 
                            edge_bundle = kino_TI_eb_SOC,
                            use_fixed_sampling_time=False,
                            sampling_time_step=1.5,
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
                            num_skip_edges= 100,
                            num_random_edges= 10,
                            eb_kd_tree=kd_tree_TI_eb_SOC,
                            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                            kd_tree_delta_radius=0.5,
                            udf_seed = s,
                            debug_flag=False,
                            print_logs=True,
                            )

    kino_eb_rrt.plan_path()
    rrt_node_ids, states, actions, timesteps = kino_eb_rrt.get_path()
    times.append(kino_eb_rrt.path_time)
    costs.append(kino_eb_rrt.path_cost)




#For testing point translation stuff with Second Order Car model 
from src.Agents import SecondOrderCar

agent = SecondOrderCar(agent_id = 1, 
                       max_speed = 1.0,
                       max_acceleration = 1.0,
                       max_phi = np.pi/3,
                       max_steering_rate = 0.5,
                       radius = 0.3,
                       wheelbase = 0.7,
                       rng_seed=42
                       )

s0 = np.array([0,0,0,0.7,0.33])
a0 = np.array([0.5, 0.3])
t0 = 0.9
ns0, p0 = agent.get_next_state(s0, a0, t0, num_steps=round(t0/0.1))


s0 = np.array([20,10,0,0.7,0.33])
a0 = np.array([0.5, 0.3])
t0 = 0.9
ns0, p0 = agent.get_next_state(s0, a0, t0, num_steps=round(t0/0.1))


a0 = np.array([np.random.uniform(-1,1), np.random.uniform(-0.5, 0.5)])
t0 = 0.9

s0 = np.array([0,0,0,0.7,0.33])
ns0, p0 = agent.get_next_state(s0, a0, t0, num_steps=round(t0/0.1))

s1 = np.array([5,5,np.pi/12,0.7,0.33])
ns1, p1 = agent.get_next_state(s1, a0, t0, num_steps=round(t0/0.1))

base_point = np.array([5,5,np.pi/12,0.7,0.33])
edge_end_point = ns0
ns1_translated = agent.kd_tree_point_translate_function(base_point, s0, edge_end_point)

print("New state from propagation: ", ns1)
print("New state from translation: ", ns1_translated)