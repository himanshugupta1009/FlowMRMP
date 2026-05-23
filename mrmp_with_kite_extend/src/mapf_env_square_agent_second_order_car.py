import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar
from kd_tree_second_order_car import *
from edge_bundle_rrt import * 
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *


def get_second_order_car_agent(agent_id):
    agent = SecondOrderCar(agent_id = agent_id, 
                       max_speed = 1.0,
                       max_acceleration = 2.0,
                       max_phi = np.pi/3,
                       max_steering_rate = 0.5,
                       radius = 0.3,
                       wheelbase = 0.7,
                       rng_seed=42
                       )
    return agent


def get_rrt_planner(start,goal,gr,agent,env,filler_input=''):

    rrt_planner  = ConstrainedRRT( 
            start=start, goal=goal,
            goal_radius=gr, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=2.0,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=10.0, 
            num_extension_trials=1,        
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = np.random.randint(0, 1000),
            # udf_seed = 17
           )
    return rrt_planner


def get_kino_TI_eb_rrt_planner_SOC(start, goal, goal_radius, agent, env,
    edge_bundle_file_location = 'edge_bundles_unclamped/eb_second_order_car_kinodynamic_TI_edges_100000.npz'):

    data = np.load(edge_bundle_file_location)
    kino_TI_eb = EdgeBundle(data, fix_num_edges=50000, 
                use_all_edges=False,rng_seed=42 + agent.id)
    edge_ids = np.arange(kino_TI_eb.num_edges, dtype=np.int64)
    speeds = kino_TI_eb.start_states[:, 3]  # v
    phis = kino_TI_eb.start_states[:, 4]   # phi
    v_scale = agent.max_speed
    phi_scale = agent.max_phi
    kd_tree_TI_eb = VPhiTree(speeds, phis, ids=edge_ids, 
                        v_scale=v_scale, phi_scale=phi_scale)
    kino_eb_rrt = ConstrainedKinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb,
            use_fixed_sampling_time=False,
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
            num_skip_edges= 10,
            num_random_edges= 1,
            eb_kd_tree = kd_tree_TI_eb,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = 0, #Will be overwritten by KCBS init
            debug_flag=False,
            print_logs=False,
            )
    return kino_eb_rrt
    


"""
import sys
sys.path.append('./src')
from run_script_optimized_kcbs import *
from printer import *

num_agents = 5

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

agent_state_lengths = []
agent_radiuses = []
for agent in agents:
    agent_state_lengths.append(agent.state_length)
    agent_radiuses.append(agent.radius)

planners = []
planner_function = get_rrt_planner
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],0.5,agents[i],env))


# s = np.random.randint(0, 1000)
# print("RNG Seed: ", s)
s = 553  
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 600.0,
                    rng_seed = s,
                    print_logs=True
                    )
t = time.time()  
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Time taken for planning: ", t)

paths_dict, trees_dict, paths_cost = kcbs_planner.find_initial_paths()

kcbs_planner.find_first_collision(paths_dict)


for k, v in paths_dict[1].items():
    print(f"Key: {k}, Value type: {type(v)}")

for i in range(100):
    planners = []
    planner_function = get_rrt_planner
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],0.5,agents[i],env))

    s = np.random.randint(0, 1000)
    print("RNG Seed: ", s)  
    kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 600.0,
                    rng_seed = s,
                    print_logs=True,
                    debug_flag=True
                    )
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    sleep(1.0)


agents, starts, obstacles, goals, goal_radii = self.get_env_parms(seed)


p,t,c = find_initial_paths(planners)

p[1][500:600] = p[0][500:600]  # Introduce a collision for testing

find_first_collision_numba2(p, agent_state_lengths, agent_radiuses)

    
"""



"""
Visualize the paths found by the planner

from printer import MultiRRTPrinter

# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']

planner_list = [planners[i] for i in range(num_agents)]
mprint = MultiRRTPrinter(env, planner_list, [], [], [])
# The speed param is the time between each step in the gif (lower number->faster animation speed)
mprint.print_highres_simulation(paths, "./media/six_second_order_kcbs_rrt_sim.gif", animation_speed=50)

"""


"""
import sys
sys.path.append('./src')
from run_script_optimized_kcbs import *
from printer import *

num_agents = 6

pos1 = (2.0, 2.0, 0, 0.0, 0.0)
pos2 = (2.0, 5.0, 0, 0.0, 0.0)
pos3 = (2.0, 8.0, 0, 0.0, 0.0)
pos4 = (8.0, 2.0, 0, 0.0, 0.0)
pos5 = (8.0, 5.0, 0, 0.0, 0.0)
pos6 = (8.0, 8.0, 0, 0.0, 0.0)

starts = [pos1, pos2, pos3, pos4, pos5, pos6]
goals = [pos6[0:2], pos4[0:2], pos5[0:2], pos3[0:2], pos2[0:2], pos1[0:2]]
goal_radii = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

agent_ids = List()
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_second_order_car_agent(agent_id))

planners = []
planner_function = get_rrt_planner
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],0.5,agents[i],env))

agent_state_lengths = List()
agent_radiuses = List()
for agent in agents:
    agent_state_lengths.append(agent.state_length)
    agent_radiuses.append(agent.radius)


# s = np.random.randint(0, 1000)
# print("RNG Seed: ", s)
s = 610  
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 10,
                    planning_time = 600.0,
                    rng_seed = s,
                    print_logs=True
                    )

path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()


"""

"""
Empty

import sys
sys.path.append('./src')
from mapf_env_square_agent_second_order_car import *
from printer import *

obstacles = []                     
env = SquareEnvironment(40, 40, obstacles)

loc1 = (20, 30, 0 ,0, 0)
loc2 = (30, 20, 0 ,0, 0)
loc3 = (25, 10, 0 ,0, 0)
loc4 = (15, 10, 0 ,0, 0)
loc5 = (10, 20, 0 ,0, 0)

#3Agents
starts = [loc1, loc2, loc3]
goals = [loc3, loc4, loc5]
goal_radii = [2.0, 2.0, 2.0]


#4Agents
starts = [loc1, loc2, loc3, loc4]
goals = [loc3, loc4, loc5, loc1]
goal_radii = [2.0, 2.0, 2.0, 2.0]


#5Agents
starts = [loc1, loc2, loc3, loc4, loc5]
goals = [loc3, loc4, loc5, loc1, loc2]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]


"""

"""
Obstacles

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *

obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(18, 16, 3),
            CircularObstacle2D(20, 5, 3)
            ] 
                    
env = SquareEnvironment(40, 40, obstacles)

goal = (33.0, 10.0)
start = (14.0, 2.0, 0, 0.0, 0.0)

start2 = (5.0, 2.0, 0, 0.0, 0.0)
goal2 = (25.0, 25.0)

start3 = (30.0, 5.0, 0, 0.0, 0.0)
goal3 = (5.0, 30.0)

start4 = (30.0, 20.0, 0, 0.0, 0.0)
goal4 = (2.0, 10.0)

start5 = (27.0, 15.0, 0, 0.0, 0.0)
goal5 = (10.0, 3.0)


#3Agents
starts = [start, start2, start3]
goals = [goal, goal2, goal3]
goal_radii = [2.0, 2.0, 2.0]


#4Agents
starts = [start, start2, start3, start4]
goals = [goal, goal2, goal3, goal4]
goal_radii = [2.0, 2.0, 2.0, 2.0]


#5Agents
starts = [start, start2, start3, start4, start5]
goals = [goal, goal2, goal3, goal4, goal5]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]


"""


"""
#Cluttered Environment

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
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

goal = (24.0, 37.0)
start = (7.0, 5.0, 0)

start2 = (2.0, 26.0, 0)
goal2 = (37.0, 30.0)

start3 = (28.0, 5.0, 0)
goal3 = (5.0, 29.0)

start4 = (32.0, 18.0, 0)
goal4 = (2.0, 10.0)

start5 = (16.0, 37.0, 0)
goal5 = (36.0, 10.0)


#3Agents
starts = [start, start2, start3]
goals = [goal, goal2, goal3]
goal_radii = [2.0, 2.0, 2.0]



#4Agents
starts = [start, start2, start3, start4]
goals = [goal, goal2, goal3, goal4]
goal_radii = [2.0, 2.0, 2.0, 2.0]



#5Agents
starts = [start, start2, start3, start4, start5]
goals = [goal, goal2, goal3, goal4, goal5]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]



"""



"""

agents = {}
for agent_id in range(len(starts)):
    agents[agent_id] = get_unicycle_agent(agent_id)


cost_arr = []
time_arr = []
num_experiments = 20

for k in range(num_experiments):

    planners = {}
    for i in range(len(starts)):
        planners[agents[i].id] = planner_function(starts[i],goals[i],goal_radii[i],agents[i])

    kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 100.0
                    )

    st = time.time()
    path_found, paths, path_cost = kcbs_planner.plan_multi_agent_paths()
    # path_cost = sum(path_cost_dict.values())
    td = time.time() - st

    print("Cost for iteration ", k+1, " is : ",path_cost)
    cost_arr.append(path_cost)
    print("Time for iteration ", k+1, " is : ",td)
    time_arr.append(td)
                    
mc = np.mean(cost_arr)
sdc = np.std(cost_arr)
print( "Mean Cost: ", mc, " SDE: ", sdc/np.sqrt(num_experiments) )


mt = np.mean(time_arr)
sdt = np.std(time_arr)
print( "Mean Cost: ", mt, " SDE: ", sdt/np.sqrt(num_experiments) )


"""