import sys
sys.path.append('./src')
from Environments import RectangleObstacle2D, SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from rrt import RRT
from printer import *


obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 20, 4),
            CircularObstacle2D(20, 6, 3),
            ] 
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
# obstacles = [
#             CircularObstacle2D(10, 10, 2),
#             CircularObstacle2D(16, 25, 3),
#             CircularObstacle2D(20, 5, 2),
#             CircularObstacle2D(35, 15, 2),
#             CircularObstacle2D(30, 34, 4),
#             CircularObstacle2D(25, 15, 4),
#             # RectangleObstacle2D(7, 19, 4, 4),
#             ]
obstacles= []                    
env = SquareEnvironment(40, 40, obstacles)
start = np.array([5.0, 5.0, 0.0])
goal = np.array([25.0, 25.0])
goal_radius = 0.5


#BugTrap 

# obstacles = [
#     RectangleObstacle2D(4.5, 3.0, 0.2, 3.2),
#     RectangleObstacle2D(3.0, 1.5, 3.2, 0.2),
#     RectangleObstacle2D(3.0, 4.5, 3.2, 0.2),
#     RectangleObstacle2D(1.5, 4.05, 0.2, 1.1),
#     RectangleObstacle2D(1.5, 1.95, 0.2, 1.1),
# ]

# env = SquareEnvironment(10.0, 10.0, obstacles, obs_buffers=False)
# start = np.array((3.8, 3.0, 0.0), dtype=np.float64)   # x, y, theta
# goal = np.array((9.2, 9.2), dtype=np.float64)         # x, y
# goal_radius = 0.5

agent = get_unicycle_agent(1)
edge_bundle = None

s = np.random.randint(0, 1000)
rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.0,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            max_iter = 10000,
            num_extension_trials=10,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_unicycle.png')

import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from rrt import RRT
from printer import *

from pyinstrument import Profiler

obstacles= []
obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 20, 4),
            CircularObstacle2D(20, 6, 3),
            # CircularObstacle2D(35, 15, 2),
            # CircularObstacle2D(10, 30, 5),
            # CircularObstacle2D(25, 15, 5),
            # CircularObstacle2D(7, 17, 5),
            # CircularObstacle2D(16, 13, 2),
            ] 
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
                    

env = SquareEnvironment(33, 33, obstacles)
agent = get_unicycle_agent(1)
edge_bundle = None
start = np.array([5.0, 5.0, 0.0, 1.0, 0.0])
goal = np.array([25.0, 25.0])
goal_radius = 0.5


rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.0,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            num_extension_trials=10,
            max_iter = np.inf,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = np.random.randint(0, 1000)
            # udf_seed = 545
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_unicycle.png')

s = np.random.randint(0, 1000)
rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.0,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            num_extension_trials=10,
            max_iter = 10000,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True
           )
rrt.plan_path()
t = time.time()
d = rrt.get_high_resolution_path()
t = time.time() - t
print(f"Time taken to get high resolution path: {t:.11f} seconds")
t = time.time()
p = rrt.get_high_resolution_path_numpy_array()
t = time.time() - t
print(f"Time taken to get high resolution path 2: {t:.11f} seconds")
t = time.time()
q = rrt.get_high_resolution_path3()
t = time.time() - t
print(f"Time taken to get high resolution path 3: {t:.11f} seconds")

import time
num_trials = 100
time_array= np.zeros(num_trials)
for i in range(num_trials):
    s = np.random.randint(0, 1000)
    rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.0,
            minimum_time_step=0.1,
            num_extension_trials=10,
            max_iter = np.inf,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s
            # udf_seed = 7
           )
    start_time = time.time()
    rrt.plan_path()
    end_time = time.time()
    t = end_time - start_time
    print(f"Trial {i}: Time taken = {t:.2f} seconds and seed = {s}")
    time_array[i] = t




p = Profiler()
p.start()
rrt.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  # Opens the profiler output in a web browser



kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 600.0
                    )  
p = Profiler(async_mode='disabled')
p.start()
# Plan paths for multiple agents using KCBS
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  # Opens the profiler output in a web browser