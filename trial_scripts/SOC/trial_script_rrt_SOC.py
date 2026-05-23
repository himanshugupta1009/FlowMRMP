import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_second_order_car import get_second_order_car_agent
from rrt import RRT
from printer import *


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
edge_bundle = None
# start = np.array([5.0, 5.0, 0.0, 1.0, 0.0])
start = np.array([7.0, 5.0, 0, 0.0, 0.0])
goal = np.array([24.0, 37.0])
# goal = np.array([25.0, 25.0])
goal_radius = 0.5

s = np.random.randint(0, 1000)
rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=2.0,
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
v.print_rrt('media/rrt_graph_second_order_car.png')

import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar
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
                    

env = SquareEnvironment(33, 33, obstacles)
# agent = SecondOrderCar(agent_id = 1, 
#                        max_speed = 2.0,
#                        max_acceleration = 1.0,
#                        max_phi = np.pi/3,
#                        max_steering_rate = 0.5,
#                        radius = 0.3,
#                        wheelbase = 0.7,
#                        rng_seed=42
#                        )
agent = get_second_order_car_agent(1)
edge_bundle = None
start = np.array([5.0, 5.0, 0.0, 1.0, 0.0])
goal = np.array([25.0, 25.0])
goal_radius = 0.5


rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
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
v.print_rrt('media/rrt_graph_second_order_car.png')

s = np.random.randint(0, 1000)
rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
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
            sampling_time_step=1.5,
            minimum_time_step=0.1,
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





import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_second_order_car import get_second_order_car_agent
from rrt import RRT
from modified_rrt import GoalTruncatedRRTType1, GoalTruncatedRRTType2, MatrixNNGoalTruncatedRRTType1
from printer import *

from pyinstrument import Profiler

obstacles = [
            # CircularObstacle2D(10, 10, 2),
            # CircularObstacle2D(16, 20, 4),
            # CircularObstacle2D(20, 6, 3)
            ] 
env = SquareEnvironment(33, 33, obstacles)
# agent = SecondOrderCar(agent_id = 1, 
#                        max_speed = 2.0,
#                        max_acceleration = 1.0,
#                        max_phi = np.pi/3,
#                        max_steering_rate = 0.5,
#                        radius = 0.3,
#                        wheelbase = 0.7,
#                        rng_seed=42
#                        )
agent = get_second_order_car_agent(1)
edge_bundle = None
# start = np.array([5.0, 5.0, 0.0, 1.0, 0.0], dtype=np.float64)
start = (5.0, 5.0, 0.0, 1.0, 0.0)
goal = np.array([25.0, 25.0], dtype=np.float64)
goal_radius = 0.5


s =  np.random.randint(0, 1000)
print(f"Using seed: {s}")
num_trials = np.inf
# num_trials = 242
# num_trials = 283

# rrt  = GoalTruncatedRRTType1(
#             start=start, goal=goal,
#             goal_radius=goal_radius, 
#             env = env, agent=agent,
#             use_fixed_sampling_time=False,
#             sampling_time_step=1.5,
#             minimum_time_step=0.1,
#             max_iter = num_trials,
#             planning_time=300.0,         
#             isvalid_function=agent.is_new_node_valid,
#             cost_function=agent.get_cost,
#             random_point_function=agent.get_random_point, 
#             reached_goal_function = agent.agent_reached_goal,
#             udf_seed = s,
#             print_logs= False,
#            )

# rrt.plan_path()
# rrt_node_ids, states, actions, timesteps = rrt.get_path()
# v = RRTPrinter(env, rrt,rrt_node_ids)
# v.print_rrt('media/rrt_graph_second_order_car_t1.png')

# rrt  = GoalTruncatedRRTType2(
#             start=start, goal=goal,
#             goal_radius=goal_radius, 
#             env = env, agent=agent,
#             use_fixed_sampling_time=False,
#             sampling_time_step=1.5,
#             minimum_time_step=0.1,
#             max_iter = num_trials,
#             planning_time=300.0,         
#             isvalid_function=agent.is_new_node_valid,
#             cost_function=agent.get_cost,
#             random_point_function=agent.get_random_point, 
#             reached_goal_function = agent.agent_reached_goal,
#             udf_seed = s,
#            )

# rrt.plan_path()
# rrt_node_ids, states, actions, timesteps = rrt.get_path()
# v = RRTPrinter(env, rrt,rrt_node_ids)
# v.print_rrt('media/rrt_graph_second_order_car_t2.png')


rrt  = MatrixNNGoalTruncatedRRTType1(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

rrt.plan_path()

# rrt_node_ids, states, actions, timesteps = rrt.get_path()
# v = RRTPrinter(env, rrt,rrt_node_ids)
# v.print_rrt('media/rrt_graph_second_order_car_linalgNN.png')

from modified_rrt import MatrixNNGoalTruncatedRRTType2
rrt  = MatrixNNGoalTruncatedRRTType2(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

rrt.plan_path()
# rrt_node_ids, states, actions, timesteps = rrt.get_path()
# v = RRTPrinter(env, rrt,rrt_node_ids)
# v.print_rrt('media/rrt_graph_second_order_car_linalgNN2.png')

from modified_rrt import MatrixNNGoalTruncatedRRTType3
rrt  = MatrixNNGoalTruncatedRRTType3(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

rrt.plan_path()
# rrt_node_ids, states, actions, timesteps = rrt.get_path()
# v = RRTPrinter(env, rrt,rrt_node_ids)
# v.print_rrt('media/rrt_graph_second_order_car_linalgNN3.png')



rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_second_order_car.png')


agent.get_next_state([0,0,0,0,0], (1,1), 5.0, num_steps=10)


"""
Seeds to test out on:

max_speed=2.0, steering_rate=1.0
644
770

max_speed=1.0
806
"""

random_point = rrt.sample_random_point()
nearest_node_id, nearest_node = rrt.get_nearest_node(random_point)
rrt.extend_tree(nearest_node_id, nearest_node, random_point)


s = 803
rrt  = MatrixNNGoalTruncatedRRTType2(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

p = Profiler(async_mode='disabled')
p.start()
rrt.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  # Opens the profiler output in a web browser


from modified_rrt import MatrixNNGoalTruncatedRRTType2
s = 434
rrt  = MatrixNNGoalTruncatedRRTType1(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = num_trials,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_second_order_car_linalgNN2.png')


from pyinstrument import Profiler

s = np.random.randint(0, 1000)
rrt  = RRT(
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = np.inf,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            # print_logs= True,
           )

p = Profiler(async_mode='disabled')
p.start()
rrt.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  # Opens the profiler output in a web browser
