import sys
sys.path.append('./src')
import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from utils import euclidean_distance, euclidean_distance_satisfaction_numba
from edge_bundle_rrt import * 
from edge_bundle import EdgeBundle
from printer import *


start = (5.0, 5.0, 0.0)
goal = (25.0, 25.0)
goal_radius = 1.0
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
                    
# obstacles = []
env = SquareEnvironment(40, 40, obstacles)
agent = UniCycle(agent_id = 1, 
                 max_speed = 2.0,
                 max_omega= np.pi/2,
                 radius = 1.0,
                 rng_seed= 77)


edge_bundle_file_location = 'edge_bundles/eb_unicycle_edges_100000.npz' 
data = np.load(edge_bundle_file_location)
eb_unicycle = EdgeBundle(data, fix_num_edges=1000)

s = np.random.randint(0, 1000)
eb_rrt  = EdgeBundleType2RRT( 
                        start=start, goal=goal,
                        goal_radius=goal_radius,
                        env = env, agent=agent, 
                        edge_bundle = eb_unicycle,
                        use_fixed_sampling_time=False,
                        sampling_time_step=1.5,
                        minimum_time_step=0.1,
                        max_iter = 100000,
                        num_random_edges= 10,
                        num_skip_edges= 100,
                        planning_time=600.0,
                        isvalid_function=agent.is_new_node_valid,
                        cost_function=agent.get_cost,
                        random_point_function=agent.get_random_point,
                        reached_goal_function = agent.agent_reached_goal,
                        translate_function = agent.point_translate_function,
                        sort_edges_function=agent.sort_edges,
                        udf_seed = s,
                        debug_flag=False,
                        print_logs=True
                        )

eb_rrt.plan_path()
rrt_node_ids, states, actions, timesteps = eb_rrt.get_path()
v = RRTPrinter(env, eb_rrt, rrt_node_ids)
v.print_rrt('media/edge_bundle_rrt_graph_unicycle.png')



rrt  = RRT( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=600.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function=agent.agent_reached_goal,
            udf_seed = s,
            debug_flag=False,
            print_logs=True,
           )
rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_unicycle.png')


from pyinstrument import Profiler

s = np.random.randint(0, 1000)
s = 730
eb_rrt  = EdgeBundleType2RRT( 
                        start=start, goal=goal,
                        goal_radius=goal_radius,
                        env = env, agent=agent, 
                        edge_bundle = eb_unicycle,
                        use_fixed_sampling_time=False,
                        sampling_time_step=1.5,
                        minimum_time_step=0.1,
                        max_iter = 100000,
                        num_random_edges= 10,
                        num_skip_edges= 100,
                        planning_time=600.0,
                        isvalid_function=agent.is_new_node_valid,
                        cost_function=agent.get_cost,
                        random_point_function=agent.get_random_point,
                        reached_goal_function = agent.agent_reached_goal,
                        translate_function = agent.point_translate_function,
                        sort_edges_function=agent.sort_edges,
                        udf_seed = s,
                        debug_flag=False,
                        print_logs=False,
                        )
p = Profiler(async_mode='disabled')
p.start()
eb_rrt.plan_path()
p.stop()
print(p.output_text(unicode=True, color=True))
p.open_in_browser()  # Opens the profiler output in a web browser

















"""


constrained_rrt = ConstrainedRRT(
                                start = start, goal = goal,
                                goal_radius = goal_radius,
                                env = env,
                                agent = agent,
                                sampling_time_step = 1.5,
                                minimum_time_step=0.1,
                                max_iter = 10000,
                                planning_time = 600.0,
                                isvalid_function=is_new_node_valid,
                                cost_function=get_cost,
                                random_point_function=get_random_point,
                                reached_goal_function = agent_reached_goal,
                                udf_seed = np.random.randint(0, 1000),
                                # udf_seed = 7
                                )

arr = np.arange(1.0, 100.1, 0.1)
time_array = np.round(arr, 1)
position_array1 = np.tile( (15.0, 10.0, 0.0) , (len(time_array),1) )
conflict1 = ConflictBlock(agent.id,position_array1,time_array,agent) 
position_array2 = np.tile( (14.0, 10.0, 0.0) , (len(time_array),1) )
conflict2 = ConflictBlock(agent.id,position_array2,time_array,agent) 
position_array3 = np.tile( (16.0, 10.0, 0.0) , (len(time_array),1) )
conflict3 = ConflictBlock(agent.id,position_array3,time_array,agent) 
position_array4 = np.tile( (17.0, 10.0, 0.0) , (len(time_array),1) )
conflict4 = ConflictBlock(agent.id,position_array4,time_array,agent) 
position_array5 = np.tile( (18.0, 10.0, 0.0) , (len(time_array),1) )
conflict5 = ConflictBlock(agent.id,position_array5,time_array,agent) 
# position_array6 = np.tile( (19.0, 10.0, 0.0) , (len(time_array),1) )
# conflict6 = ConflictBlock(agent.id,position_array6,time_array,agent) 
cons = [conflict1,conflict2,conflict3,conflict4,conflict5]
constrained_rrt.set_constraints(cons)



import sys
sys.path.append('./src')
from env_square_agent_unicycle import *
from printer import *

rrt.plan_path()
rrt_node_ids, states, actions, timesteps = rrt.get_path()
v = RRTPrinter(env, rrt,rrt_node_ids)
v.print_rrt('media/rrt_graph_unicycle.png')


import sys
sys.path.append('./src')
from env_square_agent_unicycle import *
from printer import *

eb_rrt.plan_path()
rrt_node_ids, states, actions, timesteps = eb_rrt.get_path()
v = RRTPrinter(env, eb_rrt, rrt_node_ids)
v.print_rrt('media/edge_bundle_rrt_graph_unicycle.png')



import sys
sys.path.append('./src')
from env_square_agent_unicycle import *
from printer import *

constrained_rrt.plan_path()
rrt_node_ids, states, actions, timesteps = constrained_rrt.get_path() 
v = RRTPrinter(env,constrained_rrt,rrt_node_ids)
v.print_rrt('media/constrained_rrt_graph_unicycle.png')


"""