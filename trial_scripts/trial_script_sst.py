import sys
sys.path.append('./src')
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar, UniCycle
from rrt import RRT
from sst import SST
from sst_printer import SSTPrinter
import numpy as np


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
# obstacles= []
env = SquareEnvironment(40, 40, obstacles)
agent = SecondOrderCar(agent_id = 1, 
                       max_speed = 2.0,
                       max_acceleration = 1.0,
                       max_phi = np.pi/3,
                       max_steering_rate = 0.5,
                       radius = 0.3,
                       wheelbase = 0.7,
                       rng_seed=42
                       )
start = np.array([7.0, 5.0, 0, 0.0, 0.0])
goal = np.array([24.0, 37.0])
# goal = (25.0, 25.0)
goal_radius = 0.5


s = np.random.randint(0, 1000)
sst  = SST( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True,
            # debug_flag=True,
            best_near_radius=5.0
           )

sst.plan_path()
sst.path_time

node_ids, states, actions, timesteps = sst.get_path_to_node_id(800)
node_ids, states, actions, timesteps = sst.get_path()

printer = SSTPrinter(env, sst)
printer.print_sst("media/sst_graph.png",
                  show_tree=True, show_path=True,
                  show_active=True, show_inactive=False,
                  show_witness=True, show_prune_circles=False)


# optional animation
# printer.print_sst_step_ani("media/sst_build.gif", animation_speed=2)


"""
#Check if the returned states and actions to find the path are correct

node_ids, states, actions, timesteps = sst.get_path_to_node_id(sst.goal_node_id)
for i in range(len(node_ids)-1):
    parent_state = states[i]
    action = actions[i]
    duration = timesteps[i]
    num_steps = round(duration/sst.minimum_time_step)
    next_state, _ = agent.get_next_state(parent_state, action, duration, num_steps=num_steps)
    print("Next State from propagation: ", next_state)
    print("Stored State in SST: ", states[i+1])

"""


"""
**********************************
With Unicycle Agent
**********************************
"""

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
# obstacles= []
env = SquareEnvironment(40, 40, obstacles)
agent = UniCycle(agent_id = 1, 
                 max_speed = 2.0,
                 max_omega= np.pi/2,
                 radius = 1.0,
                 rng_seed= 77)
start = np.array([7.0, 5.0, 0.0])
goal = np.array([24.0, 37.0])
# goal = (25.0, 25.0)
goal_radius = 0.5


s = np.random.randint(0, 1000)
sst  = SST( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True,
            # debug_flag=True,
            best_near_radius=5.0
           )

sst.plan_path()
sst.path_time
node_ids, states, actions, timesteps = sst.get_path()

printer = SSTPrinter(env, sst)
printer.print_sst("media/sst_graph.png",
                  show_tree=True, show_path=True,
                  show_active=True, show_inactive=False,
                  show_witness=True, show_prune_circles=False)


# optional animation
# printer.print_sst_step_ani("media/sst_build.gif", animation_speed=2)

