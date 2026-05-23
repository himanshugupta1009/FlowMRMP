import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import DubinsCar
from utils import euclidean_distance, euclidean_distance_satisfaction_numba
from rrt import RRT

def is_state_valid(env, agent, state):

    if(state[0] < (agent.radius + env.boundary_buffer) 
            or state[0] > (env.size[0]-agent.radius-env.boundary_buffer)):
        return False
    if(state[1] < (agent.radius + env.boundary_buffer)
            or state[1] > (env.size[1]-agent.radius-env.boundary_buffer)):
        return False
    
    for obs in env.obstacles:
        if obs.check_collision(state, agent.radius, obstacle_buffer=env.obstacle_buffer):
            return False
    return True

def is_state_valid_moving_obs(env, agent, state, time):
    """Check if a single state is valid for an environment with moving 
    obstacles (i.e. obstacles with a time dimension)
    
    :PRE: Environment contains circular obstacles with a TIME COMPONENT ONLY!

    Args:
        env: Environment object 
        agent: DubinsCar agent object
        state (DubinsCar state tuple): agent location
        time (float): current time, fractional seconds 

    Returns:
        bool: True if new state valid, false else 
    """
    # check if agent state x component is within the env
    if(state[0] < (agent.radius + env.boundary_buffer) 
            or state[0] > (env.size[0]-agent.radius-env.boundary_buffer)):
        return False
    # check if agent state y component is within the env
    if(state[1] < (agent.radius + env.boundary_buffer)
            or state[1] > (env.size[1]-agent.radius-env.boundary_buffer)):
        return False
    
    # For each obstacle...
    for obs in env.obstacles:
        # check if it is far enough away to not be a collision 
        if obs.check_collision(state, agent.radius, t=time, obstacle_buffer=env.obstacle_buffer):
            return False
    return True

def is_new_node_valid_moving_obs(env, agent, path_to_new_state, start_time, interval):
    """
    Checks if a new state and the path to that new state is valid for 
    an environment with moving obstacles (i.e. obstacles with a time dimension)

    :PRE: Environment contains circular obstacles with a TIME COMPONENT ONLY!

    Args:
        env: Environment object 
        agent: DubinsCar agent object
        path_to_new_state (list[dubins car state tuple]): Path taken by the agent 
            from the parent state, non-inclusive of the parent state but inclusive
            of the final state 
        start_time (float): timepoint agent leaves the parent state
        interval (float): length of time the agent takes to complete the path 

    Returns:
        bool: True if new node is valid, false else 
    """
    # find the interval of time between each step in the path
    dt = interval/len(path_to_new_state)
    # since the path is not inclusive of the parent node state, 
    # the time for the first item in the path is one step away 
    # from the start time 
    t = start_time + dt
    # for each state...
    for state in path_to_new_state:
        # check if the path entry is not colliding with the 
        # env boundary or any obstacles 
        if not is_state_valid_moving_obs(env, agent, state, t):
            return False
        t += dt 
    return True

def is_new_node_valid(env, agent, path_to_new_state):
    for state in path_to_new_state:
        if not is_state_valid(env, agent, state):
            return False
    return True


def get_cost(env, agent, s, a, t, edge): 
    sp = edge[-1]
    x, y, theta, v = sp
    distance_covered = v * t
    return distance_covered


def agent_reached_goal(state, goal, goal_radius, agent):
    return euclidean_distance_satisfaction_numba(state, goal, goal_radius)


def get_random_point(env, agent, rng):
    # x = rng.uniform(0, env.size[0])
    # y = rng.uniform(0, env.size[1])
    x, y = rng.uniform([0, 0], env.size)
    p = (x, y)

    while not is_state_valid(env, agent, p):
        x, y = rng.uniform([0, 0], env.size)
        # theta = rng.uniform(0, 2 * np.pi)
        # p = (x, y, theta)
        p = (x, y)
        
    return np.array(p, dtype=np.float64)

"""
start = (5.0, 5.0, 0.0, 0.0)
goal = (25.0, 25.0)
goal_radius = 1.0
obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(15, 17, 2),
            CircularObstacle2D(20, 6, 3)
            ] 
                    
env = SquareEnvironment(40, 40, obstacles)
agent = DubinsCar(2.0, 1.0, 1.0, 77)
edge_bundle = None

print("Creating RRT")
rrt  = RRT( start=start, goal=goal, goal_radius=goal_radius, 
           env = env, agent=agent, max_iter = 5000, planning_time=60.0,         
           isvalid_function=is_new_node_valid, cost_function=get_cost,
           random_point_function=get_random_point, 
           reached_goal_function = agent_reached_goal,
           udf_seed = 7
           )

path_rrt_nodes, states, controls = rrt.plan_path()
"""

"""
import sys
sys.path.append('./src')
from env_square_agent_dubins import *


from printer import *
v = RRTPrinter(env, rrt, states)
v.print_rrt('media/rrt_graph.png')


a,b,c = rrt.plan_path()

"""