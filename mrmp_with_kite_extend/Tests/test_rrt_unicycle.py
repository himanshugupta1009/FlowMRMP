import sys
sys.path.append('./src')
from Environments import *
from Agents import UniCycle
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from mapf_env_square_agent_unicycle import get_unicycle_agent, \
        get_rrt_planner, get_eb_rrt_planner, get_kino_TI_eb_rrt_planner_unicycle
from printer import *

starts = [
    (1.0, 1.0, 0.0),
    (6.0, 5.0, 0.0),
    (12.0, 5.0, 0.0),
    (14.0, 4.0, 0.0),
    (9.0, 10.0, 0.0),
    (3.0, 11.0, 0.0),
    (6.0, 14.0, 0.0),
    (8.0, 5.0, 0.0),
    (10.0, 1.0, 0.0),
    (11.0, 14.0, 0.0),
    (13.0, 10.0, 0.0),
    (4.0, 8.0, 0.0),
    (9.0, 7.0, 0.0),
    (8.0, 12.0, 0.0),
    (1.0, 6.0, 0.0),
    (14.0, 12.0, 0.0),
    (4.0, 3.0, 0.0),
    (14.0, 1.0, 0.0),
    (6.0, 8.0, 0.0),
    (7.0, 1.0, 0.0)

]

goals = [
    (6.0, 3.0),
    (2.0, 11.0),
    (5.0, 12.0),
    (9.0, 1.0),
    (14.0, 3.0),
    (12.0, 12.0),
    (3.0, 7.0),
    (13.0, 6.0),
    (11.0, 8.0),
    (3.0, 14.0),
    (12.0, 3.0),
    (1.0, 3.0),
    (5.0, 9.0),
    (4.0, 5.0),
    (10.0, 6.0),
    (8.0, 3.0),
    (6.0, 11.0),
    (9.0, 8.0),
    (10.0, 12.0), 
    (3.0, 9.0)
]

obstacles = [
    RectangleObstacle2D(4.5, 1.5, 3, 1),
    RectangleObstacle2D(12.5, 1.5, 1, 1),
    RectangleObstacle2D(2.0, 4.5, 2, 1),
    RectangleObstacle2D(10.0, 3.5, 2, 1),
    RectangleObstacle2D(0.5, 7.5, 1, 1),
    RectangleObstacle2D(1.5, 9.5, 1, 1),
    RectangleObstacle2D(13.5, 8.0, 1, 2),
    RectangleObstacle2D(7.5, 8.0, 1, 4),
    RectangleObstacle2D(2.5, 12.5, 3, 1),
    RectangleObstacle2D(10.5, 10.5, 1, 1),
    RectangleObstacle2D(7.5, 14.5, 1, 1),
    RectangleObstacle2D(13.0, 14.0, 2, 2),
]

env = SquareEnvironment(15.0, 15.0, obstacles, obs_buffers=False)

num_agents = 1
goal_radius = 0.5

agents = get_unicycle_agent(1)

planner = get_rrt_planner(starts[0], goals[0], goal_radius, agents, env)
planner_function = get_eb_rrt_planner
planner = get_kino_TI_eb_rrt_planner_unicycle(starts[0], goals[0], 
                                            goal_radius, agents, env)


s = np.random.randint(0, 1000)
s = 42  
print("RNG Seed: ", s)

planner.print_logs = True
# planner.debug_flag = True
# planner.max_iter = 5
planner.rng = np.random.default_rng(s)
planner.plan_path()