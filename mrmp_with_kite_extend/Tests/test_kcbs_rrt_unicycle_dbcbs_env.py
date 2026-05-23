import sys
import os
# Ensure the repository root and `src` are on sys.path so tests can import
# top-level modules like `dcbs_envs.py` as well as the `src` package modules.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))
from Environments import *
from Agents import UniCycle
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from mapf_env_square_agent_unicycle import (
        get_unicycle_agent,
        get_rrt_planner, get_eb_rrt_planner, get_kino_TI_eb_rrt_planner_unicycle,
        get_constrained_db_rrt_planner_unicycle)
from printer import *
from dcbs_envs import *


weird_behavior_env_name = "gen_p10_n8_2_unicycle_sphere"
data_dict = envs[weird_behavior_env_name]

starts = data_dict['starts']
goals = data_dict['goals']
obstacles = data_dict['obs']

env = SquareEnvironment(10.0, 10.0, obstacles, obs_buffers=False)
num_agents = len(starts)
goal_radius = 0.8

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agent = UniCycle(agent_id = agent_id, 
                     max_speed = 0.5,
                     max_omega= 2.0,
                     radius = 0.4,
                     rng_seed= 42
                     )
    agents.append(agent)

goal_radii = [goal_radius for _ in range(num_agents)]

pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
MultiRRTPrinter.print_rrt_env('./media/env_db_cbs_' + str(num_agents) + '_' + str(weird_behavior_env_name) + '.png',
                                        env, agents, starts, goals, goal_radii, pcol)


planners = []
# planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
# s = 42  
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 3000.0,
                    rng_seed = s,
                    print_logs=True,
                    debug_flag=True
                    )
t = time.time()
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Planning Time: ", t)