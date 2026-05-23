import sys
sys.path.append('./src')
from Environments import *
from Agents import UniCycle
from utils import euclidean_distance
from cRRT import *
from printer import *
from dcbs_envs import *


weird_behavior_env_name = "gen_p10_n8_2_unicycle_sphere"
data_dict = envs[weird_behavior_env_name]

starts = data_dict['starts']
goals = data_dict['goals']
obstacles = data_dict['obs']

obstacles = []
env = SquareEnvironment(10.0, 10.0, obstacles, obs_buffers=False)
num_agents = len(starts)
goal_radius = 0.4

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


# curate list of helper functions for call to CRRT 
# each of these functions must match the agent at the same index
isvalid_funcs = [UniCycle.is_new_node_valid for _ in range(len(agents))]
cost_funcs = [UniCycle.get_cost for _ in range(len(agents))]
random_pt_funcs = [UniCycle.get_random_point for _ in range(len(agents))]
reached_goal_funcs = [UniCycle.agent_reached_goal for _ in range(len(agents))]


s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
# s = 42  
crrt = CRRT(agents=agents, 
            starts=starts,
            goals=goals,
            goal_radii=goal_radii,
            env=env,
            max_iter = math.inf, planning_time=math.inf,         
            isvalid_function=isvalid_funcs, 
            cost_function=cost_funcs,
            random_point_function=random_pt_funcs, 
            reached_goal_function = reached_goal_funcs,
            udf_seed = s, 
            print_logs=True
            )
# plan path 
planning_time = crrt.plan_path()

path_rrt_nodes, states, controls, timesteps, costs = crrt.get_path()

# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']

mprint = MultiRRTPrinter(env, crrt, path_rrt_nodes, tcol, pcol, joint_states=True)
# setting print_tree to true will show the tree as well
mprint.print_rrt('./media/db_cbs_8agents_crrt_unicycle.png', print_tree=False)

mprint.print_rrt_animation('./media/db_cbs_8agents_crrt_unicycle.gif', 
                           animation_speed=150, print_tree=False)