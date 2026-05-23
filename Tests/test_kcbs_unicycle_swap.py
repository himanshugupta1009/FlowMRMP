import sys
sys.path.append('./src')
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


chose_num_agents = 20
swap_type = 'circle'
# swap_type = 'square'

if chose_num_agents == 10:
    if swap_type == 'circle':
        starts = [
        (52.000000000000000, 30.000000000000000, 3.141592653589793),
        (46.062305898749052, 43.062305898749052, 3.769911184307752),
        (35.062305898749052, 52.062305898749052, 4.398229715025710),
        (24.937694101250948, 52.062305898749052, 5.026548245743669),
        (13.937694101250946, 43.062305898749052, 5.654866776461628),
        (8.000000000000000, 30.000000000000000, 0.000000000000000),
        (13.937694101250946, 16.937694101250948, 0.628318530717959),
        (24.937694101250948, 7.937694101250948, 1.256637061435917),
        (35.062305898749052, 7.937694101250948, 1.884955592153876),
        (46.062305898749052, 16.937694101250948, 2.513274122871834),
        ]

        goals = [
            (8.000000000000000, 30.000000000000000),
            (13.937694101250946, 16.937694101250948),
            (24.937694101250948, 7.937694101250948),
            (35.062305898749052, 7.937694101250948),
            (46.062305898749052, 16.937694101250948),
            (52.000000000000000, 30.000000000000000),
            (46.062305898749052, 43.062305898749052),
            (35.062305898749052, 52.062305898749052),
            (24.937694101250948, 52.062305898749052),
            (13.937694101250946, 43.062305898749052),
        ]
    elif swap_type == 'square':
        starts = [
            # Group A: bottom-left -> top-right
            (10.0, 10.0, math.atan2(40.0, 40.0)),
            (10.0, 18.0, math.atan2(32.0, 40.0)),
            (10.0, 26.0, math.atan2(24.0, 40.0)),
            (10.0, 34.0, math.atan2(16.0, 40.0)),
            (10.0, 42.0, math.atan2(8.0, 40.0)),

            # Group B: top-left -> bottom-right
            (10.0, 50.0, math.atan2(-40.0, 40.0)),
            (10.0, 42.0, math.atan2(-32.0, 40.0)),
            (10.0, 34.0, math.atan2(-24.0, 40.0)),
            (10.0, 26.0, math.atan2(-16.0, 40.0)),
            (10.0, 18.0, math.atan2(-8.0, 40.0)),
        ]

        goals = [
            # Goals for Group A (to the right side, upward)
            (50.0, 50.0),
            (50.0, 50.0 - 8.0),   # 42
            (50.0, 50.0 - 16.0),  # 34
            (50.0, 50.0 - 24.0),  # 26
            (50.0, 50.0 - 32.0),  # 18

            # Goals for Group B (to the right side, downward)
            (50.0, 10.0),
            (50.0, 10.0 + 8.0),   # 18
            (50.0, 10.0 + 16.0),  # 26
            (50.0, 10.0 + 24.0),  # 34
            (50.0, 10.0 + 32.0),  # 42
        ]
    

if chose_num_agents == 20:
    if swap_type == 'circle':
        starts = [
            (68.000000000000000, 40.000000000000000, 3.141592653589793),
            (66.629582456264302, 48.652475842498525, 3.455751918948772),
            (62.652475842498532, 56.457987064189247, 3.769911184307752),
            (56.457987064189247, 62.652475842498532, 4.084070449666731),
            (48.652475842498532, 66.629582456264302, 4.398229715025710),
            (40.000000000000000, 68.000000000000000, 4.712388980384690),
            (31.347524157501471, 66.629582456264302, 5.026548245743669),
            (23.542012935810749, 62.652475842498532, 5.340707511102648),
            (17.347524157501471, 56.457987064189247, 5.654866776461628),
            (13.370417543735698, 48.652475842498525, 5.969026041820607),
            (12.000000000000000, 40.000000000000000, 0.000000000000000),
            (13.370417543735698, 31.347524157501475, 0.314159265358979),
            (17.347524157501468, 23.542012935810757, 0.628318530717959),
            (23.542012935810749, 17.347524157501475, 0.942477796076938),
            (31.347524157501468, 13.370417543735702, 1.256637061435917),
            (40.000000000000000, 12.000000000000000, 1.570796326794897),
            (48.652475842498525, 13.370417543735698, 1.884955592153876),
            (56.457987064189247, 17.347524157501468, 2.199114857512855),
            (62.652475842498525, 23.542012935810749, 2.513274122871834),
            (66.629582456264302, 31.347524157501468, 2.827433388230814),
        ]

        goals = [
            (12.000000000000000, 40.000000000000000),
            (13.370417543735698, 31.347524157501475),
            (17.347524157501468, 23.542012935810757),
            (23.542012935810749, 17.347524157501475),
            (31.347524157501468, 13.370417543735702),
            (40.000000000000000, 12.000000000000000),
            (48.652475842498525, 13.370417543735698),
            (56.457987064189247, 17.347524157501468),
            (62.652475842498525, 23.542012935810749),
            (66.629582456264302, 31.347524157501468),
            (68.000000000000000, 40.000000000000000),
            (66.629582456264302, 48.652475842498525),
            (62.652475842498532, 56.457987064189247),
            (56.457987064189247, 62.652475842498532),
            (48.652475842498532, 66.629582456264302),
            (40.000000000000000, 68.000000000000000),
            (31.347524157501471, 66.629582456264302),
            (23.542012935810749, 62.652475842498532),
            (17.347524157501471, 56.457987064189247),
            (13.370417543735698, 48.652475842498525),
        ]

    
    elif swap_type == 'square':
        starts = [
                # Left side -> heading east (0 rad)
                (10.0, 7.5, 0.0),
                (10.0, 12.5, 0.0),
                (10.0, 17.5, 0.0),
                (10.0, 22.5, 0.0),
                (10.0, 27.5, 0.0),
                (10.0, 32.5, 0.0),
                (10.0, 37.5, 0.0),
                (10.0, 42.5, 0.0),
                (10.0, 47.5, 0.0),
                (10.0, 52.5, 0.0),

                # Right side -> heading west (pi rad)
                (50.0, 7.5, math.pi),
                (50.0, 12.5, math.pi),
                (50.0, 17.5, math.pi),
                (50.0, 22.5, math.pi),
                (50.0, 27.5, math.pi),
                (50.0, 32.5, math.pi),
                (50.0, 37.5, math.pi),
                (50.0, 42.5, math.pi),
                (50.0, 47.5, math.pi),
                (50.0, 52.5, math.pi),
            ]

        goals = [
                # Goals for left starters (to right)
                (50.0, 7.5),
                (50.0, 12.5),
                (50.0, 17.5),
                (50.0, 22.5),
                (50.0, 27.5),
                (50.0, 32.5),
                (50.0, 37.5),
                (50.0, 42.5),
                (50.0, 47.5),
                (50.0, 52.5),

                # Goals for right starters (to left)
                (10.0, 7.5),
                (10.0, 12.5),
                (10.0, 17.5),
                (10.0, 22.5),
                (10.0, 27.5),
                (10.0, 32.5),
                (10.0, 37.5),
                (10.0, 42.5),
                (10.0, 47.5),
                (10.0, 52.5),
            ]


obstacles = []
env = SquareEnvironment(80.0, 80.0, obstacles)

num_agents = len(starts)
goal_radius = 1.0
goal_radii = [goal_radius for _ in range(num_agents)]


agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))


pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
MultiRRTPrinter.print_rrt_env('./media/env_swap_' + str(num_agents) + 'agents' + '.png',
                            env, agents, starts, goals, goal_radii, pcol)



planners = []
planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 10000,
                    planning_time = 200.0,
                    rng_seed = s,
                    print_logs=True,
                    debug_flag=False
                    )
t = time.time()
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Planning Time: ", t)


planner_list = [planners[i] for i in range(num_agents)]

path_node = kcbs_planner.path_cbs_node
dict_paths = {}
ids_list = []
for i in range(num_agents):
    t = path_node.agent_trees[i]
    planners[i].tree = t[0]
    planners[i]._node_matrix = t[1]
    goal_id = next(reversed(t[0]._node))
    planners[i].goal_node_id = goal_id
    planners[i].path_found = True
    dict_paths[i] = planners[i].get_high_resolution_path()
    ids, states, controls, timesteps = planners[i].get_path()
    ids_list.append(ids)


# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b','b', 'g', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
mprint = MultiRRTPrinter(env, planner_list, ids_list, tcol, pcol)
mprint.print_rrt('./media/swap_env_kcbs_paths.png', print_tree=False)
mprint.print_highres_simulation(dict_paths, "./media/swap_env_unicycle_kcbs.gif", 
                            animation_speed=50)


"""

seeds = np.random.randint(0, 10000, size=10)
for s in seeds:
    print("\nRNG Seed: ", s)
    kcbs_planner = KCBS(
                        env = env,
                        agents = agents,
                        low_level_planners = planners,
                        max_trials = 10000,
                        planning_time = 600.0,
                        rng_seed = s,
                        print_logs=False,
                        debug_flag=False
                        )
    t = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print("Planning Time: ", t)
    print("Paths Cost: ", cost)

"""