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

# seed: 1049

starts = [
	(11.777928026849953, 21.59844486823614, 4.88647878507452),
	(20.88221806694752, 18.5715364546508, 5.783653477198367),
	(17.26787681391891, 24.577219097361247, 1.634977677442681),
	(30.351033533894803, 16.537015427676415, 4.4588686087289835),
	(4.562214631169679, 28.35277618607915, 5.33199369149357),
	(18.01110024280572, 7.534413109091251, 4.766415496383577),
	(27.468922231056446, 26.57728684704878, 5.108978544587677),
	(6.608877736613589, 10.015873303463604, 0.8007753424826467),
	(35.83657083137417, 14.528914971558782, 0.626384426708539),
	(27.095496298989172, 22.913389810443295, 3.5204270508218616),
]

goals = [
	(26.888782417316712, 17.7429599279794),
	(29.873658660226127, 13.849995177191571),
	(23.048259578106435, 7.184855222188487),
	(10.176878071346723, 25.16868907134616),
	(17.09627683864005, 22.514479251911347),
	(20.2184709196353, 33.95194160838833),
	(12.675392792153557, 33.24418608632648),
	(24.603200318900925, 27.670466695857932),
	(25.13025406716617, 11.38210327043382),
	(9.380839186810054, 19.52126983951552),
]

goal_radii = [
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
	2.0,
]

obstacles = [
	RectangleObstacle2D(11.8757112695261, 14.7753869011411, 2.9222297732556215, 3.773504415643333),
	RectangleObstacle2D(26.454113809683605, 34.006665905957064, 2.061124225305981, 6.292944782355627),
	RectangleObstacle2D(30.909268092072253, 4.238809607434206, 3.3991570907837527, 2.274400260287219),
	RectangleObstacle2D(35.63329077135981, 32.553480390964495, 3.215440308252057, 3.716070416877762),
	RectangleObstacle2D(11.386398975871206, 6.4401604230387735, 1.0914669614143433, 5.296017252359851),
	RectangleObstacle2D(9.326812710136899, 36.07996378662771, 2.1516863565546345, 1.4279793502613005),
	RectangleObstacle2D(36.352041672106566, 25.97874105018752, 1.730096554018254, 1.1463871077243368),
]

env = SquareEnvironment(40, 40, obstacles)
agents = [
	UniCycle(max_speed=0.5, agent_id=0, rng_seed=1049,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=1, rng_seed=1050,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=2, rng_seed=1051,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=3, rng_seed=1052,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=4, rng_seed=1053,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=5, rng_seed=1054,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=6, rng_seed=1055,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=7, rng_seed=1056,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=8, rng_seed=1057,max_omega=2.0, radius=0.4),
	UniCycle(max_speed=0.5, agent_id=9, rng_seed=1058,max_omega=2.0, radius=0.4),
]


num_agents = len(agents)
planners = []
eb_file_location = 'edge_bundles/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
# planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radii[i],
					agents[i],env,eb_file_location))

s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
# s = 42  
# s = 984
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



"""
#Print environment
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
env_fn = './media/env_test_pipeline_error_agents' + str(num_agents) + '_' + str(planner_function.__name__) + '.png'
MultiRRTPrinter.print_rrt_env(env_fn, env, agents, starts, goals, goal_radii, pcol)

"""

"""
#Print the paths

tcol = ['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
        'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']



#Method 1 : Reconstruct planners' trees from KCBS path node and then access paths
path_node = kcbs_planner.path_cbs_node
dict_paths = {}
for i in range(num_agents):
    t = path_node.agent_trees[i]
    planners[i].tree = t[0]
    planners[i]._node_matrix = t[1]
    goal_id = next(reversed(t[0]._node))
    planners[i].goal_node_id = goal_id
    planners[i].path_found = True
    dict_paths[i] = planners[i].get_high_resolution_path()

#Method 2 : Directly access paths from KCBS path node - Preferred
path_node = kcbs_planner.path_cbs_node
dict_paths = {}
for i in range(num_agents):
	dict_paths[i] = path_node.agent_paths[i]
    
planner_list = [planners[i] for i in range(num_agents)]
mprint = MultiRRTPrinter(env, planner_list, [], [], [])

mprint.print_rrt('./media/kcbs_paths_kino_TI_eb_rrt_random_env.png', print_tree=True)

# The speed param is the time between each step in the gif (lower number->faster animation speed)
mprint.print_highres_simulation(dict_paths, "./media/kcbs_kino_TI_eb_rrt_random_env.gif", 
						animation_speed=50)

"""


"""

#Unicycle: Generate an edge bundle and save it 

import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from edge_bundle import GenerateEdgeBundle
obstacles = []
env = SquareEnvironment(40, 40, obstacles)

def get_start_state_kinodynamic_TI_unicycle(env, agent, rng):
    theta = rng.uniform(0.0, 2*np.pi)
    start_state = np.array([0.0, 0.0, theta])
    return start_state

from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=100000,
                            get_start_state=get_start_state_kinodynamic_TI_unicycle,
                            minimum_time_step=0.1,
                            max_sample_T=1.5,
                            rng_seed=77,
                            )
kinodynamic_TI_unicycle_edge_bundle = gen_eb.generate_edge_bundle()
eb_filename = 'edge_bundles/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
np.savez_compressed(eb_filename, **kinodynamic_TI_unicycle_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)


"""