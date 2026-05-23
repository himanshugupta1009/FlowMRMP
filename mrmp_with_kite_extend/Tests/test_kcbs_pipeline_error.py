import sys
sys.path.append('./src') 
from Environments import *
from Agents import UniCycle
from edge_bundle import EdgeBundle
import numpy as np
from printer import MultiRRTPrinter

seed = 1001
print(f"Using seed {seed}")
starts = [(23.82822757137833, 3.533815905891272, 1.1792883917752939),
 (6.351112539713126, 8.174949753350191, 1.1325127822680088),
 (29.015209180151945, 11.617909101178611, 3.0826625567401735),
 (34.04626002267773, 5.134067851913, 1.2640877857966655),
 (29.652861003742586, 6.689821253298707, 5.587260288260063),
 (5.412350497026313, 15.602102357851676, 6.143615900060464),
 (15.178618426562412, 7.227503617894657, 0.546369289132863),
 (20.38772581210646, 30.311473842602894, 2.4729648173845655),
 (25.11683578294977, 25.993035041325694, 2.521727952189311),
 (20.88900147056047, 34.46643804285898, 4.141258225088885)]

goals = [(11.032707333362723, 23.903029981192002),
 (8.957851640539754, 34.52350938796867),
 (11.332482045836802, 15.814563547833536),
 (12.981711039407909, 33.68993622048185),
 (16.401149854636824, 14.871239988690709),
 (16.666096193133313, 6.360144197045411),
 (26.541996256868966, 11.132561243924666),
 (8.131341482163071, 9.720148954712432),
 (15.796139390144766, 21.63025648703264),
 (29.583363072494464, 14.369700273087622)]

goal_radii = [2. for _ in goals]
num_agents = len(starts)

obstacles = [ 
    CircularObstacle2D(32.34932759101222,26.358510837923617,4.31278195287735),
    CircularObstacle2D(24.012744469423275,19.29330821905245,3.621200196603753),
    CircularObstacle2D(22.129276800741632,10.115900439045047,1.1495006187859385),
    CircularObstacle2D(3.902646598821973,5.086853592348842,0.7931039753861231),
    CircularObstacle2D(27.739381861285565,35.36716573896719,1.1353748307894884),
    CircularObstacle2D(10.496618631702718,29.11106794566068,1.5962699633927933),
    CircularObstacle2D(35.45609834480462,35.503458289984515,1.7049178565313934),
    CircularObstacle2D(19.964138035917983,25.939784804985553,0.6557955005635376),
    CircularObstacle2D(7.529253455981176,23.682895463626476,0.5522897127565072),
    CircularObstacle2D(34.21427934578041,13.821352334763557,1.3581034121748599),
    CircularObstacle2D(7.566835552300009,19.620008549760712,0.7900972849205461),
    CircularObstacle2D(11.910790389958468,9.321849722978174,0.9382464244815695),
    CircularObstacle2D(4.453380745139248,31.172223158487444,1.4322515896691241),
    CircularObstacle2D(13.505558667842866,3.7703053543217755,0.8662267510018618),
    CircularObstacle2D(27.092806096773078,3.2699068372279085,0.506028193971614),
    CircularObstacle2D(16.725397970145195,33.77238252967681,0.8889376821586197)
]

agents = [UniCycle(agent_id=i,
                 max_speed = 2.0,
                 max_omega= np.pi/2,
                 radius = 1.0,
                 rng_seed=i*seed) for i in range(10)]

env = SquareEnvironment(40.0, 40.0, obstacles)

edge_bundle_file_location = './edge_bundles/eb_unicycle_edges_100000.npz'
data = np.load(edge_bundle_file_location, allow_pickle=True)
edge_bundle = EdgeBundle(data, fix_num_edges=100) # NOTE THIS!!!!!!!!

"""
EB KCBS
"""
from constrainedX import ConstrainedEdgeBundleType2RRT
from kcbs import KCBS
agent_objs = []
planners = []
seed_counter = 0
for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
    agent_obj = agent
    agent_objs.append(agent_obj)
    planners.append(ConstrainedEdgeBundleType2RRT( 
                                    start=start, goal=goal,
                                    goal_radius=goal_radius, 
                                    env = env, agent=agent_obj,
                                    edge_bundle=edge_bundle,
                                    sampling_time_step=1.5,
                                    minimum_time_step=0.1,
                                    max_iter = 10000,
                                    num_random_edges= 10,
                                    num_skip_edges= 10,
                                    planning_time=300.0,        
                                    isvalid_function=UniCycle.is_new_node_valid,
                                    cost_function=UniCycle.get_cost,
                                    random_point_function=UniCycle.get_random_point, 
                                    reached_goal_function = UniCycle.agent_reached_goal,
                                    translate_function = UniCycle.point_translate_function,
                                    sort_edges_function=UniCycle.sort_edges,
                                    print_logs=False,
                                    debug_flag=False,
                                    udf_seed = (seed_counter*seed) * seed, #NOTE THIS!!!
                                    use_fixed_sampling_time=False
                                ))
    seed_counter+=1

kcbs_planner = KCBS(
            env = env,
            agents = agent_objs,
            low_level_planners = planners,
            max_trials = np.inf,
            clearance_threshold=0.0,
            planning_time = 300.0,
            print_logs=False,
            debug_flag=False,
            rng_seed=seed,
            )
path_found, paths, cost, time = kcbs_planner.plan_multi_agent_paths()
print("Time taken for planning: ", time)
# c = kcbs_planner.plan_multi_agent_paths()

"""

find_first_collision_numba(paths, 
    kcbs_planner.agents_state_length,kcbs_planner.agents_radius,
    kcbs_planner.distance_metric_state_size,
    kcbs_planner.clearance_threshold,kcbs_planner.roundoff_digits)

"""



"""
#Visualize the paths

# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
                'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
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

planner_list = [planners[i] for i in range(num_agents)]
mprint = MultiRRTPrinter(env, planner_list, [], [], [])
# The speed param is the time between each step in the gif (lower number->faster animation speed)
mprint.print_highres_simulation(dict_paths, "./media/kcbs_error_env_unicycle_10_agents.gif", 
                            animation_speed=50)


"""