import sys
sys.path.append('./src')
from Environments import *
from Agents import UniCycle
from utils import euclidean_distance
from edge_bundle import EdgeBundle
from constrainedX import *
from cRRT import *
from kcbs import check_high_resolution_paths_collision_free
from mapf_env_square_agent_unicycle import (
        get_unicycle_agent,
        get_rrt_planner, get_eb_rrt_planner, 
        get_kino_TI_eb_rrt_planner_unicycle,
        get_constrained_db_rrt_planner_unicycle)
from printer import *

starts = [
    (8.475727025696276, 36.399396853155096, 2.351704883710214),
    (11.586823146518856, 36.0126973803567, 5.186804678458259),
    (21.60583263510259, 16.476363676107802, 0.38223657632653696),
    (21.38462394891068, 35.9108526038565, 5.585473581275507),
    (11.526827287177163, 28.961425844173046, 4.923798055189963),
    (17.154555241298887, 27.572396342307854, 4.059382377997044),
    (9.194008174486147, 32.51411098476325, 4.262651508019563),
    (3.0553400627138885, 10.412419495743238, 3.308785409838181),
    (8.419711803902793, 23.011167560297043, 1.8299192267904372),
    (35.61389396530191, 29.538268355135777, 0.46715533976667406),
]

goals = [
    (34.864329915511654, 9.003899733940381),
    (27.300942040287637, 20.57283260237884),
    (21.415433893642362, 32.11505993112027),
    (18.965682757246398, 22.875128976549142),
    (15.782169681359699, 10.592860806584735),
    (23.895727973205105, 35.318459073195),
    (29.95421575545637, 31.261370762087413),
    (21.695409776980473, 21.46828957867312),
    (20.28401116779998, 34.36720330871995),
    (10.471055047841116, 33.422171568528704),
]


obstacles = [
            RectangleObstacle2D(7.2318180528742, 19.11557739153202, 3.6001340616706443, 1.2342962240137418),
            RectangleObstacle2D(24.640769120616746, 11.249958595699702, 1.8879733859231442, 2.7977578114173465),
            RectangleObstacle2D(33.52165105929789, 14.712475719357597, 4.987964054602642, 4.762326271741823),
            RectangleObstacle2D(25.087543392907982, 4.588990511682701, 4.740777979767133, 2.78139498421943),
            RectangleObstacle2D(33.89683742111731, 25.475322697342563, 1.933936687930509, 2.2835108061898968),
            RectangleObstacle2D(34.37105405176375, 35.2127489370062, 4.163599185649419, 2.673047891734387),
            RectangleObstacle2D(16.765313221135678, 33.66062958329129, 2.9450461561358456, 6.237881829980317),
            RectangleObstacle2D(4.578790030760515, 32.45567757895103, 2.1340532155846708, 2.612702802456554),
            RectangleObstacle2D(25.12892430122978, 22.266743199270557, 1.4311726211916573, 7.232857246763847),
            RectangleObstacle2D(13.385729240385057, 24.03349439022954, 1.3892207989803393, 3.6348281532725073),
            RectangleObstacle2D(6.971415847887632, 14.013040661417069, 3.8846399568092878, 2.352161305699923),
            RectangleObstacle2D(17.938939841099074, 6.780920639306122, 1.4716421879683201, 3.6851890360628623),
            RectangleObstacle2D(10.257196133603149, 5.4469093419979275, 4.115726446827529, 1.8541809184556888),
            RectangleObstacle2D(31.15259510533822, 4.60663958022095, 1.0702469277264908, 1.6214788027072329),
        ]
obstacles = []


env = SquareEnvironment(40.0, 40.0, obstacles, obs_buffers=False)
num_agents = len(starts)
# num_agents = 3
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

# pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
#         'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
#         'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
#         'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
# MultiRRTPrinter.print_rrt_env('./media/env_db_cbs_' + str(num_agents) + '_' + str(weird_behavior_env_name) + '.png',
#                                         env, agents, starts, goals, goal_radii, pcol)


# curate list of helper functions for call to CRRT 
# each of these functions must match the agent at the same index
isvalid_funcs = [UniCycle.is_new_node_valid for _ in range(len(agents))]
cost_funcs = [UniCycle.get_cost for _ in range(len(agents))]
random_pt_funcs = [UniCycle.get_random_point for _ in range(len(agents))]
reached_goal_funcs = [UniCycle.agent_reached_goal for _ in range(len(agents))]


s = np.random.randint(0, 1000)
s = 823
print("RNG Seed: ", s)
crrt = CRRT(agents=agents, 
            starts=starts[:num_agents],
            goals=goals[:num_agents],
            goal_radii=goal_radii,
            env=env,
            max_iter = math.inf,
            planning_time=300.0,
            truncate_paths=False,
            truncation_check_threshold=3.0,
            num_extension_trials=20,         
            isvalid_function=isvalid_funcs, 
            cost_function=cost_funcs,
            random_point_function=random_pt_funcs, 
            reached_goal_function = reached_goal_funcs,
            udf_seed = s, 
            print_logs=True
            )
# plan path 
planning_time = crrt.plan_path()
print(crrt.path_cost)

high_res_paths = crrt.get_high_resolution_path_numpy_array()
result = check_high_resolution_paths_collision_free(
    high_res_paths,
    agents=crrt.agents,
    distance_metric_state_size=crrt.agents[0].distance_metric_state_size,
    dynamic_agent_clearance=crrt.dynamic_agent_clearance,
    roundoff_digits=crrt.roundoff_digits,
)

print("Collision free:", result["collision_free"])

# path_rrt_nodes, states, controls, timesteps, costs = crrt.get_path()

# # define a list of colors for trees and paths to use in drawing function
# tcol =['y', 'c', 'b', 'g', 'b', 'b']
# pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
#         'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']

# mprint = MultiRRTPrinter(env, crrt, path_rrt_nodes, tcol, pcol, joint_states=True)
# # setting print_tree to true will show the tree as well
# mprint.print_rrt('./media/db_cbs_8agents_crrt_unicycle.png', print_tree=False)

# mprint.print_rrt_animation('./media/db_cbs_8agents_crrt_unicycle.gif', 
#                            animation_speed=150, print_tree=False)
