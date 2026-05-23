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


env = SquareEnvironment(40.0, 40.0, obstacles)

num_agents = 4
goal_radius = 0.5

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

planners = []
planner_function = get_rrt_planner
planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,
                                     agents[i],env))
                                    #  agents[i],env, use_optimizer=False))

s = np.random.randint(0, 1000)
print("RNG Seed: ", s)
# s = 42  
s = 99
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

seeds = np.random.randint(0, 10000, size=10)
seeds = [5431, 5930, 5126,  983, 2899, 6983, 6879, 1363, 9582, 7075]
for s in seeds:
    print("\nRNG Seed: ", s)

    planners = []
    planner_function = get_rrt_planner
    # planner_function = get_eb_rrt_planner
    # planner_function = get_kino_TI_eb_rrt_planner_unicycle
    for i in range(num_agents):
        planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))
    
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