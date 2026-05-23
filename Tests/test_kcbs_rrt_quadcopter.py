import sys
sys.path.append('./src')
from Environments import CuboidEnvironment, SphericalObstacle3D
from Agents import QuadCopter6D
from constrainedX import ConstrainedRRT
from kcbs import *
from printer import *
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent, \
    get_rrt_planner, get_kino_TI_eb_rrt_planner, get_kino_TI_eb_rrt_planner_grid_quadcopter6d
from visualizations.quadcopter_visualization import visualize_quadcopter_path


obstacles = [
    SphericalObstacle3D(x=2.5, y=2.5, z=0.8, r=0.4),
    SphericalObstacle3D(x=5.5, y=3.5, z=1.6, r=0.6),
    SphericalObstacle3D(x=3.5, y=6.0, z=1.0, r=0.5),
]
env = CuboidEnvironment(length=8.0, breadth=8.0, height=3.0, obs=obstacles)

starts = [
    np.array((0.8, 0.8, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((6.5, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((1.0, 6.0, 0.8, 0.0, 0.0, 0.0), dtype=np.float64),
]
goals = [
    np.array((7.0, 7.0, 1.8), dtype=np.float64),
    np.array((1.5, 6.5, 1.2), dtype=np.float64),
    np.array((6.2, 1.5, 1.5), dtype=np.float64),
]
goal_radius = 0.35

num_agents = len(starts)
agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_quadcopter_agent(agent_id))

seeds = [93, 228, 828, 760, 472, 701, 881, 140, 365, 160]
# seeds = np.random.randint(0, 10000, size=100)
costs = []
time_array = []

print("Starting KCBS Planning Experiments")
print("#######################################################")

for seed_index in range(len(seeds)):
    s = seeds[seed_index]
    planners = []
    planner_function = get_rrt_planner
    # planner_function = get_kino_TI_eb_rrt_planner
    # planner_function = get_kino_TI_eb_rrt_planner_grid_quadcopter6d
    for i in range(num_agents):
        planners.append(planner_function(starts[i], goals[i], goal_radius,
                                         agents[i], env))

    kcbs_planner = KCBS(
        env=env,
        agents=agents,
        low_level_planners=planners,
        max_trials=1000,
        planning_time=300.0,
        rng_seed=s,
        print_logs=False,
        debug_flag=False,
    )
    t = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    t = time.time() - t
    print(f"Iteration: {seed_index}")
    print(f"RNG Seed: {s}")
    print("Time taken for planning: {:.3f}s".format(t))
    time_array.append(t)
    print(f"Cost: {cost}")
    costs.append(cost)
    print("#######################################################")

# Toggle visualization for this test file
VISUALIZE_QUAD = False
if path_found and VISUALIZE_QUAD:
    # visualize the first agent's path
    if planners:
        path_dict = planners[0].get_high_resolution_path()
        visualize_quadcopter_path(path_dict, env, agent_radius=agents[0].radius)

"""

In [7]: time_array
Out[7]: 
[0.023640871047973633,
 0.10140037536621094,
 0.02068018913269043,
 0.03774094581604004,
 0.1528306007385254,
 0.025589942932128906,
 0.2661707401275635,
 0.08477139472961426,
 0.15779685974121094,
 0.5114643573760986]

In [8]: costs
Out[8]: 
[35.63233888093578,
 31.374771684618278,
 35.31407011626095,
 35.911161523181654,
 36.839895070911155,
 38.46821462432147,
 33.82640120930107,
 36.77672102399859,
 34.21869110384854,
 37.610587989274684]
 
"""