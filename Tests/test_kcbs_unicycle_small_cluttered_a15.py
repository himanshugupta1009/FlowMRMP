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

seed = int(os.environ.get("DEBUG_SEED", "1502"))
num_agents = 18
goal_radius = 0.5
planning_time = float(os.environ.get("DEBUG_TIME", "300.0"))
kd_tree_delta_radius = 0.1
print_logs = os.environ.get("DEBUG_PRINT_LOGS", "1") != "0"

env = SquareEnvironment(15.0, 15.0, obstacles)

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

planners = []
planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_unicycle
# planner_function = get_constrained_db_rrt_planner_unicycle
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,
                                     agents[i],env))
                                    #  agents[i],env, use_optimizer=False))


print("Starting debug KCBS run")
print("Seed:", seed)
print("Num agents:", num_agents)
print("Print logs:", print_logs)

seed_rng = np.random.default_rng(seed)
kcbs_seeds = seed_rng.integers(0, 1000000, size=10)
# kcbs_seeds = [801, 610, 42, 999, 123, 456, 789, 321, 654, 987]
results = []

for run_idx, s in enumerate(kcbs_seeds, start=1):
    s = int(s)
    print("\n=====================================")
    print("KCBS run:", run_idx, "of", len(kcbs_seeds))
    print("RNG Seed:", s)

    planners = []
    for i in range(num_agents):
        planners.append(planner_function(starts[i], goals[i], goal_radius,
                                         agents[i], env))
                                        #  agents[i],env, use_optimizer=False))

    kcbs_planner = KCBS(
                        env=env,
                        agents=agents,
                        low_level_planners=planners,
                        max_trials=10000,
                        planning_time=planning_time,
                        rng_seed=s,
                        print_logs=print_logs,
                        debug_flag=False
                        )
    # kcbs_planner.low_level_planners[0].rng_seed = s
    # kcbs_planner.low_level_planners[0].rng = np.random.default_rng(s)
    t0 = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    elapsed = time.time() - t0

    results.append((s, path_found, cost, delta_t, elapsed,
                    kcbs_planner.node_list.count))

    print("Path found:", path_found)
    print("Cost:", cost)
    print("KCBS reported time:", delta_t)
    print("Wall time:", elapsed)
    print("Conflict node count:", kcbs_planner.node_list.count)
    print("Collision count matrix:")
    print(kcbs_planner.collision_count)

print("\n=====================================")
print("Summary:")
for s, path_found, cost, delta_t, elapsed, node_count in results:
    print("seed=", s,
          "path_found=", path_found,
          "cost=", cost,
          "kcbs_time=", delta_t,
          "wall_time=", elapsed,
          "conflict_nodes=", node_count)