import sys
sys.path.append('./src')

import os
import time
import numpy as np

from Environments import *
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba
from kinodynamic_TI_eb_rrt import KinoTIEBRRT
from rrt import RRT
from prioritized_planning import PrioritizedPlanning
from kcbs import check_high_resolution_paths_collision_free
from mapf_env_square_agent_unicycle import get_unicycle_agent


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
    (7.0, 1.0, 0.0),
    (2.5, 2.5, 0.0),
    (11.0, 1.0, 0.0),
    (1.0, 14.0, 0.0),
    (12.0, 9.5, 0.0),
    (5.5, 10.0, 0.0),
    (9.5, 13.5, 0.0),
    (14.0, 6.0, 0.0),
    (5.0, 6.5, 0.0),
    (11.5, 6.5, 0.0),
    (2.8, 7.2, 0.0),
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
    (3.0, 9.0),
    (13.8, 11.0),
    (2.5, 6.0),
    (11.5, 13.5),
    (5.5, 4.0),
    (14.0, 5.0),
    (1.2, 13.8),
    (6.2, 6.2),
    (12.2, 7.0),
    (3.5, 10.8),
    (8.8, 4.8),
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


num_agents = 30
goal_radius = 0.5
planning_time = 300.0
sampling_time_step = 1.0
minimum_time_step = 0.1
num_extension_trials = 20
goal_sampling_probability = 0.01
dynamic_agent_clearance = 0.0
kd_tree_delta_radius = 0.1
max_num_edges_per_node = 1000
num_skip_edges = 10
kino_num_random_edges = 1
epsilon_random = 0.01
kino_ti_edge_bundle_file = "edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz"
rng_seed = np.random.randint(0, 1000)
print("RNG Seed: ", rng_seed)
print_logs = True
debug_flag = False
planner_kind = os.environ.get("PLANNER_KIND", "kino_ti_eb_rrt").lower()


def get_plain_rrt_planner(agent_id, agent):
    return RRT(
        start=starts[agent_id],
        goal=goals[agent_id],
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        use_fixed_sampling_time=False,
        sampling_time_step=sampling_time_step,
        minimum_time_step=minimum_time_step,
        max_iter=np.inf,
        planning_time=planning_time,
        num_extension_trials=num_extension_trials,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=rng_seed + 1000 * agent_id,
        goal_sampling_probability=goal_sampling_probability,
        dynamic_agent_clearance=dynamic_agent_clearance,
        print_logs=print_logs,
        debug_flag=debug_flag,
    )


def get_kino_ti_eb_rrt_planner(agent_id, agent):
    data = np.load(kino_ti_edge_bundle_file)
    edge_bundle = EdgeBundle(
        data, fix_num_edges=30000, use_all_edges=False,
        rng_seed=42 + agent.id)
    edge_ids = np.arange(edge_bundle.num_edges, dtype=np.int64)
    thetas = edge_bundle.start_states[:, 2]
    eb_kd_tree = CircularAngleIndexNumba(thetas, ids=edge_ids)

    return KinoTIEBRRT(
        start=starts[agent_id],
        goal=goals[agent_id],
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        edge_bundle=edge_bundle,
        use_fixed_sampling_time=False,
        sampling_time_step=sampling_time_step,
        minimum_time_step=minimum_time_step,
        max_iter=np.inf,
        planning_time=planning_time,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=max_num_edges_per_node,
        num_skip_edges=num_skip_edges,
        num_random_edges=kino_num_random_edges,
        epsilon_random=epsilon_random,
        eb_kd_tree=eb_kd_tree,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=kd_tree_delta_radius,
        udf_seed=rng_seed + 1000 * agent_id,
        goal_sampling_probability=goal_sampling_probability,
        dynamic_agent_clearance=dynamic_agent_clearance,
        debug_flag=debug_flag,
        print_logs=print_logs,
    )


def get_low_level_planner(agent_id, agent):
    if planner_kind == "rrt":
        return get_plain_rrt_planner(agent_id, agent)
    if planner_kind in ("kino_ti_eb_rrt", "kinoti", "kinoti_eb_rrt"):
        return get_kino_ti_eb_rrt_planner(agent_id, agent)
    raise ValueError(
        "Unknown PLANNER_KIND: "
        + planner_kind
        + ". Use rrt or kino_ti_eb_rrt."
    )


agents = []
planners = []
for agent_id in range(num_agents):
    agent = get_unicycle_agent(agent_id)
    agents.append(agent)
    planners.append(get_low_level_planner(agent_id, agent))

print("Prioritized Planning seed:", rng_seed)
print("Low-level planner:", planner_kind)
print("Agents:", num_agents)
print("Goal sampling probability:", goal_sampling_probability)
print("Dynamic agent clearance:", dynamic_agent_clearance)

t0 = time.time()
path_found, planner_time, path_cost = PrioritizedPlanning.plan_multi(
    planners=planners,
    planning_time=planning_time,
    print_logs=print_logs,
)
wall_time = time.time() - t0

print("Path found:", path_found)
print("Planner reported time:", planner_time)
print("Wall time:", wall_time)
print("Total path cost:", path_cost)

for i, planner in enumerate(planners):
    print(
        "Agent", i,
        "path_found:", planner.path_found,
        "path_time:", planner.path_time,
        "path_cost:", planner.path_cost,
        "tree_nodes:", len(planner.tree.nodes),
    )

if path_found:
    high_res_paths = [planner.get_high_resolution_path_numpy_array() for planner in planners]
    collision_result = check_high_resolution_paths_collision_free(
        high_res_paths,
        agents=agents,
        distance_metric_state_size=agents[0].distance_metric_state_size,
        dynamic_agent_clearance=dynamic_agent_clearance,
        roundoff_digits=planners[0].roundoff_digits,
    )
    print("Collision free:", collision_result["collision_free"])
    if not collision_result["collision_free"]:
        print("Collision times:", collision_result["collision_times"])
        print("First agent:", collision_result["first_agent"])
        print("Second agent:", collision_result["second_agent"])
