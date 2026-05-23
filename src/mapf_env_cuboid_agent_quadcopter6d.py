import numpy as np

from Agents import QuadCopter6D
from kd_tree_quadcopter6d import VxyzTree
from kd_tree_grid_quadcopter6d import VxyzGridTree
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from motion_primitives import (
    load_quadcopter6d_motion_primitives,
    transform_quadcopter6d_trajectory_numba,
)
from db.constrained_db_optimize_quadcopter6d import (
    optimize_dbrrt_quadcopter6d_path as optimize_constrained_dbrrt_quadcopter6d_path,
    Quadcopter6DTrajOptOptions as ConstrainedQuadcopter6DTrajOptOptions,
)


def get_quadcopter_agent(agent_id):
    return QuadCopter6D(
        agent_id=agent_id,
        max_speed=0.5,
        max_acceleration=2.0,
        radius=0.25,
        rng_seed=agent_id + 77,
    )


def get_rrt_planner(start, goal, goal_radius, agent,
                                 env, filler_input=''):
    return ConstrainedRRT(
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        use_fixed_sampling_time=False,
        sampling_time_step=1.0,
        minimum_time_step=0.1,
        max_iter=10000,
        planning_time=10.0,
        num_extension_trials=10,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=np.random.randint(0, 1000),
        prune_tree=True,
    )


def get_kino_TI_eb(agent, 
            edge_bundle_file_location = 'edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'):
    data = np.load(edge_bundle_file_location)
    kino_TI_eb = EdgeBundle(data, fix_num_edges=100000, 
                use_all_edges=False,rng_seed=42 + agent.id)
    edge_ids = np.arange(kino_TI_eb.num_edges, dtype=np.int64)
    vx = kino_TI_eb.start_states[:, 3]  # vx
    vy = kino_TI_eb.start_states[:, 4]  # vy
    vz = kino_TI_eb.start_states[:, 5]  # vz
    v_scale = 1.0
    kd_tree_TI_eb = VxyzTree(vx, vy, vz, ids=edge_ids, 
                    scales=(v_scale,v_scale,v_scale))
    return kino_TI_eb, kd_tree_TI_eb


def get_kino_TI_eb_rrt_planner(start, goal, goal_radius, 
    agent, env,
    edge_bundle_file_location = 'edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'):

    data = np.load(edge_bundle_file_location)
    kino_TI_eb = EdgeBundle(data, fix_num_edges=100000, 
                use_all_edges=False,rng_seed=42 + agent.id)
    edge_ids = np.arange(kino_TI_eb.num_edges, dtype=np.int64)
    vx = kino_TI_eb.start_states[:, 3]  # vx
    vy = kino_TI_eb.start_states[:, 4]  # vy
    vz = kino_TI_eb.start_states[:, 5]  # vz
    v_scale = 1.0
    kd_tree_TI_eb = VxyzTree(vx, vy, vz, ids=edge_ids, 
                    scales=(v_scale,v_scale,v_scale))
    kino_eb_rrt = ConstrainedKinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb,
            use_fixed_sampling_time=False,
            sampling_time_step=1.0,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            max_num_edges_per_node=1000,
            num_skip_edges= 10,
            num_random_edges= 1,
            eb_kd_tree = kd_tree_TI_eb,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = 0, #Will be overwritten by KCBS init
            debug_flag=False,
            print_logs=True,
            )
    return kino_eb_rrt
    
    
def get_kino_TI_eb_rrt_planner_grid_quadcopter6d(start, goal, goal_radius, 
    agent, env,
    edge_bundle_file_location = 'edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'):

    data = np.load(edge_bundle_file_location)
    kino_TI_eb = EdgeBundle(data, fix_num_edges=100000, 
                use_all_edges=False,rng_seed=42 + agent.id)
    edge_ids = np.arange(kino_TI_eb.num_edges, dtype=np.int64)
    vx = kino_TI_eb.start_states[:, 3]  # vx
    vy = kino_TI_eb.start_states[:, 4]  # vy
    vz = kino_TI_eb.start_states[:, 5]  # vz
    v_scale = 1.0
    delta_radius = 0.1
    kd_tree_TI_eb = VxyzGridTree(vx, vy, vz, ids=edge_ids, 
                    scales=(v_scale,v_scale,v_scale),
                    vmin=-agent.max_speed,
                    vmax= agent.max_speed,
                    cell_size=delta_radius/2,
                    initial_out_capacity=2048,
                    return_ids=False
                    )
    kino_eb_rrt = ConstrainedKinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb,
            use_fixed_sampling_time=False,
            sampling_time_step=1.0,
            minimum_time_step=0.1,
            max_iter = np.inf,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            max_num_edges_per_node=1000,
            num_skip_edges= 10,
            num_random_edges= 1,
            eb_kd_tree = kd_tree_TI_eb,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=delta_radius,
            udf_seed = 0, #Will be overwritten by KCBS init
            debug_flag=False,
            print_logs=True,
            )
    return kino_eb_rrt


def get_constrained_db_rrt_planner_quadcopter6d(
    start, goal, goal_radius, agent, env, use_optimizer=True, 
    num_edges=1000,
    motion_primitive_dt=0.1,
    primitive_file_location="motion_primitives/quadcopter6d_long_50_1000_primitives.npz"
    ):

    motion_primitives, kd_tree = load_quadcopter6d_motion_primitives(
        num_edges=num_edges,
        dt=motion_primitive_dt,
        primitive_file_location=primitive_file_location,
    )

    planner = ConstrainedDbRRTPlanner(
        start=np.asarray(start, dtype=np.float64),
        goal=np.asarray(goal, dtype=np.float64),
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        motion_primitives=motion_primitives,
        alpha=0.5,
        delta=0.3,
        minimum_time_step=0.1,
        max_iter=10000,
        planning_time=600.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        transform_trajectory_function=transform_quadcopter6d_trajectory_numba,
        motion_primitive_kd_tree=kd_tree,
        get_motion_primitive_kd_tree_query=agent.get_eb_kd_tree_query,
        max_candidate_motions_per_expand=1000,
        allow_intermediate_goal=True,
        cost_delta_factor=0.0,
        goal_bias=0.1,
        goal_expand_mode="focused",
        random_expand_mode="randomized",
        dynamic_agent_clearance=0.0,
        udf_seed=0, # Will be overwritten by KCBS init
        debug_flag=False,
        print_logs=False,
    )
    if use_optimizer:
        planner.set_optimizer(
            lambda curr_planner: optimize_constrained_dbrrt_quadcopter6d_path(
                curr_planner,
                options=ConstrainedQuadcopter6DTrajOptOptions(),
            )
        )
    return planner
