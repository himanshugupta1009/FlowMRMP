import numpy as np
import os
from dataclasses import dataclass
import msgpack

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from edge_bundle import EdgeBundle
from constrainedX import *
from kcbs import *
from kd_tree_unicycle import CircularAngleIndexNumba
from db.constrained_db_optimize_unicycle import (
    optimize_dbrrt_unicycle_path as optimize_constrained_dbrrt_unicycle_path,
    UnicycleTrajOptOptions as ConstrainedUnicycleTrajOptOptions,
)
from motion_primitives import transform_unicycle_trajectory_numba

# Old agent parameters that we used to test with
# def get_unicycle_agent(agent_id):
#     agent = UniCycle(
#                 agent_id = agent_id,
#                 max_speed = 2.0,
#                 max_omega= np.pi/2,
#                 radius = 1.0,
#                 rng_seed = 77 + agent_id
#                 )
#     return agent
@dataclass
class DynoplanPrimitiveBundle:
    start_states: np.ndarray
    final_states: np.ndarray
    trajectories: np.ndarray
    trajectory_lengths: np.ndarray
    actions: np.ndarray
    action_lengths: np.ndarray
    representative_actions: np.ndarray
    timesteps: np.ndarray
    num_edges: int
    dt: float


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def load_unicycle_dbrrt_primitives(
    num_edges=15000,
    dt=0.1,
    primitive_file_location='motion_primitives/unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack',
):
    if not os.path.exists(primitive_file_location):
        raise FileNotFoundError(f"Missing primitive file: {primitive_file_location}")

    with open(primitive_file_location, "rb") as f:
        packed = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

    primitives = packed["data"] if isinstance(packed, dict) and "data" in packed else packed
    primitives = primitives[:num_edges]
    if len(primitives) == 0:
        raise ValueError("primitive file loaded, but contains zero primitives")

    lengths = np.array([len(p["states"]) for p in primitives], dtype=np.int64)
    action_lengths = np.array([len(p["actions"]) for p in primitives], dtype=np.int64)
    max_len = int(lengths.max())
    max_action_len = int(action_lengths.max())
    n = len(primitives)

    start_states = np.empty((n, 3), dtype=np.float64)
    final_states = np.empty((n, 3), dtype=np.float64)
    trajectories = np.full((n, max_len, 3), np.nan, dtype=np.float64)
    timesteps = np.empty(n, dtype=np.float64)
    actions = np.full((n, max_action_len, 2), np.nan, dtype=np.float64)
    representative_actions = np.empty((n, 2), dtype=np.float64)

    for i, primitive in enumerate(primitives):
        states = _as_float_array(primitive["states"])
        acts = _as_float_array(primitive["actions"])
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError(f"primitive {i} has states with shape {states.shape}, expected (*, 3)")
        if acts.ndim != 2 or acts.shape[1] != 2:
            raise ValueError(f"primitive {i} has actions with shape {acts.shape}, expected (*, 2)")

        L = states.shape[0]
        start_states[i] = states[0]
        final_states[i] = states[-1]
        trajectories[i, :L] = states
        timesteps[i] = acts.shape[0] * dt
        if acts.shape[0] > 0:
            actions[i, :acts.shape[0]] = acts
            representative_actions[i] = acts[0]
        else:
            representative_actions[i] = np.zeros(2, dtype=np.float64)

    motion_primitives = DynoplanPrimitiveBundle(
        start_states=start_states,
        final_states=final_states,
        trajectories=trajectories,
        trajectory_lengths=lengths,
        actions=actions,
        action_lengths=action_lengths,
        representative_actions=representative_actions,
        timesteps=timesteps,
        num_edges=n,
        dt=dt,
    )

    edge_ids = np.arange(motion_primitives.num_edges, dtype=np.int64)
    thetas = motion_primitives.start_states[:, 2]
    kd_tree = CircularAngleIndexNumba(thetas, ids=edge_ids)
    return motion_primitives, kd_tree

#New agent parameters to match dbCBS paper
def get_unicycle_agent(agent_id):
    agent = UniCycle(
                agent_id = agent_id, 
                max_speed = 0.5,
                max_omega = 0.5,
                radius = 0.3,
                rng_seed= 77 + agent_id
                )
    return agent


def get_rrt_planner(start,goal,gr,agent,env,filler_input=''):

    rrt_planner  = ConstrainedRRT( 
            start=start, goal=goal,
            goal_radius=gr, 
            env = env, agent=agent,
            use_fixed_sampling_time=False,
            sampling_time_step=1.0,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=10.0,
            num_extension_trials=10,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function=agent.agent_reached_goal,
            udf_seed=np.random.randint(0, 1000)
            # udf_seed = 7
           )
    return rrt_planner


def get_eb_rrt_planner(start,goal,gr,agent,env,
                    edge_bundle_file_location = 'edge_bundles/eb_unicycle_edges_100000.npz'):
    data = np.load(edge_bundle_file_location)
    eb_unicycle = EdgeBundle(data, fix_num_edges=1000)
    eb_rrt  = ConstrainedEdgeBundleType2RRT( 
                        start=start, goal=goal,
                        goal_radius=gr,
                        env = env, agent=agent, 
                        edge_bundle = eb_unicycle,
                        use_fixed_sampling_time=False,
                        sampling_time_step=1.0,
                        minimum_time_step=0.1,
                        max_iter = 10000,
                        planning_time = 10.0,
                        num_random_edges = 10,
                        num_skip_edges = 100,
                        isvalid_function = agent.is_new_node_valid,
                        cost_function = agent.get_cost,
                        random_point_function = agent.get_random_point,
                        reached_goal_function = agent.agent_reached_goal,
                        translate_function = agent.point_translate_function,
                        sort_edges_function= agent.sort_edges,
                        udf_seed = np.random.randint(0, 1000),
                        # udf_seed = 7
                        )
    return eb_rrt


def get_kino_TI_eb_rrt_planner_unicycle(start, goal, goal_radius, agent, env,
    edge_bundle_file_location = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'):

    data = np.load(edge_bundle_file_location)
    kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, 
                use_all_edges=False,rng_seed=42 + agent.id)
                # use_all_edges=False,rng_seed=(42 + agent.id)*67)
    edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
    thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
    kd_tree_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)

    kino_eb_rrt = ConstrainedKinoTIEBRRT( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_unicycle,
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
            num_skip_edges = 10,
            num_random_edges = 1,
            epsilon_random = 0.01,
            eb_kd_tree = kd_tree_TI_eb_unicycle,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            udf_seed = 0, #Will be overwritten by KCBS init
            debug_flag=False,
            print_logs=False,
            )
    return kino_eb_rrt


def get_constrained_db_rrt_planner_unicycle(
    start,goal,goal_radius,agent,env,
    primitive_file_location='motion_primitives/unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack',
    use_optimizer=True, num_edges=30000, motion_primitive_dt=0.1):

    motion_primitives, kd_tree = load_unicycle_dbrrt_primitives(
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
        transform_trajectory_function=transform_unicycle_trajectory_numba,
        motion_primitive_kd_tree=kd_tree,
        get_motion_primitive_kd_tree_query=agent.get_eb_kd_tree_query,
        max_candidate_motions_per_expand=1000,
        allow_intermediate_goal=True,
        cost_delta_factor=0.0,
        goal_bias=0.1,
        goal_expand_mode="focused",
        random_expand_mode="randomized",
        dynamic_agent_clearance=0.0,
        udf_seed=0,  # will be overwritten by KCBS init
        debug_flag=False,
        print_logs=False,
    )
    if use_optimizer:
        planner.set_optimizer(
            lambda curr_planner: optimize_constrained_dbrrt_unicycle_path(
                curr_planner,
                options=ConstrainedUnicycleTrajOptOptions(allow_raw_fallback=False),
            )
        )
    return planner
    


"""
import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *


num_agents = 5

start1 = np.array((7.0, 5.0, 0))
goal1 = np.array((24.0, 37.0))

start2 = np.array((2.0, 26.0, 0))
goal2 = np.array((37.0, 30.0))

start3 = np.array((28.0, 5.0, 0))
goal3 = np.array((5.0, 29.0))

start4 = np.array((32.0, 18.0, 0))
goal4 = np.array((2.0, 10.0))

start5 = np.array((16.0, 37.0, 0))
goal5 = np.array((36.0, 10.0))

starts = [start1, start2, start3, start4, start5]
goals = [goal1, goal2, goal3, goal4, goal5]

agents = {}
for agent_id in range(num_agents):
    agents[agent_id] = get_unicycle_agent(agent_id)
planners = {}
planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
for i in range(num_agents):
    planners[agents[i].id] = planner_function(starts[i],goals[i],2.0,agents[i],env)


kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 600.0,
                    print_logs=True
                    )
t = time.time()  
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Time taken for planning: ", t)



#Visualize the paths

from printer import MultiRRTPrinter

# define a list of colors for trees and paths to use in drawing function
tcol =['y', 'c', 'b', 'g', 'b', 'b']
pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
                'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
planner_list = [planners[i] for i in range(num_agents)]
mprint = MultiRRTPrinter(env, planner_list, [], [], [])

# The speed param is the time between each step in the gif (lower number->faster animation speed)
mprint.print_highres_simulation(paths, "./media/five_unicycle_kcbs_eb_rrt_sim.gif", 
                            animation_speed=50)



"""


"""

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *


for i in range(10):

    planners = {}
    for k in range(num_agents):
        planners[agents[k].id] = planner_function(starts[k],goals[k],2.0,agents[k],env)

    s = np.random.randint(0, 1000)
    kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 600.0,
                    rng_seed = s
                    )
    print("Planning for Experiment ", i+1, " with seed: ", s)
    st = time.time()
    path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
    td = time.time() - st
    print("Time for Experiment ", i+1, " is : ",td)

    
"""




"""
Empty

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *

obstacles = []                     
env = SquareEnvironment(40, 40, obstacles)

loc1 = (20, 30, 0)
loc2 = (30, 20, 0)
loc3 = (25, 10, 0)
loc4 = (15, 10, 0)
loc5 = (10, 20, 0)

#3Agents
starts = [loc1, loc2, loc3]
goals = [loc3, loc4, loc5]
goal_radii = [2.0, 2.0, 2.0]


#4Agents
starts = [loc1, loc2, loc3, loc4]
goals = [loc3, loc4, loc5, loc1]
goal_radii = [2.0, 2.0, 2.0, 2.0]


#5Agents
starts = [loc1, loc2, loc3, loc4, loc5]
goals = [loc3, loc4, loc5, loc1, loc2]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]


"""

"""
Obstacles

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *

obstacles = [CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(18, 16, 3),
            CircularObstacle2D(20, 5, 3)
            ] 
                    
env = SquareEnvironment(40, 40, obstacles)

goal = (33.0, 10.0)
start = (14.0, 2.0, 0)

start2 = (5.0, 2.0, 0)
goal2 = (25.0, 25.0)

start3 = (30.0, 5.0, 0)
goal3 = (5.0, 30.0)

start4 = (30.0, 20.0, 0)
goal4 = (2.0, 10.0)

start5 = (27.0, 15.0, 0)
goal5 = (10.0, 3.0)


#3Agents
starts = [start, start2, start3]
goals = [goal, goal2, goal3]
goal_radii = [2.0, 2.0, 2.0]


#4Agents
starts = [start, start2, start3, start4]
goals = [goal, goal2, goal3, goal4]
goal_radii = [2.0, 2.0, 2.0, 2.0]


#5Agents
starts = [start, start2, start3, start4, start5]
goals = [goal, goal2, goal3, goal4, goal5]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]


"""


"""
#Cluttered Environment

import sys
sys.path.append('./src')
from mapf_env_square_agent_unicycle import *
from printer import *

obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 2),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 4),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ] 
            
env = SquareEnvironment(40, 40, obstacles)

goal = (24.0, 37.0)
start = (7.0, 5.0, 0)

start2 = (2.0, 26.0, 0)
goal2 = (37.0, 30.0)

start3 = (28.0, 5.0, 0)
goal3 = (5.0, 29.0)

start4 = (32.0, 18.0, 0)
goal4 = (2.0, 10.0)

start5 = (16.0, 37.0, 0)
goal5 = (36.0, 10.0)


#3Agents
starts = [start, start2, start3]
goals = [goal, goal2, goal3]
goal_radii = [2.0, 2.0, 2.0]



#4Agents
starts = [start, start2, start3, start4]
goals = [goal, goal2, goal3, goal4]
goal_radii = [2.0, 2.0, 2.0, 2.0]



#5Agents
starts = [start, start2, start3, start4, start5]
goals = [goal, goal2, goal3, goal4, goal5]
goal_radii = [2.0, 2.0, 2.0, 2.0, 2.0]



"""



"""

agents = {}
for agent_id in range(len(starts)):
    agents[agent_id] = get_unicycle_agent(agent_id)


cost_arr = []
time_arr = []
num_experiments = 20

for k in range(num_experiments):

    planners = {}
    for i in range(len(starts)):
        planners[agents[i].id] = planner_function(starts[i],goals[i],goal_radii[i],agents[i])

    kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 100.0
                    )

    st = time.time()
    path_found, paths, path_cost = kcbs_planner.plan_multi_agent_paths()
    # path_cost = sum(path_cost_dict.values())
    td = time.time() - st

    print("Cost for iteration ", k+1, " is : ",path_cost)
    cost_arr.append(path_cost)
    print("Time for iteration ", k+1, " is : ",td)
    time_arr.append(td)
                    
mc = np.mean(cost_arr)
sdc = np.std(cost_arr)
print( "Mean Cost: ", mc, " SDE: ", sdc/np.sqrt(num_experiments) )


mt = np.mean(time_arr)
sdt = np.std(time_arr)
print( "Mean Cost: ", mt, " SDE: ", sdt/np.sqrt(num_experiments) )


"""
