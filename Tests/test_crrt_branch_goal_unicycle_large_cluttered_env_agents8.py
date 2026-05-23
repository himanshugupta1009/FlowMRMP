import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapf_mplconfig")
sys.path.append("./src")

from Agents import UniCycle
from Environments import RectangleObstacle2D, SquareEnvironment
from cRRT import CRRT
from kcbs import check_high_resolution_paths_collision_free
from mapf_env_square_agent_unicycle import get_unicycle_agent


starts = [
    (8.475727025696276, 36.399396853155096, 2.351704883710214),
    (11.586823146518856, 36.0126973803567, 5.186804678458259),
    (21.60583263510259, 16.476363676107802, 0.38223657632653696),
    (21.38462394891068, 35.9108526038565, 5.585473581275507),
    (11.526827287177163, 28.961425844173046, 4.923798055189963),
    (17.154555241298887, 27.572396342307854, 4.059382377997044),
    (9.194008174486147, 32.51411098476325, 4.262651508019563),
    (3.0553400627138885, 10.412419495743238, 3.308785409838181),
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


def run_seed(seed, planning_time_limit=300.0, num_extension_trials=20):
    env = SquareEnvironment(40.0, 40.0, obstacles, obs_buffers=False)
    num_agents = len(starts)
    goal_radii = [0.5 for _ in range(num_agents)]
    agents = [get_unicycle_agent(agent_id)
              for agent_id in range(num_agents)]

    crrt = CRRT(
        agents=agents,
        starts=starts,
        goals=goals,
        goal_radii=goal_radii,
        env=env,
        use_fixed_sampling_time=False,
        sampling_time_step=1.0,
        minimum_time_step=0.1,
        max_iter=np.inf,
        planning_time=planning_time_limit,
        num_extension_trials=num_extension_trials,
        truncate_paths=False,
        branch_goal_parking=True,
        truncation_check_threshold=1.0,
        isvalid_function=[UniCycle.is_new_node_valid for _ in agents],
        cost_function=[UniCycle.get_cost for _ in agents],
        reached_goal_function=[UniCycle.agent_reached_goal for _ in agents],
        random_point_function=[UniCycle.get_random_point for _ in agents],
        udf_seed=seed,
        print_logs=False,
        debug_flag=False,
    )

    start_time = time.time()
    planning_time = crrt.plan_path()
    wall_time = time.time() - start_time

    result = {
        "seed": seed,
        "success": crrt.path_found,
        "planning_time": min(planning_time, planning_time_limit),
        "wall_time": wall_time,
        "total_path_cost": 0.0,
        "average_agent_path_time": 0.0,
        "max_agent_path_time": 0.0,
        "collision_free": False,
        "tree_nodes": len(crrt.tree.nodes),
        "goal_seen_count": int(np.count_nonzero(crrt.goal_seen_by_agent)),
    }

    if crrt.path_found and planning_time <= planning_time_limit:
        high_res_paths = crrt.get_high_resolution_path_numpy_array()
        collision_result = check_high_resolution_paths_collision_free(
            high_res_paths,
            agents=crrt.agents,
            distance_metric_state_size=crrt.agents[0].distance_metric_state_size,
            dynamic_agent_clearance=crrt.dynamic_agent_clearance,
            roundoff_digits=crrt.roundoff_digits,
        )
        agent_path_times = crrt.get_agent_path_times(high_res_paths)
        result.update({
            "success": collision_result["collision_free"],
            "total_path_cost": float(np.sum(crrt.path_cost)),
            "average_agent_path_time": float(np.average(agent_path_times)),
            "max_agent_path_time": float(np.max(agent_path_times)),
            "collision_free": collision_result["collision_free"],
        })

    return result


def print_averages(results):
    successes = [result for result in results if result["success"]]
    print("\nAverage results")
    print("Total runs:", len(results))
    print("Successes:", len(successes))
    print("Percent success:", len(successes) / len(results))
    print("Average computation time over all runs:",
          np.average([r["planning_time"] for r in results]))
    print("Average wall time over all runs:",
          np.average([r["wall_time"] for r in results]))
    print("Average tree nodes over all runs:",
          np.average([r["tree_nodes"] for r in results]))
    print("Average goal_seen_count over all runs:",
          np.average([r["goal_seen_count"] for r in results]))

    if len(successes) == 0:
        return

    print("Average total path cost over successful runs:",
          np.average([r["total_path_cost"] for r in successes]))
    print("Average agent path time over successful runs:",
          np.average([r["average_agent_path_time"] for r in successes]))
    print("Average max agent path time over successful runs:",
          np.average([r["max_agent_path_time"] for r in successes]))
    print("Average tree nodes over successful runs:",
          np.average([r["tree_nodes"] for r in successes]))


if __name__ == "__main__":
    planning_time_limit = float(os.environ.get("MRMP_PLANNING_TIME", "300.0"))
    num_extension_trials = int(os.environ.get("MRMP_EXTENSION_TRIALS", "20"))
    seeds = [808, 809, 810, 811, 812, 813, 814, 815, 816, 817]
    results = []
    for seed in seeds:
        print("RNG Seed:", seed)
        result = run_seed(seed, planning_time_limit, num_extension_trials)
        results.append(result)
        print("Path found:", result["success"])
        print("Planning time:", result["planning_time"])
        print("Wall time:", result["wall_time"])
        print("Total path cost:", result["total_path_cost"])
        print("Average agent path time:", result["average_agent_path_time"])
        print("Max agent path time:", result["max_agent_path_time"])
        print("Collision free:", result["collision_free"])
        print("Tree nodes:", result["tree_nodes"])
        print("Goal seen count:", result["goal_seen_count"])
        print()

    print_averages(results)
