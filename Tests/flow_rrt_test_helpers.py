from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
for path in (ROOT_DIR, MRMP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Environments import RectangleObstacle2D, SquareEnvironment  # noqa: E402
from Agents import SecondOrderCar  # noqa: E402
from src.flow_eb_rrt import get_flow_eb_rrt_planner_soc  # noqa: E402


GOAL_RADIUS = 0.5
SEED = 0
DEFAULT_CHECKPOINT = "checkpoints/soc_edge_flow/soc_edge_flow_k32_v1/best.pt"


def get_second_order_car_agent(agent_id=1):
    return SecondOrderCar(
        agent_id=agent_id,
        max_speed=1.0,
        max_acceleration=2.0,
        max_phi=np.pi / 3,
        max_steering_rate=0.5,
        radius=0.3,
        wheelbase=0.7,
        rng_seed=42,
    )


def make_test_environment_small():
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
    return SquareEnvironment(15.0, 15.0, obstacles, obs_buffers=False)


def make_test_environment_large():
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
    return SquareEnvironment(40.0, 40.0, obstacles, obs_buffers=False)


make_test_environment = make_test_environment_small
make_test_environment = make_test_environment_large


def apply_common_planner_args(planner, args):
    planner.max_iter = args.max_iter
    planner.planning_time = args.planning_time
    planner.print_logs = args.print_logs
    planner.debug_flag = args.debug
    planner.rng = np.random.default_rng(args.seed)
    return planner


def make_flow_eb_rrt_planner_soc(
    start,
    goal,
    goal_radius,
    agent,
    env,
    *,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device="cuda:1",
    sample_steps=8,
    flow_prefetch_batch_size=1,
):
    return get_flow_eb_rrt_planner_soc(
        start,
        goal,
        goal_radius,
        agent,
        env,
        checkpoint_path=checkpoint_path,
        device=device,
        sample_steps=sample_steps,
        flow_prefetch_batch_size=flow_prefetch_batch_size,
    )


def get_rrt_iterations(planner):
    if hasattr(planner, "profile") and "extend_calls" in planner.profile:
        return planner.profile["extend_calls"]
    return None


def print_planner_result(planner):
    rrt_iterations = get_rrt_iterations(planner)
    print("Path found:", planner.path_found)
    if rrt_iterations is not None:
        print("RRT iterations:", rrt_iterations)
    print("Nodes:", planner.num_rrt_nodes())
    print("Path time:", planner.path_time)
    print("Path cost:", planner.path_cost)

    if planner.path_found:
        ids, states, controls, timesteps = planner.get_path()
        print("Path node ids:", ids)
        print("Path states shape:", states.shape)
        print("Controls shape:", controls.shape)
        print("Timesteps:", timesteps)
