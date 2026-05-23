"""
KCBS + PyBullet visualization for a multi-agent quadcopter scenario.
Creates an urban-style environment with vertical cuboid obstacles, plans
paths for 5 agents using KCBS, and animates the result.
"""
import os
import sys
import time
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from Environments import CuboidEnvironment, CuboidObstacle3D
from Agents import Quadcopter3D
from constrainedX import ConstrainedRRT
from kcbs import KCBS


def build_urban_environment():
    env_length = 14.0
    env_breadth = 14.0
    env_height = 3.0
    z_bounds = (0.2, 3.0)

    obstacles = []
    rng = np.random.default_rng(2026)
    building_height = 2.8
    building_z = building_height / 2.0
    min_edge_clearance = 1.1
    keepout_radius = 1.6

    keepout_points = [
        (1.0, 1.0),
        (13.0, 1.0),
        (1.0, 13.0),
        (13.0, 13.0),
        (7.0, 1.0),
        (13.0, 13.0),
        (1.0, 13.0),
        (13.0, 1.0),
        (1.0, 1.0),
        (7.0, 13.0),
    ]
    placed = []

    def sample_building_dims():
        if rng.random() < 0.45:
            long_side = rng.uniform(1.0, 1.6)
            short_side = rng.uniform(0.4, 0.7)
            if rng.random() < 0.5:
                return long_side, short_side
            return short_side, long_side
        return rng.uniform(0.6, 1.1), rng.uniform(0.6, 1.1)

    def is_valid_position(x, y, l, w):
        for kx, ky in keepout_points:
            if np.hypot(x - kx, y - ky) < keepout_radius:
                return False
        buffer = 0.35
        for px, py, pl, pw in placed:
            if abs(x - px) < (pl + l) / 2.0 + buffer and abs(y - py) < (pw + w) / 2.0 + buffer:
                return False
        return True

    target_count = 26
    attempts = 0
    max_attempts = target_count * 40
    while len(obstacles) < target_count and attempts < max_attempts:
        attempts += 1
        l, w = sample_building_dims()
        x = rng.uniform(min_edge_clearance, env_length - min_edge_clearance)
        y = rng.uniform(min_edge_clearance, env_breadth - min_edge_clearance)
        if not is_valid_position(x, y, l, w):
            continue
        obstacles.append(CuboidObstacle3D(x=x, y=y, z=building_z, l=l, w=w, h=building_height))
        placed.append((x, y, l, w))

    env = CuboidEnvironment(
        length=env_length,
        breadth=env_breadth,
        height=env_height,
        obs=obstacles,
        z_bounds=z_bounds,
    )
    return env


def build_agents(num_agents, z_bounds):
    agents = []
    for agent_id in range(num_agents):
        agents.append(
            Quadcopter3D(
                agent_id=agent_id,
                max_speed=3.0,
                max_acceleration=2.0,
                radius=0.25,
                z_bounds=z_bounds,
                rng_seed=agent_id + 77,
            )
        )
    return agents


def build_start_goal_pairs():
    starts = [
        np.array((1.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
        np.array((13.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
        np.array((1.0, 13.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
        np.array((13.0, 13.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
        np.array((7.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
        np.array((7.0, 3.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    ]
    goals = [
        np.array((13.0, 13.0, 1.6), dtype=np.float64),
        np.array((1.0, 13.0, 1.6), dtype=np.float64),
        np.array((13.0, 1.0, 1.6), dtype=np.float64),
        np.array((1.0, 1.0, 1.6), dtype=np.float64),
        np.array((7.0, 13.0, 1.6), dtype=np.float64),
        np.array((7.0, 7.0, 1.6), dtype=np.float64),
    ]
    return starts, goals


def build_rrt_planner(start, goal, goal_radius, agent, env, rng_seed=None):
    agent.env = env
    seed = rng_seed if rng_seed is not None else np.random.randint(0, 1000)
    return ConstrainedRRT(
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        sampling_time_step=0.6,
        minimum_time_step=0.1,
        max_iter=7000,
        planning_time=50.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point_3d,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=seed,
        prune_tree=True,
    )


def plan_kcbs_paths(env, agents, starts, goals, goal_radius, rng_seed=19):
    planners = []
    for i in range(len(agents)):
        planners.append(build_rrt_planner(starts[i], goals[i], goal_radius, agents[i], env, rng_seed=rng_seed))

    kcbs_planner = KCBS(
        env=env,
        agents=agents,
        low_level_planners=planners,
        max_trials=150,
        planning_time=240.0,
        rng_seed=rng_seed,
        print_logs=False,
        debug_flag=False,
        prune_tree=True,
    )

    path_found, paths, cost, elapsed = kcbs_planner.plan_multi_agent_paths()
    print(f"KCBS path found: {path_found}")
    print(f"KCBS total cost: {cost}")
    print(f"KCBS elapsed time: {elapsed:.2f}s")
    return path_found, paths, planners


def _load_urban_obstacles(p, env):
    for obs in getattr(env, "obstacles", []):
        if isinstance(obs, CuboidObstacle3D):
            half_extents = [obs.l / 2.0, obs.w / 2.0, obs.h / 2.0]
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[1.0, 0.0, 0.0, 0.85],
            )
            p.createMultiBody(
                baseVisualShapeIndex=visual,
                basePosition=[obs.x, obs.y, obs.z],
            )


def _draw_env_bounds(p, env):
    x0, y0 = env.env_start
    x1 = x0 + env.size[0]
    y1 = y0 + env.size[1]
    z0, z1 = env.z_bounds
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start_idx, end_idx in edges:
        p.addUserDebugLine(
            corners[start_idx],
            corners[end_idx],
            lineColorRGB=[0.0, 0.0, 0.0],
            lineWidth=2.0,
        )


def _draw_agent_paths(p, path_list, colors, stride=1):
    for idx, path in enumerate(path_list):
        color = colors[idx % len(colors)]
        rgb = color[:3]
        if len(path) < 2:
            continue
        for j in range(1, len(path), stride):
            start = path[j - 1][:3]
            end = path[j][:3]
            p.addUserDebugLine(start, end, lineColorRGB=rgb, lineWidth=1.5)


def visualize_kcbs_paths(env, paths, goals, agent_radius, time_step=0.05):
    if not paths:
        print("No paths available to visualize.")
        return

    path_list = list(paths)
    if any(len(path) == 0 for path in path_list):
        print("One or more paths are empty; skipping visualization.")
        return

    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise RuntimeError("PyBullet is required for visualization. Install pybullet to continue.") from exc

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[1.0, 1.0, 1.0, 0.0])
    floor_half_height = 0.02
    floor_half_extents = [env.size[0] / 2.0, env.size[1] / 2.0, floor_half_height]
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=floor_half_extents,
        rgbaColor=[0.96, 0.96, 0.96, 1.0],
    )
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half_extents)
    floor_center = [
        env.env_start[0] + env.size[0] / 2.0,
        env.env_start[1] + env.size[1] / 2.0,
        -floor_half_height,
    ]
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=floor_center,
    )
    p.resetDebugVisualizerCamera(
        cameraDistance=20.0,
        cameraYaw=45.0,
        cameraPitch=-35.0,
        cameraTargetPosition=[7.0, 7.0, 1.0],
    )
    _load_urban_obstacles(p, env)
    _draw_env_bounds(p, env)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "quadrotor.urdf")

    agent_colors = [
        [0.2, 0.6, 1.0, 1.0],
        [1.0, 0.6, 0.2, 1.0],
        [0.5, 1.0, 0.4, 1.0],
        [1.0, 0.4, 0.6, 1.0],
        [0.9, 0.9, 0.2, 1.0],
        [0.6, 0.4, 1.0, 1.0],
    ]

    agent_ids = []
    for i, path in enumerate(path_list):
        start_pos = path[0][:3]
        agent_id = p.loadURDF(urdf_path, basePosition=start_pos, useFixedBase=True)
        p.changeVisualShape(agent_id, -1, rgbaColor=agent_colors[i % len(agent_colors)])
        agent_ids.append(agent_id)

        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=agent_radius * 0.9,
            rgbaColor=[0.0, 1.0, 0.0, 0.35],
        )
        p.createMultiBody(baseVisualShapeIndex=goal_visual, basePosition=goals[i])

    _draw_agent_paths(p, path_list, agent_colors)

    max_len = max(len(path) for path in path_list)
    idx = 0
    while p.isConnected(client) and idx < max_len:
        for i, path in enumerate(path_list):
            step = idx if idx < len(path) else len(path) - 1
            pos = path[step][:3]
            p.resetBasePositionAndOrientation(agent_ids[i], pos, [0, 0, 0, 1])
        p.stepSimulation()

        keys = p.getKeyboardEvents()
        if (p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED) or (
            27 in keys and keys[27] & p.KEY_WAS_TRIGGERED
        ):
            p.disconnect(client)
            return

        time.sleep(time_step)
        idx += 1

    idx = max_len - 1
    while p.isConnected(client):
        keys = p.getKeyboardEvents()
        if (p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED) or (
            27 in keys and keys[27] & p.KEY_WAS_TRIGGERED
        ):
            p.disconnect(client)
            return
        step_dir = 0
        if p.B3G_RIGHT_ARROW in keys:
            if keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                step_dir = 1
            elif keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
                step_dir = 1
        if p.B3G_LEFT_ARROW in keys:
            if keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                step_dir = -1
            elif keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
                step_dir = -1
        if step_dir:
            idx = min(max(idx + step_dir, 0), max_len - 1)
        for i, path in enumerate(path_list):
            step = idx if idx < len(path) else len(path) - 1
            pos = path[step][:3]
            p.resetBasePositionAndOrientation(agent_ids[i], pos, [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(0.02)


def main():
    env = build_urban_environment()
    starts, goals = build_start_goal_pairs()
    z_bounds = env.z_bounds
    agents = build_agents(len(starts), z_bounds)
    goal_radius = 0.4

    path_found, paths, _ = plan_kcbs_paths(env, agents, starts, goals, goal_radius)
    if path_found:
        visualize_kcbs_paths(env, paths, goals, agents[0].radius, time_step=0.05)
    else:
        print("No collision-free KCBS solution found for this scenario.")


if __name__ == "__main__":
    main()
