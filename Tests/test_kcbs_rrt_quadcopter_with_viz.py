"""
KCBS + PyBullet visualization for a multi-agent quadcopter scenario.
Creates an urban-style environment with vertical cuboid obstacles, plans
paths for 5 agents using KCBS, and animates the result.
"""

import sys
sys.path.append('./src')
import os
from pathlib import Path


from Environments import CuboidEnvironment, CuboidObstacle3D
from Agents import QuadCopter6D
from constrainedX import ConstrainedRRT
from kcbs import *
from printer import *
from mapf_env_cuboid_agent_quadcopter6d import (
    get_quadcopter_agent,
    get_rrt_planner, get_kino_TI_eb_rrt_planner, 
    get_kino_TI_eb_rrt_planner_grid_quadcopter6d,
    get_constrained_db_rrt_planner_quadcopter6d)
from visualizations.quadcopter_visualization import visualize_quadcopter_path



def build_urban_environment():
    env_length = 14.0
    env_breadth = 14.0
    env_height = 10.0

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
        building_height = rng.uniform(6.5, 9.0)
        building_z = building_height / 2.0
        obstacles.append(CuboidObstacle3D(x=x, y=y, z=building_z, l=l, w=w, h=building_height))
        placed.append((x, y, l, w))

    # obstacles = []
    env = CuboidEnvironment(
        length=env_length,
        breadth=env_breadth,
        height=env_height,
        obs=obstacles,
    )
    return env


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
    x0, y0, z0 = env.env_start
    x1 = x0 + env.size[0]
    y1 = y0 + env.size[1]
    z1 = z0 + env.size[2]
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


def find_project_root(marker="visualizations"):
    p = Path.cwd().resolve()
    for parent in [p] + list(p.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")


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

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # urdf_path = os.path.join(current_dir, "quadrotor.urdf")
    PROJECT_ROOT = find_project_root(marker="visualizations")
    urdf_path = PROJECT_ROOT / "visualizations" / "quadrotor.urdf"
    urdf_path = str(urdf_path)

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



env = build_urban_environment()

starts = [
    np.array((1.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((13.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((1.0, 13.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((13.0, 13.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((7.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
    np.array((7.0, 3.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64),
]
goals = [
    np.array((13.0, 13.0, 8.6), dtype=np.float64),
    np.array((1.0, 13.0, 3.6), dtype=np.float64),
    np.array((13.0, 1.0, 7.6), dtype=np.float64),
    np.array((1.0, 1.0, 1.6), dtype=np.float64),
    np.array((7.0, 13.0, 5.6), dtype=np.float64),
    np.array((7.0, 7.0, 8.6), dtype=np.float64),
]
goal_radius = 0.3
num_agents = len(starts)

agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_quadcopter_agent(agent_id))


s = np.random.randint(0, 10000)
planners = []
planner_function = get_rrt_planner
# planner_function = get_kino_TI_eb_rrt_planner
planner_function = get_kino_TI_eb_rrt_planner_grid_quadcopter6d
# planner_function = get_constrained_db_rrt_planner_quadcopter6d
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
    print_logs=True,
    debug_flag=False,
)
t = time.time()
path_found, paths, cost, delta_t = kcbs_planner.plan_multi_agent_paths()
t = time.time() - t
print("Time taken for planning: {:.3f}s".format(t))
print(f"KCBS path found: {path_found}")
print(f"KCBS total cost: {cost}")
print(f"KCBS elapsed time: {delta_t:.2f}s")
print("#######################################################")


# if path_found:
#     visualize_kcbs_paths(env, paths, goals, agents[0].radius, time_step=0.05)
# else:
#     print("No collision-free KCBS solution found for this scenario.")


