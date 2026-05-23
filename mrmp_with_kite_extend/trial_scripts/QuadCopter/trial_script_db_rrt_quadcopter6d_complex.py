"""
Single-agent Db-RRT trial for QuadCopter6D on the complex KCBS environment.

This isolates raw Db-RRT search from KCBS so each start/goal pair can be
checked independently.

Run from repo root:
    python Tests/trial_script_db_rrt_quadcopter6d_complex.py
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Environments import CuboidEnvironment, CuboidObstacle3D
from mapf_env_cuboid_agent_quadcopter6d import (
    get_quadcopter_agent,
    get_constrained_db_rrt_planner_quadcopter6d,
    get_kino_TI_eb_rrt_planner_grid_quadcopter6d
)


def _signed_distance_to_sphere(x, y, z, cx, cy, cz, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return float(np.sqrt(dx * dx + dy * dy + dz * dz) - r)


def _signed_distance_to_box(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    hz = 0.5 * (zmax - zmin)

    dx = abs(x - cx) - hx
    dy = abs(y - cy) - hy
    dz = abs(z - cz) - hz

    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    outside_dz = max(dz, 0.0)
    outside_dist = float(np.sqrt(outside_dx * outside_dx + outside_dy * outside_dy + outside_dz * outside_dz))
    inside_dist = min(max(dx, max(dy, dz)), 0.0)
    return outside_dist + inside_dist


def _signed_distance_to_obstacles(env, x, y, z):
    best = float("inf")
    spheres_src = getattr(env, "static_circular_obstacles", None)
    boxes_src = getattr(env, "static_rectangular_obstacles", None)
    if spheres_src is not None:
        spheres = np.asarray(spheres_src, dtype=np.float64).reshape(-1, 4)
        for sphere in spheres:
            best = min(best, _signed_distance_to_sphere(x, y, z, sphere[0], sphere[1], sphere[2], sphere[3]))
    if boxes_src is not None:
        boxes = np.asarray(boxes_src, dtype=np.float64).reshape(-1, 6)
        for box in boxes:
            best = min(best, _signed_distance_to_box(x, y, z, box[0], box[1], box[2], box[3], box[4], box[5]))
    return best


def verify_returned_path(path_view, *, env, goal, goal_radius, agent, assert_rollout=True):
    states, controls, timesteps = path_view.get_high_resolution_path_and_actions()
    reported_states = path_view.get_high_resolution_path_numpy_array()

    print("\n===== Returned path verification =====")
    print("states shape:", states.shape)
    print("controls shape:", controls.shape)
    print("timesteps shape:", timesteps.shape)

    assert states.shape == reported_states.shape, (
        f"High-res state shape mismatch: {states.shape} vs {reported_states.shape}"
    )
    assert np.allclose(states, reported_states), "Returned states disagree across accessors"
    assert states.shape[0] == controls.shape[0] + 1, (
        f"State/control length mismatch: {states.shape[0]} vs {controls.shape[0]}"
    )
    assert controls.shape[0] == timesteps.shape[0], (
        f"Control/timestep length mismatch: {controls.shape[0]} vs {timesteps.shape[0]}"
    )

    if assert_rollout:
        rollout = np.empty_like(states)
        rollout[0] = np.asarray(path_view.start, dtype=np.float64)
        curr = rollout[0].copy()
        for i, (u, dt) in enumerate(zip(controls, timesteps)):
            next_state, _ = path_view.agent.get_next_state(curr, u, float(dt), num_steps=1)
            rollout[i + 1] = next_state
            curr = next_state

        max_abs_err = float(np.max(np.abs(rollout - states)))
        print("rollout shape:", rollout.shape)
        print("rollout max_abs_err:", max_abs_err)
        print("rolled final state:", rollout[-1])
        print("reported final state:", states[-1])
        assert np.allclose(rollout, states, atol=1e-4, rtol=1e-4), (
            f"Returned path is not rollout-consistent; max_abs_err={max_abs_err}"
        )
        print(f"[OK] returned path is rollout-consistent, max_abs_err={max_abs_err:.3e}")

    obstacle_buffer = float(getattr(env, "obstacle_buffer", 0.0))
    boundary_buffer = float(getattr(env, "boundary_buffer", 0.0))
    required_clearance = float(agent.radius + obstacle_buffer)
    lower = np.array([agent.radius + boundary_buffer] * 3, dtype=np.float64)
    upper = np.array(
        [
            env.size[0] - agent.radius - boundary_buffer,
            env.size[1] - agent.radius - boundary_buffer,
            env.size[2] - agent.radius - boundary_buffer,
        ],
        dtype=np.float64,
    )

    for i, x in enumerate(states):
        signed_dist = _signed_distance_to_obstacles(env, x[0], x[1], x[2])
        assert signed_dist >= required_clearance - 1e-6, (
            f"State {i} violates obstacle clearance: signed_distance={signed_dist}, "
            f"required>={required_clearance}"
        )
        assert np.all(x[:3] >= lower - 1e-6) and np.all(x[:3] <= upper + 1e-6), (
            f"State {i} violates boundary limits: state={x[:3]}, lower={lower}, upper={upper}"
        )
        assert np.all(np.abs(x[3:6]) <= agent.max_speed + 1e-6), (
            f"State {i} violates speed limit: state={x}, max_speed={agent.max_speed}"
        )

    goal_dist = np.linalg.norm(states[-1][:3] - np.asarray(goal[:3], dtype=np.float64))
    assert goal_dist <= goal_radius + 1e-6, (
        f"Final state outside goal radius: dist={goal_dist}, goal_radius={goal_radius}"
    )
    print(f"[OK] returned path is feasible, goal_dist={goal_dist:.3f}")


obstacles = [
    CuboidObstacle3D(x=5.288024544853885, y=10.428115301068528, z=4.381429795846468, l=1.3839478994290928, w=0.5401805203430455, h=8.762859591692935),
    CuboidObstacle3D(x=11.954231894339882, y=8.603275488906887, z=4.1909151213488585, l=0.4894908302066708, w=1.3916708816110792, h=8.381830242697717),
    CuboidObstacle3D(x=5.09798694910239, y=4.379210742901877, z=3.5329163210542545, l=1.012947629187785, w=0.8241902727000516, h=7.065832642108509),
    CuboidObstacle3D(x=1.2515176919283175, y=6.382882438156907, z=3.7064760014154894, l=0.8154560303422158, w=0.9315903102362977, h=7.412952002830979),
    CuboidObstacle3D(x=3.571110230587391, y=11.42056387486772, z=4.2468278944017905, l=1.3569195226036317, w=0.5305939467851933, h=8.493655788803581),
    CuboidObstacle3D(x=7.747853019130037, y=6.206600327515435, z=4.375561998755278, l=0.7725502893336461, w=1.0734098954898244, h=8.751123997510556),
    CuboidObstacle3D(x=9.369922496110094, y=3.789125351106821, z=3.866387903731101, l=1.41759692208027, w=0.494146098178711, h=7.732775807462202),
    CuboidObstacle3D(x=7.572100162570637, y=8.433738463586065, z=3.7151803959960374, l=0.6944531823772292, w=0.9656203291792224, h=7.430360791992075),
    CuboidObstacle3D(x=1.3788626583269576, y=4.612971585766165, z=3.602016866375745, l=0.6801371563340349, w=1.0735310416558315, h=7.20403373275149),
    CuboidObstacle3D(x=1.2517230753901782, y=8.230925811928927, z=3.8643541654261577, l=0.8436815330362557, w=0.6464442743787236, h=7.7287083308523155),
    CuboidObstacle3D(x=11.937082922873042, y=3.321434501007907, z=4.427651086838168, l=0.8820119319327673, w=1.0452295541498402, h=8.855302173676336),
    CuboidObstacle3D(x=2.6144639370162315, y=4.4039025815115345, z=3.2908979727461443, l=0.5014458528208032, w=1.3934776335531336, h=6.5817959454922885),
    CuboidObstacle3D(x=8.882920490259526, y=11.497996916324377, z=3.6938262682356733, l=0.4695094664176922, w=1.3328617718899038, h=7.387652536471347),
    CuboidObstacle3D(x=10.771924060959083, y=6.681457358183268, z=3.947871746185684, l=0.4353548933573186, w=1.1882000449382848, h=7.895743492371368),
    CuboidObstacle3D(x=5.156669442822354, y=9.227777527413963, z=4.005500757253257, l=0.9586972526631968, w=1.021113964861763, h=8.011001514506514),
    CuboidObstacle3D(x=3.654977319031023, y=2.4409283243404722, z=3.7095992809212595, l=0.5059145301349421, w=1.061011454249298, h=7.419198561842519),
    CuboidObstacle3D(x=4.893883080637327, y=3.00533470339922, z=3.936839447293449, l=0.6398146687878034, w=0.9314806688844245, h=7.873678894586898),
    CuboidObstacle3D(x=7.606232768113648, y=4.199662449682638, z=3.7703559644576634, l=1.0397160299358554, w=0.7027863287983364, h=7.540711928915327),
    CuboidObstacle3D(x=10.26680979754552, y=12.264959075745383, z=4.492071066768307, l=0.7828675277900001, w=0.8176173537033076, h=8.984142133536613),
    CuboidObstacle3D(x=12.89335031969998, y=9.800204018389675, z=3.789154317261065, l=0.6230935142717156, w=1.068879463041406, h=7.57830863452213),
    CuboidObstacle3D(x=5.005793370411759, y=5.9529705241657656, z=4.1949360404462235, l=0.605115951078335, w=0.654224661506357, h=8.389872080892447),
    CuboidObstacle3D(x=5.06122558661513, y=1.20868576435954, z=3.9166955327763926, l=0.6123033693701111, w=1.3060735517962003, h=7.833391065552785),
    CuboidObstacle3D(x=11.029541389163448, y=10.855220717874019, z=3.7291117038396293, l=1.3154669800477419, w=0.42925454489060627, h=7.458223407679259),
    CuboidObstacle3D(x=4.4800948958876, y=7.186138305183269, z=4.072058450009972, l=1.3962236028963804, w=0.48787476278139835, h=8.144116900019943),
    CuboidObstacle3D(x=11.623496817952, y=5.3426442034016315, z=3.545363354615578, l=1.0812007218429494, w=0.6723078524366748, h=7.090726709231156),
    CuboidObstacle3D(x=1.4026451861099662, y=10.081135439533615, z=3.953521963710439, l=0.9446566529311644, w=0.7971433356450859, h=7.907043927420878),
]

env = CuboidEnvironment(length=14.0, breadth=14.0, height=10.0, obs=obstacles)
goal_radius = 0.3

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

print("Running complex-environment single-agent Db-RRT quadcopter trials")
print("Environment size:", env.size)
print("Num obstacles:", len(obstacles))
seed = 91
for agent_index in range(len(starts)):
    print("\n=======================================================")
    print(f"Agent index: {agent_index}")
    print(f"Seed: {seed}")
    print(f"Start: {starts[agent_index]}")
    print(f"Goal:  {goals[agent_index]}")

    agent = get_quadcopter_agent(agent_index)
    planner = get_constrained_db_rrt_planner_quadcopter6d(
        starts[agent_index],
        goals[agent_index],
        goal_radius,
        agent,
        env,
        use_optimizer=True,
    )
    # planner = get_kino_TI_eb_rrt_planner_grid_quadcopter6d(
    #     starts[agent_index],
    #     goals[agent_index],
    #     goal_radius,
    #     agent,
    #     env,)
    
    planner.udf_seed = seed
    planner.print_logs = True

    t0 = time.time()
    planner.plan_path()
    elapsed = time.time() - t0

    print("path_found:", planner.path_found)
    print("num_nodes:", len(planner.tree.nodes))
    print("path_cost:", planner.path_cost)
    print("path_time:", planner.path_time)
    print("wall_time:", elapsed)

    if planner.path_found:
        returned_path = (
            planner.optimized_path_view
            if getattr(planner, "optimized_path_view", None) is not None
            else planner
        )
        verify_returned_path(
            returned_path,
            env=planner.env,
            goal=planner.goal,
            goal_radius=planner.goal_radius,
            agent=planner.agent,
            assert_rollout=bool(getattr(planner, "optimized_path_view", None) is not None),
        )
        highres_states = planner.get_high_resolution_path_numpy_array()
        print("highres shape:", highres_states.shape)
        print("final state:", highres_states[-1])
        reached, dist = planner.reached_goal(
            highres_states[-1],
            planner.goal,
            planner.goal_radius,
            planner.agent,
        )
        relaxed_dist = np.linalg.norm(highres_states[-1][:3] - planner.goal[:3])
        print("reached_goal:", reached)
        print("goal_dist:", dist)
        print("relaxed_xyz_dist:", relaxed_dist)
