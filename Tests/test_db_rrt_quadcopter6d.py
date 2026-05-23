"""
Single-agent Db-RRT smoke test scaffold for QuadCopter6D.

Run from the repo root:
    python Tests/test_db_rrt_quadcopter6d.py
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Environments import CuboidEnvironment, SphericalObstacle3D, CuboidObstacle3D
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent
from db.db_rrt import DbRRTPlanner
from db.db_optimize_quadcopter6d import (
    optimize_dbrrt_quadcopter6d_path,
    Quadcopter6DTrajOptOptions,
)
from motion_primitives import (
    load_quadcopter6d_motion_primitives,
    transform_quadcopter6d_trajectory_numba,
)
from quadcopter_viz_3d import (
    save_quadcopter_rrt_tree_3d,
    save_quadcopter_path_3d,
    save_quadcopter_path_overlay_3d,
)


def make_agent(agent_id=0):
    return get_quadcopter_agent(agent_id)


def make_planner(*, debug=False, delta=0.3, max_motions=30000, udf_seed=77):

    # obstacles = [
    #     SphericalObstacle3D(x=2.5, y=2.5, z=0.8, r=0.4),
    #     SphericalObstacle3D(x=5.5, y=3.5, z=1.6, r=0.6),
    #     SphericalObstacle3D(x=3.5, y=6.0, z=1.0, r=0.5),
    # ]
    # env = CuboidEnvironment(length=8.0, breadth=8.0, height=3.0, obs=obstacles)
    # start = np.array((0.8, 0.8, 0.6, 0.0, 0.0, 0.0), dtype=np.float64)
    # goal = np.array((7.0, 7.0, 1.8), dtype=np.float64)


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
    env = CuboidEnvironment(length=18.0, breadth=18.0, height=10.0, obs=obstacles)
    start = np.array((0.8, 0.8, 0.6, 0.0, 0.0, 0.0), dtype=np.float64)
    goal = np.array((17.0, 17.0, 7.8), dtype=np.float64)

    agent = make_agent(0)
    motion_primitives, kd_tree = load_quadcopter6d_motion_primitives(
        num_edges=max_motions, repo_root=REPO_ROOT
    )


    return DbRRTPlanner(
        start=start,
        goal=goal,
        goal_radius=0.35,
        env=env,
        agent=agent,
        motion_primitives=motion_primitives,
        alpha=0.5,
        delta=delta,
        minimum_time_step=0.1,
        max_iter=15000,
        planning_time=60.0,
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
        udf_seed=udf_seed,
        debug_flag=debug,
        print_logs=True,
    )


def verify_rollout_consistency(path_view, *, atol=1e-4, rtol=1e-4):
    states, controls, timesteps = path_view.get_high_resolution_path_and_actions()
    reported_highres = path_view.get_high_resolution_path_numpy_array()

    assert states.shape == reported_highres.shape, (
        f"High-res state shape mismatch: actions-path states={states.shape}, "
        f"highres={reported_highres.shape}"
    )
    assert np.allclose(states, reported_highres, atol=atol, rtol=rtol), (
        "Reported states from high-resolution accessors differ"
    )
    assert states.shape[0] == controls.shape[0] + 1, (
        f"High-res warm start mismatch: states={states.shape[0]}, controls={controls.shape[0]}"
    )
    assert controls.shape[0] == timesteps.shape[0], (
        f"Control/timestep length mismatch: controls={controls.shape[0]}, timesteps={timesteps.shape[0]}"
    )

    rollout = np.empty_like(states)
    rollout[0] = np.asarray(path_view.start, dtype=np.float64)
    curr = rollout[0].copy()

    for i, (u, dt) in enumerate(zip(controls, timesteps)):
        next_state, _ = path_view.agent.get_next_state(curr, u, float(dt), num_steps=1)
        rollout[i + 1] = next_state
        curr = next_state

    max_abs_err = float(np.max(np.abs(rollout - states)))
    print("\n===== Rollout consistency =====")
    print("rollout shape:", rollout.shape)
    print("max_abs_err:", max_abs_err)
    print("rolled final state:", rollout[-1])
    print("reported final state:", states[-1])

    assert np.allclose(rollout, states, atol=atol, rtol=rtol), (
        f"Returned path is not rollout-consistent; max_abs_err={max_abs_err}"
    )
    print(f"[OK] returned states are rollout-consistent, max_abs_err={max_abs_err:.3e}")


def inspect_dbrrt_returned_path(planner, *, atol=1e-4, rtol=1e-4):
    ids, coarse_states, action_sequences, primitive_timesteps = planner.get_path_to_node_id(planner.goal_node_id)
    highres_states, highres_controls, highres_timesteps = planner.get_high_resolution_path_and_actions()
    reported_highres = planner.get_high_resolution_path_numpy_array()

    print("\n===== Raw Db-RRT path inspection =====")
    print("coarse node count:", ids.shape[0])
    print("coarse state shape:", coarse_states.shape)
    print("primitive count:", len(action_sequences))
    print("highres state shape:", highres_states.shape)
    print("highres control shape:", highres_controls.shape)
    print("highres timestep shape:", highres_timesteps.shape)

    assert highres_states.shape == reported_highres.shape, (
        f"High-res state shape mismatch: actions-path states={highres_states.shape}, "
        f"highres={reported_highres.shape}"
    )
    assert np.allclose(highres_states, reported_highres, atol=atol, rtol=rtol), (
        "Reported states from high-resolution accessors differ"
    )
    assert highres_states.shape[0] == highres_controls.shape[0] + 1, (
        f"High-res warm start mismatch: states={highres_states.shape[0]}, "
        f"controls={highres_controls.shape[0]}"
    )
    assert highres_controls.shape[0] == highres_timesteps.shape[0], (
        f"Control/timestep length mismatch: controls={highres_controls.shape[0]}, "
        f"timesteps={highres_timesteps.shape[0]}"
    )
    print("[OK] raw path accessors are internally consistent")

    rollout = np.empty_like(highres_states)
    rollout[0] = np.asarray(planner.start, dtype=np.float64)
    curr = rollout[0].copy()

    for i, (u, dt) in enumerate(zip(highres_controls, highres_timesteps)):
        next_state, _ = planner.agent.get_next_state(curr, u, float(dt), num_steps=1)
        rollout[i + 1] = next_state
        curr = next_state

    max_abs_err = float(np.max(np.abs(rollout - highres_states)))
    print("live-dynamics rollout shape:", rollout.shape)
    print("live-dynamics max_abs_err:", max_abs_err)
    print("rolled final state:", rollout[-1])
    print("reported final state:", highres_states[-1])
    print(
        "Note: raw quadcopter Db-RRT primitives are transformed by position shift, "
        "so this rollout diagnostic is informative only and is not asserted."
    )


def run_search(planner):
    t0 = time.time()
    planner.plan_path()
    elapsed = time.time() - t0

    print("\n===== QuadCopter6D Db-RRT result =====")
    print("path_found:", planner.path_found)
    print("num_nodes:", len(planner.tree.nodes))
    print("path_cost:", planner.path_cost)
    print("path_time:", planner.path_time)
    print("wall_time:", elapsed)

    assert len(planner.tree.nodes) > 1, "Planner did not add any nodes beyond the start."

    if not planner.path_found:
        print("\nNo path found in this quadcopter Db-RRT smoke test.")
        print("Things to try:")
        print("  - increase delta, e.g., 0.6")
        print("  - increase max_candidate_motions_per_expand")
        print("  - increase num_edges in load_quadcopter_primitives")
        print("  - make the goal easier / larger goal_radius")
        return False

    ids, states, action_sequences, timesteps = planner.get_path_to_node_id(planner.goal_node_id)
    highres, highres_controls, highres_timesteps = planner.get_high_resolution_path_and_actions()

    print("\n===== Path summary =====")
    print("coarse node ids:", ids)
    print("coarse states shape:", states.shape)
    print("first action sequence shape:", action_sequences[0].shape)
    print("timesteps:", timesteps)
    print("highres shape:", highres.shape)
    print("highres controls shape:", highres_controls.shape)
    print("final highres state:", highres[-1])

    assert highres.shape[0] > 1, "High-resolution path is empty or has only start."
    reached, dist = planner.reached_goal(highres[-1], planner.goal, planner.goal_radius, planner.agent)
    relaxed_dist = np.linalg.norm(highres[-1][:3] - planner.goal[:3])
    assert reached or relaxed_dist <= planner.delta, (
        f"Final state is neither inside goal nor within delta; dist={dist}, relaxed_dist={relaxed_dist}"
    )
    print(f"[OK] final state reaches relaxed goal condition, dist={relaxed_dist:.3f}")
    inspect_dbrrt_returned_path(planner)

    return True


def maybe_optimize(planner):
    try:
        t0 = time.time()
        result = optimize_dbrrt_quadcopter6d_path(
            planner,
            options=Quadcopter6DTrajOptOptions(),
        )
        elapsed = time.time() - t0
    except ImportError as exc:
        print(f"\nSkipping Crocoddyl repair stage: {exc}")
        print("If Crocoddyl is installed, this test can also optimize the Db-RRT warm start.")
        return None

    print("\n===== QuadCopter6D optimization result =====")
    print("success:", result.success)
    print("feasible:", result.feasible)
    print("solver_iters:", result.solver_iters)
    print("cost:", result.cost)
    print("wall_time:", elapsed)
    print("final optimized state:", result.xs[-1])
    verify_rollout_consistency(result.path_view)
    return result


if __name__ == "__main__":
    planner = make_planner(debug=False, udf_seed=np.random.randint(0, 1000))
    ok = run_search(planner)
    if ok:
        opt_result = maybe_optimize(planner)
        print("\nSmoke test passed.")

        path_ids, _, _, _ = planner.get_path()
        save_quadcopter_rrt_tree_3d(
            planner,
            "media/db_rrt_quadcopter6d_tree_3d.png",
            print_tree=True,
        )
        print("\nSaved 3D raw tree to media/db_rrt_quadcopter6d_tree_3d.png")

        raw_states, _, _ = planner.get_high_resolution_path_and_actions()
        save_quadcopter_path_3d(
            env=planner.env,
            start=planner.start,
            goal=planner.goal,
            goal_radius=planner.goal_radius,
            highres_states=raw_states,
            filename="media/db_rrt_quadcopter6d_raw_path_3d.png",
            agent_radius=planner.agent.radius,
            path_color="xkcd:orange",
            path_label="Raw Db-RRT path",
        )
        print("Saved 3D raw path to media/db_rrt_quadcopter6d_raw_path_3d.png")

        if opt_result is not None:
            save_quadcopter_path_overlay_3d(
                planner,
                opt_result.path_view,
                "media/db_rrt_quadcopter6d_optimized_overlay_3d.png",
            )
            print("Saved 3D optimized overlay to media/db_rrt_quadcopter6d_optimized_overlay_3d.png")
