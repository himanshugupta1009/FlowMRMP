"""
Single-agent Db-A* smoke test using the Dynoplan unicycle .msgpack primitives.

Run from the repo root:
    python Tests/test_db_astar_unicycle.py

This is intentionally a single-agent test. Do this before plugging Db-A* into KCBS.
"""

import os
import sys
import time
from dataclasses import dataclass

import msgpack
import numpy as np

# Run from repo root OR from the Tests folder.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from mapf_env_square_agent_unicycle import get_unicycle_agent
from kd_tree_unicycle import CircularAngleIndexNumba
from db_astar import DbAStarPlanner, _transform_unicycle_trajectory_numba
from printer import DbAStarPrinter

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


def make_agent(agent_id=0):
    return get_unicycle_agent(agent_id)


def load_unicycle_primitives(num_edges=15000, dt=0.1):
    primitive_file = os.path.join(
        REPO_ROOT,
        "motion_primitives",
        "unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack",
    )
    if not os.path.exists(primitive_file):
        raise FileNotFoundError(f"Missing primitive file: {primitive_file}")

    with open(primitive_file, "rb") as f:
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
        lengths[i] = L
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


def make_planner(*, debug=False):
    # Empty environment first. Add obstacles only after this works.
    env = SquareEnvironment(12, 12, [], obs_buffers=False)
    agent = make_agent(0)
    motion_primitives, kd_tree = load_unicycle_primitives(num_edges=15000, dt=0.1)

    # Keep this deliberately easy. For a first test, the goal is to check plumbing,
    # primitive lookup, transform, collision checking, and path reconstruction.
    start = np.array([4.0, 4.0, 0.0], dtype=np.float64)
    goal = np.array([9.0, 9.0], dtype=np.float64)

    planner = DbAStarPlanner(
        start=start,
        goal=goal,
        goal_radius=0.25,
        env=env,
        agent=agent,
        motion_primitives=motion_primitives,
        alpha=0.5,
        delta=0.3,
        minimum_time_step=0.1,
        max_iter=10000,
        planning_time=120.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        transform_trajectory_function=_transform_unicycle_trajectory_numba,
        motion_primitive_kd_tree=kd_tree,
        get_motion_primitive_kd_tree_query=agent.get_eb_kd_tree_query,
        max_candidate_motions_per_expand=10,
        allow_intermediate_goal=False,
        cost_delta_factor=0.0,
        limit_branching_factor=10,
        duplicate_policy="soft",
        debug_flag=debug,
        print_logs=True,
    )
    return planner


def check_transform(planner):
    """A tiny unit check before running search."""
    edge_id = 0
    state = planner.start
    new_state, path = planner.transform_primitive(state, edge_id)

    assert path.ndim == 2, f"Expected path to be 2D, got shape {path.shape}"
    assert path.shape[1] == planner.agent.state_length, (
        f"Expected path state dim {planner.agent.state_length}, got {path.shape[1]}"
    )
    assert np.all(np.isfinite(path)), "Transformed primitive path contains NaN/Inf"
    assert np.allclose(new_state, path[-1]), "new_state should equal final transformed path state"
    print("[OK] transform_primitive sanity check passed")


def check_candidate_lookup(planner):
    ids = planner._get_candidate_edges(planner.start)
    assert len(ids) > 0, (
        "No primitives found near the start. Increase alpha*delta, "
        "or check the motion-primitive KD-tree query / primitive start_states."
    )
    print(f"[OK] primitive lookup returned {len(ids)} candidates at the start")


def run_search(planner):
    t0 = time.time()
    planner.plan_path()
    elapsed = time.time() - t0

    print("\n===== Db-A* result =====")
    print("path_found:", planner.path_found)
    print("num_nodes:", len(planner.tree.nodes))
    print("path_cost:", planner.path_cost)
    print("path_time:", planner.path_time)
    print("wall_time:", elapsed)

    assert planner._db_node_matrix.count > 1, \
    "Planner did not add any nodes beyond the start."

    if not planner.path_found:
        print("\nNo path found in this smoke test with the Dynoplan primitive file.")
        print("Things to try:")
        print("  - increase delta, e.g., 1.2")
        print("  - increase max_candidate_motions_per_expand")
        print("  - increase num_edges in load_unicycle_primitives")
        print("  - make the goal easier / larger goal_radius")
        return False

    # ids, states, controls, timesteps = planner.get_path()
    # highres = planner.get_high_resolution_path_numpy_array()

    ids, states, action_sequences, timesteps = planner.get_dbastar_path_to_node_id(
                        planner.goal_node_id)

    highres, highres_action_sequences, primitive_timesteps = (
                planner.get_dbastar_high_resolution_path_and_actions())

    print("\n===== Path summary =====")
    print("coarse node ids:", ids)
    print("coarse states shape:", states.shape)
    print("action sequences shape:", action_sequences[0].shape)
    print("timesteps:", timesteps)
    print("highres shape:", highres.shape)
    print("final highres state:", highres[-1])

    assert highres.shape[0] > 1, "High-resolution path is empty or has only start."
    reached, dist = planner.reached_goal(highres[-1], planner.goal, planner.goal_radius, planner.agent)
    assert reached, f"Final state is not inside goal; dist={dist}"
    print(f"[OK] final state reaches goal, dist={dist:.3f}")
    return True


if __name__ == "__main__":
    planner = make_planner(debug=False)
    check_transform(planner)
    check_candidate_lookup(planner)
    ok = run_search(planner)
    if ok:
        print("\nSmoke test passed.")

    dbp = DbAStarPrinter(planner.env, planner)
    dbp.print_db_astar("media/db_astar_graph.png")
    # or:
    # dbp.show_db_astar()
    # dbp.print_db_astar(
    # "media/db_astar_graph_labeled.png",
    # annotate_nodes=True,
    # annotate_scores=True,
    # )
