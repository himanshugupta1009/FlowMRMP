#!/usr/bin/env python3
"""Create a self-contained Franka edge-bundle dataset from CuRobo trajectories.

Every eligible trajectory timestep is a raw edge start.  Its edge is the next
``max_edge_steps`` acceleration intervals, or the shorter remaining suffix near
the end of the source trajectory.
Bundle queries are newly sampled valid joint-position/velocity pairs rather
than copied trajectory states.  Bundles use a fixed-radius neighborhood and cap
the raw candidate pool at 128.  Every candidate is re-integrated from the exact
query state; candidates that violate acceleration, jerk, joint-position,
joint-velocity, or self-collision constraints anywhere along their complete
duration are rejected before FPS selects 32 diverse, fully executable edges.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
if str(MRMP_SRC) not in sys.path:
    sys.path.insert(0, str(MRMP_SRC))

from Agents.FrankaPanda import FrankaSelfCollisionChecker  # noqa: E402


DEFAULT_INPUT = ROOT_DIR / "data" / "dataset200k.h5"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5"
)
DEFAULT_NUM_SAMPLES = 350_000
DEFAULT_URDF = ROOT_DIR / "assets" / "robots" / "panda" / "panda.urdf"
DEFAULT_CANDIDATE_POOL = 128
DEFAULT_MAX_EDGE_STEPS = 50

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
Q_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
Q_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
DQ_MAX = np.array(
    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    dtype=np.float32,
)
DDQ_MAX = np.full(7, 15.0, dtype=np.float32)
DDDQ_MAX = np.full(7, 500.0, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--set-size", type=int, default=32)
    parser.add_argument(
        "--max-edge-steps",
        type=int,
        default=DEFAULT_MAX_EDGE_STEPS,
        help=(
            "Maximum executed acceleration intervals per raw edge "
            f"(default: {DEFAULT_MAX_EDGE_STEPS})."
        ),
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help=(
            "Number of raw radius neighbors uniformly capped before complete "
            "query-rollout validation and FPS "
            f"(default: {DEFAULT_CANDIDATE_POOL})."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help=(
            "Normalized 14D radius. Default: median distance to the "
            "--candidate-pool nearest neighbor."
        ),
    )
    parser.add_argument("--radius-samples", type=int, default=20_000)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Forked CPU workers for independent complete candidate-rollout "
            "validation (default: min(8, logical CPU count))."
        ),
    )
    parser.add_argument(
        "--query-sampling",
        choices=("uniform-valid", "source-starts"),
        default="uniform-valid",
        help=(
            "Sample new uniform valid (q,dq) pairs (default), or reproduce the "
            "legacy source-start query policy."
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def normalize_conditions(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    q_norm = 2.0 * (q - Q_LOWER) / (Q_UPPER - Q_LOWER) - 1.0
    dq_norm = dq / DQ_MAX
    return np.concatenate((q_norm, dq_norm), axis=-1).astype(np.float32)


def denormalize_conditions(conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conditions = np.asarray(conditions, dtype=np.float32)
    q = Q_LOWER + 0.5 * (conditions[..., :7] + 1.0) * (Q_UPPER - Q_LOWER)
    dq = conditions[..., 7:] * DQ_MAX
    return q.astype(np.float32, copy=False), dq.astype(np.float32, copy=False)


def sample_uniform_valid_conditions(
    count: int,
    *,
    rng: np.random.Generator,
    collision_checker: FrankaSelfCollisionChecker,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample uniform limit-valid states and reject self-colliding q values."""
    accepted_parts: list[np.ndarray] = []
    accepted = 0
    attempted = 0
    collision_rejections = 0
    while accepted < count:
        draw_count = min(batch_size, max(32, count - accepted))
        candidates = rng.uniform(-1.0, 1.0, size=(draw_count, 14)).astype(
            np.float32
        )
        q, _ = denormalize_conditions(candidates)
        keep = np.ones(draw_count, dtype=bool)
        for index, configuration in enumerate(q):
            if collision_checker.in_collision(configuration):
                keep[index] = False
        valid = candidates[keep]
        accepted_parts.append(valid)
        accepted += len(valid)
        attempted += draw_count
        collision_rejections += int((~keep).sum())
    result = np.concatenate(accepted_parts, axis=0)[:count]
    return result, {
        "attempted_uniform_states": int(attempted),
        "self_collision_rejections": int(collision_rejections),
    }


def load_raw_edge_index(
    source: h5py.File, *, max_edge_steps: int
) -> dict[str, np.ndarray | list[str] | float]:
    names = sorted(source.keys())
    counts = np.array(
        [max(0, source[name]["positions"].shape[0] - 1) for name in names],
        dtype=np.int64,
    )
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Input contains no trajectory with at least two states.")

    start_q = np.empty((total, 7), dtype=np.float32)
    start_dq = np.empty((total, 7), dtype=np.float32)
    delta_q = np.empty((total, 7), dtype=np.float32)
    delta_dq = np.empty((total, 7), dtype=np.float32)
    final_q = np.empty((total, 7), dtype=np.float32)
    final_dq = np.empty((total, 7), dtype=np.float32)
    trajectory_index = np.empty(total, dtype=np.int32)
    start_index = np.empty(total, dtype=np.int32)
    num_steps = np.empty(total, dtype=np.int32)
    trajectory_length = np.empty(total, dtype=np.int32)
    duration = np.empty(total, dtype=np.float32)
    acceleration_lengths = np.asarray(
        [source[name]["accelerations"].shape[0] for name in names],
        dtype=np.int64,
    )
    acceleration_offsets = np.zeros(len(names), dtype=np.int64)
    if len(names) > 1:
        acceleration_offsets[1:] = np.cumsum(acceleration_lengths[:-1])
    accelerations = np.empty(
        (int(acceleration_lengths.sum()), 7), dtype=np.float32
    )

    dt_values: list[float] = []
    cursor = 0
    for traj_idx, name in enumerate(names):
        group = source[name]
        q = group["positions"][:].astype(np.float32, copy=False)
        dq = group["velocities"][:].astype(np.float32, copy=False)
        ddq = group["accelerations"][:].astype(np.float32, copy=False)
        if ddq.shape != q.shape:
            raise ValueError(
                f"{name}: accelerations shape {ddq.shape} does not match "
                f"positions shape {q.shape}"
            )
        acceleration_offset = int(acceleration_offsets[traj_idx])
        accelerations[
            acceleration_offset : acceleration_offset + len(ddq)
        ] = ddq
        edge_count = max(0, q.shape[0] - 1)
        if edge_count == 0:
            continue

        dt = float(group.attrs.get("dt", source.attrs.get("interpolation_dt", 0.02)))
        dt_values.append(dt)
        sl = slice(cursor, cursor + edge_count)
        indices = np.arange(edge_count, dtype=np.int32)
        remaining_steps = q.shape[0] - 1 - indices
        edge_steps = np.minimum(remaining_steps, max_edge_steps).astype(
            np.int32, copy=False
        )
        final_indices = indices + edge_steps

        start_q[sl] = q[:-1]
        start_dq[sl] = dq[:-1]
        final_q[sl] = q[final_indices]
        final_dq[sl] = dq[final_indices]
        delta_q[sl] = final_q[sl] - start_q[sl]
        delta_dq[sl] = final_dq[sl] - start_dq[sl]
        trajectory_index[sl] = traj_idx
        start_index[sl] = indices
        num_steps[sl] = edge_steps
        trajectory_length[sl] = edge_steps + 1
        duration[sl] = edge_steps * dt
        cursor += edge_count

    unique_dt = np.unique(np.asarray(dt_values, dtype=np.float64))
    if unique_dt.size != 1:
        raise ValueError(f"Expected one shared dt; found {unique_dt.tolist()}")

    conditions_norm = normalize_conditions(start_q, start_dq)
    max_duration = float(duration.max())

    return {
        "trajectory_names": names,
        "trajectory_index": trajectory_index,
        "start_index": start_index,
        "num_steps": num_steps,
        "trajectory_length": trajectory_length,
        "duration": duration,
        "start_q": start_q,
        "start_dq": start_dq,
        "delta_q": delta_q,
        "delta_dq": delta_dq,
        "final_q": final_q,
        "final_dq": final_dq,
        "conditions_norm": conditions_norm,
        "accelerations": accelerations,
        "acceleration_offsets": acceleration_offsets,
        "dt": float(unique_dt[0]),
        "max_duration": max_duration,
    }


def measure_radius(
    tree: cKDTree,
    query_conditions: np.ndarray,
    *,
    candidate_pool: int,
    set_size: int,
) -> tuple[float, dict[str, object]]:
    distances, _ = tree.query(
        query_conditions, k=candidate_pool, workers=-1
    )
    kth_distances = np.asarray(distances)[:, -1]
    percentiles = {
        str(p): float(np.percentile(kth_distances, p))
        for p in (5, 25, 50, 75, 90, 95, 99)
    }
    radius = float(np.median(kth_distances))
    counts = tree.query_ball_point(
        query_conditions, r=radius, return_length=True, workers=-1
    )
    report = {
        "sample_count": int(len(query_conditions)),
        "neighbor_rank": int(candidate_pool),
        "distance_percentiles": percentiles,
        "median_radius": radius,
        "coverage_at_least_set_size": float(np.mean(counts >= set_size)),
        "coverage_at_least_candidate_pool": float(np.mean(counts >= candidate_pool)),
        "neighbor_count_percentiles": {
            str(p): float(np.percentile(counts, p)) for p in (5, 50, 95)
        },
    }
    return radius, report


def farthest_point_sample(
    features: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    n = features.shape[0]
    if n < k:
        raise ValueError(f"FPS requires at least {k} candidates, received {n}.")
    selected = np.empty(k, dtype=np.int64)
    selected[0] = rng.integers(n)
    min_dist2 = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        diff = features - features[selected[i - 1]]
        dist2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))
    return selected


def integrate_candidate_rollouts(
    query_condition: np.ndarray,
    candidate_ids: np.ndarray,
    raw: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate raw acceleration suffixes from one exact 14D query state.

    Returns ``(dynamically_valid, q_path, dq_path, counts)``.  Dynamic validity
    covers finite values, acceleration limits, intra-edge jerk limits, and
    every executed waypoint's joint-position and joint-velocity limits.
    Self-collision is checked separately so FPS can stop as soon as 32
    collision-free candidates have been selected.
    """
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    raw_trajectory = np.asarray(raw["trajectory_index"])
    raw_start = np.asarray(raw["start_index"])
    raw_num_steps = np.asarray(raw["num_steps"])
    acceleration_offsets = np.asarray(raw["acceleration_offsets"])
    accelerations = np.asarray(raw["accelerations"])
    counts = raw_num_steps[candidate_ids].astype(np.int64, copy=False)
    max_steps = int(raw_num_steps.max())
    flat_starts = (
        acceleration_offsets[raw_trajectory[candidate_ids]]
        + raw_start[candidate_ids]
    )
    steps = np.arange(max_steps, dtype=np.int64)[None, :]
    mask = steps < counts[:, None]
    safe_steps = np.minimum(steps, counts[:, None] - 1)
    actions = accelerations[flat_starts[:, None] + safe_steps].astype(
        np.float64, copy=False
    )
    actions = actions * mask[:, :, None]

    finite_actions = np.all(np.isfinite(actions), axis=(1, 2))
    acceleration_valid = np.all(
        ~mask[:, :, None] | (np.abs(actions) <= DDQ_MAX[None, None, :] + 1e-10),
        axis=(1, 2),
    )
    if max_steps > 1:
        jerk_mask = steps[:, 1:] < counts[:, None]
        jerk = np.diff(actions, axis=1) / float(raw["dt"])
        jerk_valid = np.all(
            ~jerk_mask[:, :, None]
            | (np.abs(jerk) <= DDDQ_MAX[None, None, :] + 1e-10),
            axis=(1, 2),
        )
    else:
        jerk_valid = np.ones(len(candidate_ids), dtype=bool)

    query_q, query_dq = denormalize_conditions(
        np.asarray(query_condition, dtype=np.float32)[None, :]
    )
    query_q = query_q[0].astype(np.float64)
    query_dq = query_dq[0].astype(np.float64)
    dt = float(raw["dt"])
    cumulative = np.cumsum(actions, axis=1)
    dq_path = query_dq[None, None, :] + dt * cumulative
    position_increment = (
        query_dq[None, None, :] * dt
        + dt * dt * (cumulative - 0.5 * actions)
    )
    q_path = query_q[None, None, :] + np.cumsum(position_increment, axis=1)
    finite_path = np.all(np.isfinite(q_path) & np.isfinite(dq_path), axis=(1, 2))
    q_valid = np.all(
        ~mask[:, :, None]
        | (
            (q_path >= Q_LOWER[None, None, :] - 1e-10)
            & (q_path <= Q_UPPER[None, None, :] + 1e-10)
        ),
        axis=(1, 2),
    )
    dq_valid = np.all(
        ~mask[:, :, None]
        | (np.abs(dq_path) <= DQ_MAX[None, None, :] + 1e-10),
        axis=(1, 2),
    )
    dynamically_valid = (
        finite_actions
        & acceleration_valid
        & jerk_valid
        & finite_path
        & q_valid
        & dq_valid
    )
    return dynamically_valid, q_path, dq_path, counts


def select_collision_free_fps(
    candidate_ids: np.ndarray,
    features: np.ndarray,
    q_paths: np.ndarray,
    counts: np.ndarray,
    *,
    set_size: int,
    collision_checker: FrankaSelfCollisionChecker,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, dict[str, int]]:
    """Select FPS members while rejecting complete self-colliding rollouts.

    Invalid candidates are removed before they can become bundle members.  The
    first collision-free member is uniform over all collision-free candidates;
    each later member is the farthest remaining collision-free candidate from
    the already accepted set.
    """
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    q_paths = np.asarray(q_paths, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    remaining = np.ones(len(candidate_ids), dtype=bool)
    selected_local: list[int] = []
    min_dist2 = np.full(len(candidate_ids), np.inf, dtype=np.float32)
    tested = 0
    collision_rejections = 0

    while len(selected_local) < set_size and np.any(remaining):
        remaining_ids = np.flatnonzero(remaining)
        if not selected_local:
            local_index = int(remaining_ids[rng.integers(len(remaining_ids))])
        else:
            local_index = int(
                remaining_ids[np.argmax(min_dist2[remaining_ids])]
            )
        remaining[local_index] = False
        tested += 1
        collided = False
        for configuration in q_paths[local_index, : counts[local_index]]:
            if collision_checker.in_collision(configuration):
                collided = True
                collision_rejections += 1
                break
        if collided:
            continue

        selected_local.append(local_index)
        diff = features - features[local_index]
        dist2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        min_dist2 = np.minimum(min_dist2, dist2)

    stats = {
        "self_collision_candidates_tested": int(tested),
        "self_collision_candidate_rejections": int(collision_rejections),
    }
    if len(selected_local) < set_size:
        return None, stats
    return candidate_ids[np.asarray(selected_local, dtype=np.int64)], stats


def validate_and_select_candidate_pool(
    query_condition: np.ndarray,
    candidate_ids: np.ndarray,
    raw: dict[str, object],
    *,
    set_size: int,
    collision_checker: FrankaSelfCollisionChecker,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, dict[str, int]]:
    (
        dynamically_valid,
        q_paths,
        dq_paths,
        counts,
    ) = integrate_candidate_rollouts(query_condition, candidate_ids, raw)
    dynamic_valid_count = int(dynamically_valid.sum())
    stats = {
        "dynamic_candidate_rejections": int((~dynamically_valid).sum()),
        "dynamic_valid_candidate_count": dynamic_valid_count,
        "insufficient_dynamic_candidates": int(dynamic_valid_count < set_size),
        "self_collision_candidates_tested": 0,
        "self_collision_candidate_rejections": 0,
        "insufficient_collision_free_candidates": 0,
    }
    if dynamic_valid_count < set_size:
        return None, stats

    valid_ids = candidate_ids[dynamically_valid]
    query_q, query_dq = denormalize_conditions(query_condition[None, :])
    final_indices = counts - 1
    row_indices = np.arange(len(candidate_ids))
    query_delta_q = q_paths[row_indices, final_indices] - query_q[0, None, :]
    query_delta_dq = dq_paths[row_indices, final_indices] - query_dq[0, None, :]
    query_features = np.concatenate(
        (
            (
                counts.astype(np.float64)
                * float(raw["dt"])
                / float(raw["max_duration"])
                * 0.5
            )[:, None],
            query_delta_q / (Q_UPPER - Q_LOWER)[None, :],
            query_delta_dq / (2.0 * DQ_MAX[None, :]),
        ),
        axis=1,
    ).astype(np.float32)
    chosen, collision_stats = select_collision_free_fps(
        valid_ids,
        query_features[dynamically_valid],
        q_paths[dynamically_valid],
        counts[dynamically_valid],
        set_size=set_size,
        collision_checker=collision_checker,
        rng=rng,
    )
    stats.update(collision_stats)
    stats["insufficient_collision_free_candidates"] = int(chosen is None)
    return chosen, stats


_VALIDATION_WORKER_RAW: dict[str, object] | None = None
_VALIDATION_WORKER_COLLISION_CHECKER: FrankaSelfCollisionChecker | None = None


def _initialize_validation_worker(urdf: str) -> None:
    global _VALIDATION_WORKER_COLLISION_CHECKER
    _VALIDATION_WORKER_COLLISION_CHECKER = FrankaSelfCollisionChecker(urdf)


def _validate_candidate_pool_worker(
    task: tuple[np.ndarray, np.ndarray, int, int],
) -> tuple[np.ndarray | None, dict[str, int]]:
    query_condition, candidate_ids, set_size, seed = task
    if (
        _VALIDATION_WORKER_RAW is None
        or _VALIDATION_WORKER_COLLISION_CHECKER is None
    ):
        raise RuntimeError("validation worker was not initialized")
    return validate_and_select_candidate_pool(
        query_condition,
        candidate_ids,
        _VALIDATION_WORKER_RAW,
        set_size=set_size,
        collision_checker=_VALIDATION_WORKER_COLLISION_CHECKER,
        rng=np.random.default_rng(seed),
    )


def build_bundles(
    tree: cKDTree,
    conditions: np.ndarray,
    raw: dict[str, object],
    *,
    num_samples: int,
    candidate_pool: int,
    set_size: int,
    radius: float,
    batch_size: int,
    rng: np.random.Generator,
    query_sampling: str,
    collision_checker: FrankaSelfCollisionChecker | None,
    validation_workers: int = 1,
    urdf: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if candidate_pool < set_size:
        raise ValueError("candidate_pool must be at least set_size")

    query_conditions = np.empty((num_samples, 14), dtype=np.float32)
    bundle_edge_ids = np.empty((num_samples, set_size), dtype=np.int64)
    accepted = 0
    attempted = 0
    self_collision_rejections = 0
    dynamic_candidate_rejections = 0
    insufficient_dynamic_candidates = 0
    insufficient_collision_free_candidates = 0
    collision_candidates_tested = 0
    collision_candidate_rejections = 0
    candidate_counts: list[int] = []
    dynamic_valid_counts: list[int] = []
    started = time.time()

    source_query_order = (
        rng.permutation(conditions.shape[0])
        if query_sampling == "source-starts"
        else None
    )
    source_offset = 0
    max_attempts = max(conditions.shape[0], num_samples * 20)
    validation_workers = max(1, int(validation_workers))
    worker_pool = None
    if validation_workers > 1:
        if urdf is None:
            raise ValueError("parallel validation requires a URDF path")
        global _VALIDATION_WORKER_RAW
        _VALIDATION_WORKER_RAW = raw
        worker_pool = mp.get_context("fork").Pool(
            processes=validation_workers,
            initializer=_initialize_validation_worker,
            initargs=(str(urdf),),
        )
    try:
        while accepted < num_samples and attempted < max_attempts:
            if query_sampling == "uniform-valid":
                if collision_checker is None:
                    raise ValueError(
                        "uniform-valid sampling requires a collision checker"
                    )
                batch_conditions, sampling_stats = sample_uniform_valid_conditions(
                    batch_size,
                    rng=rng,
                    collision_checker=collision_checker,
                    batch_size=batch_size,
                )
                self_collision_rejections += sampling_stats[
                    "self_collision_rejections"
                ]
            else:
                assert source_query_order is not None
                if source_offset >= len(source_query_order):
                    break
                batch_ids = source_query_order[
                    source_offset : source_offset + batch_size
                ]
                source_offset += len(batch_ids)
                batch_conditions = conditions[batch_ids]
            hit_lists = tree.query_ball_point(
                batch_conditions, r=radius, workers=-1
            )
            tasks: list[tuple[np.ndarray, np.ndarray, int, int]] = []
            task_queries: list[np.ndarray] = []
            for query_condition, hits in zip(batch_conditions, hit_lists):
                attempted += 1
                if len(hits) < candidate_pool:
                    continue
                hits_array = np.asarray(hits, dtype=np.int64)
                candidate_counts.append(int(hits_array.size))
                if hits_array.size > candidate_pool:
                    candidate_ids = rng.choice(
                        hits_array, size=candidate_pool, replace=False
                    )
                else:
                    candidate_ids = hits_array
                task_seed = int(
                    rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
                )
                tasks.append(
                    (
                        np.asarray(query_condition, dtype=np.float32),
                        candidate_ids,
                        set_size,
                        task_seed,
                    )
                )
                task_queries.append(query_condition)

            if worker_pool is None:
                if collision_checker is None:
                    raise ValueError("complete validation requires a collision checker")
                results = [
                    validate_and_select_candidate_pool(
                        query_condition,
                        candidate_ids,
                        raw,
                        set_size=task_set_size,
                        collision_checker=collision_checker,
                        rng=np.random.default_rng(task_seed),
                    )
                    for (
                        query_condition,
                        candidate_ids,
                        task_set_size,
                        task_seed,
                    ) in tasks
                ]
            else:
                results = worker_pool.map(
                    _validate_candidate_pool_worker, tasks, chunksize=1
                )

            for query_condition, (chosen, validation_stats) in zip(
                task_queries, results
            ):
                dynamic_candidate_rejections += validation_stats[
                    "dynamic_candidate_rejections"
                ]
                dynamic_valid_counts.append(
                    validation_stats["dynamic_valid_candidate_count"]
                )
                insufficient_dynamic_candidates += validation_stats[
                    "insufficient_dynamic_candidates"
                ]
                collision_candidates_tested += validation_stats[
                    "self_collision_candidates_tested"
                ]
                collision_candidate_rejections += validation_stats[
                    "self_collision_candidate_rejections"
                ]
                insufficient_collision_free_candidates += validation_stats[
                    "insufficient_collision_free_candidates"
                ]
                if chosen is None:
                    continue
                chosen.sort()
                query_conditions[accepted] = query_condition
                bundle_edge_ids[accepted] = chosen
                accepted += 1

                if accepted % 5_000 == 0:
                    elapsed = time.time() - started
                    print(
                        f"accepted={accepted:,}/{num_samples:,} "
                        f"attempted={attempted:,} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                if accepted == num_samples:
                    stats = {
                        "attempted_queries": int(attempted),
                        "accepted_queries": int(accepted),
                        "acceptance_fraction": float(accepted / attempted),
                        "self_collision_rejections_before_neighbor_query": int(
                            self_collision_rejections
                        ),
                        "dynamic_candidate_rejections": int(
                            dynamic_candidate_rejections
                        ),
                        "queries_with_fewer_than_set_size_dynamic_candidates": int(
                            insufficient_dynamic_candidates
                        ),
                        "self_collision_candidates_tested": int(
                            collision_candidates_tested
                        ),
                        "self_collision_candidate_rejections": int(
                            collision_candidate_rejections
                        ),
                        "queries_with_fewer_than_set_size_collision_free_candidates": int(
                            insufficient_collision_free_candidates
                        ),
                        "candidate_count_before_cap_percentiles": {
                            str(p): float(np.percentile(candidate_counts, p))
                            for p in (5, 50, 95)
                        },
                        "dynamic_valid_candidate_count_percentiles": {
                            str(p): float(np.percentile(dynamic_valid_counts, p))
                            for p in (5, 50, 95)
                        },
                        "elapsed_seconds": float(time.time() - started),
                        "validation_workers": int(validation_workers),
                    }
                    return query_conditions, bundle_edge_ids, stats
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()
        _VALIDATION_WORKER_RAW = None

    raise RuntimeError(
        f"Only found {accepted:,} eligible queries among {attempted:,} attempts."
    )


def _create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    kwargs: dict[str, object] = {}
    if data.ndim > 0 and data.size > 0:
        kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=data, **kwargs)


def write_bundle_outcomes(
    output: h5py.File,
    trajectory_group: h5py.Group,
    trajectory_names: list[str],
    raw: dict[str, object],
    split_conditions: dict[str, np.ndarray],
    split_edge_ids: dict[str, np.ndarray],
    *,
    batch_size: int = 256,
) -> None:
    """Store query-relative terminal outcomes for every selected bundle edge."""
    del trajectory_group, trajectory_names
    trajectory_offsets = np.asarray(raw["acceleration_offsets"])
    accelerations = np.asarray(raw["accelerations"])

    raw_trajectory = np.asarray(raw["trajectory_index"])
    raw_start = np.asarray(raw["start_index"])
    raw_num_steps = np.asarray(raw["num_steps"])
    max_steps = int(raw_num_steps.max())
    dt = float(output.attrs["dt"])
    if "bundle_outcomes" in output:
        raise ValueError("bundle_outcomes already exists")
    temporary_group_name = "_bundle_outcomes_building"
    if temporary_group_name in output:
        del output[temporary_group_name]
    outcome_group = output.create_group(temporary_group_name)
    outcome_group.attrs["reference_frame"] = (
        "Each selected raw acceleration sequence is integrated from its bundle "
        "query condition, not from the raw edge's original start state."
    )
    outcome_group.attrs["integrator"] = (
        "float64 exact piecewise-constant acceleration: "
        "dq_next=dq+ddq*dt; q_next=q+dq*dt+0.5*ddq*dt^2"
    )
    outcome_group.attrs["duration_formula"] = "duration = num_steps * dt"

    for split in ("train", "val"):
        conditions = split_conditions[split]
        edge_ids = split_edge_ids[split]
        num_bundles, set_size = edge_ids.shape
        chunk_rows = max(1, min(batch_size, num_bundles))
        scalar_chunks = (chunk_rows, set_size)
        vector_chunks = (chunk_rows, set_size, 7)
        datasets = {
            "num_steps": outcome_group.create_dataset(
                f"num_steps_{split}",
                shape=(num_bundles, set_size),
                dtype=np.int32,
                chunks=scalar_chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            ),
            "duration": outcome_group.create_dataset(
                f"duration_{split}",
                shape=(num_bundles, set_size),
                dtype=np.float32,
                chunks=scalar_chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            ),
        }
        for field in ("delta_q", "delta_dq", "final_q", "final_dq"):
            datasets[field] = outcome_group.create_dataset(
                f"{field}_{split}",
                shape=(num_bundles, set_size, 7),
                dtype=np.float32,
                chunks=vector_chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

        for offset in range(0, num_bundles, batch_size):
            stop = min(offset + batch_size, num_bundles)
            ids = edge_ids[offset:stop]
            counts = raw_num_steps[ids].astype(np.int64, copy=False)
            flat_starts = (
                trajectory_offsets[raw_trajectory[ids]] + raw_start[ids]
            )
            steps = np.arange(max_steps, dtype=np.int64)[None, None, :]
            mask = steps < counts[:, :, None]
            safe_steps = np.minimum(steps, counts[:, :, None] - 1)
            actions = accelerations[flat_starts[:, :, None] + safe_steps].astype(
                np.float64
            )
            actions *= mask[:, :, :, None]

            cumulative = np.cumsum(actions, axis=2)
            delta_dq_path = dt * cumulative
            _, query_dq = denormalize_conditions(conditions[offset:stop])
            position_increment = (
                query_dq[:, None, None, :].astype(np.float64) * dt
                + dt * dt * (cumulative - 0.5 * actions)
            )
            delta_q_path = np.cumsum(position_increment, axis=2)
            batch_indices = np.arange(stop - offset)[:, None]
            edge_indices = np.arange(set_size)[None, :]
            final_indices = counts - 1
            delta_q = delta_q_path[
                batch_indices, edge_indices, final_indices
            ]
            delta_dq = delta_dq_path[
                batch_indices, edge_indices, final_indices
            ]
            query_q, query_dq = denormalize_conditions(conditions[offset:stop])
            final_q = query_q[:, None, :] + delta_q
            final_dq = query_dq[:, None, :] + delta_dq

            datasets["num_steps"][offset:stop] = counts.astype(np.int32)
            datasets["duration"][offset:stop] = (counts * dt).astype(np.float32)
            datasets["delta_q"][offset:stop] = delta_q.astype(np.float32)
            datasets["delta_dq"][offset:stop] = delta_dq.astype(np.float32)
            datasets["final_q"][offset:stop] = final_q.astype(np.float32)
            datasets["final_dq"][offset:stop] = final_dq.astype(np.float32)

        print(
            f"stored query-relative outcomes for {split}: {num_bundles:,} bundles",
            flush=True,
        )
    output.move(temporary_group_name, "bundle_outcomes")


def write_output(
    output_path: Path,
    source: h5py.File,
    raw: dict[str, object],
    query_conditions: np.ndarray,
    bundle_ids: np.ndarray,
    *,
    val_fraction: float,
    metadata: dict[str, object],
    rng: np.random.Generator,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    sample_perm = rng.permutation(query_conditions.shape[0])
    num_val = int(round(query_conditions.shape[0] * val_fraction))
    val_idx = sample_perm[:num_val]
    train_idx = sample_perm[num_val:]
    with h5py.File(temp_path, "w") as output:
        output.attrs["format_name"] = "franka_variable_length_edge_bundle"
        output.attrs["format_version"] = 4
        output.attrs["self_contained"] = True
        output.attrs["created_unix_time"] = time.time()
        output.attrs["dt"] = float(raw["dt"])

        source_group = output.create_group("source_trajectories")
        for key, value in source.attrs.items():
            source_group.attrs[key] = value
        names = raw["trajectory_names"]
        assert isinstance(names, list)
        source_group.create_dataset(
            "trajectory_names",
            data=np.asarray(names, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        for index, name in enumerate(names, start=1):
            source.copy(name, source_group, name=name)
            if index % 1_000 == 0:
                print(f"copied source trajectories: {index:,}/{len(names):,}", flush=True)

        raw_group = output.create_group("raw_edges")
        for field in (
            "trajectory_index",
            "start_index",
            "num_steps",
            "trajectory_length",
            "duration",
            "start_q",
            "start_dq",
            "delta_q",
            "delta_dq",
            "final_q",
            "final_dq",
        ):
            value = raw[field]
            assert isinstance(value, np.ndarray)
            _create_dataset(raw_group, field, value)
        raw_group.attrs["variable_length_resolution"] = (
            "For raw edge e: name=source_trajectories/trajectory_names["
            "raw_edges/trajectory_index[e]]; start=raw_edges/start_index[e]; "
            "N=raw_edges/num_steps[e]; q and dq use [start:start+N+1], "
            "and ddq uses accelerations[start:start+N]."
        )

        conditions_by_split: dict[str, np.ndarray] = {}
        edge_ids_by_split: dict[str, np.ndarray] = {}
        for split, indices in (("train", train_idx), ("val", val_idx)):
            selected_conditions = query_conditions[indices]
            selected_ids = bundle_ids[indices]
            _create_dataset(output, f"conds_{split}", selected_conditions)
            q, dq = denormalize_conditions(selected_conditions)
            raw_conditions = np.concatenate((q, dq), axis=1)
            _create_dataset(output, f"conds_raw_{split}", raw_conditions)
            _create_dataset(output, f"source_edge_ids_{split}", selected_ids)
            conditions_by_split[split] = selected_conditions
            edge_ids_by_split[split] = selected_ids

        write_bundle_outcomes(
            output,
            source,
            names,
            raw,
            conditions_by_split,
            edge_ids_by_split,
        )

        metadata["num_train"] = int(train_idx.size)
        metadata["num_val"] = int(val_idx.size)
        metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
        output.create_dataset(
            "metadata_json",
            data=np.asarray(metadata_json, dtype=h5py.string_dtype("utf-8")),
        )

    os.replace(temp_path, output_path)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in [0, 1)")
    if args.set_size <= 0:
        raise ValueError("set-size must be positive")
    if args.max_edge_steps <= 0:
        raise ValueError("max-edge-steps must be positive")
    if args.candidate_pool < args.set_size:
        raise ValueError("candidate-pool must be at least set-size")
    if args.radius_samples <= 0:
        raise ValueError("radius-samples must be positive")
    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive")
    if args.validation_workers <= 0:
        raise ValueError("validation-workers must be positive")
    if args.radius is not None and args.radius <= 0.0:
        raise ValueError("radius must be positive when supplied")

    rng = np.random.default_rng(args.seed)
    total_started = time.time()
    collision_checker = None
    with h5py.File(args.input, "r") as source:
        try:
            print(f"loading raw-edge index from {args.input}", flush=True)
            raw = load_raw_edge_index(
                source, max_edge_steps=args.max_edge_steps
            )
            conditions = raw["conditions_norm"]
            assert isinstance(conditions, np.ndarray)
            print(f"eligible raw edges: {conditions.shape[0]:,}", flush=True)
            print("building 14D condition KD-tree", flush=True)
            tree = cKDTree(conditions)

            radius_count = min(args.radius_samples, conditions.shape[0])
            collision_checker = FrankaSelfCollisionChecker(args.urdf)
            if args.query_sampling == "uniform-valid":
                radius_queries, radius_sampling_stats = sample_uniform_valid_conditions(
                    radius_count,
                    rng=rng,
                    collision_checker=collision_checker,
                    batch_size=args.query_batch_size,
                )
            else:
                radius_query_ids = rng.choice(
                    conditions.shape[0], size=radius_count, replace=False
                )
                radius_queries = conditions[radius_query_ids]
                radius_sampling_stats = {
                    "attempted_uniform_states": 0,
                    "self_collision_rejections": 0,
                }
            measured_radius, radius_report = measure_radius(
                tree,
                radius_queries,
                candidate_pool=args.candidate_pool,
                set_size=args.set_size,
            )
            radius_report["query_sampling"] = args.query_sampling
            radius_report["valid_query_sampling"] = radius_sampling_stats
            radius = measured_radius if args.radius is None else float(args.radius)
            print(json.dumps(radius_report, indent=2, sort_keys=True), flush=True)
            print(f"using radius: {radius:.12f}", flush=True)

            query_conditions, bundle_ids, bundle_stats = build_bundles(
                tree,
                conditions,
                raw,
                num_samples=args.num_samples,
                candidate_pool=args.candidate_pool,
                set_size=args.set_size,
                radius=radius,
                batch_size=args.query_batch_size,
                rng=rng,
                query_sampling=args.query_sampling,
                collision_checker=collision_checker,
                validation_workers=args.validation_workers,
                urdf=args.urdf,
            )

            metadata: dict[str, object] = {
            "source_dataset_original_path": str(args.input.resolve()),
            "source_is_copied_into_output": True,
            "num_samples": int(args.num_samples),
            "set_size": int(args.set_size),
            "candidate_pool": int(args.candidate_pool),
            "val_fraction": float(args.val_fraction),
            "cond_dim": 14,
            "per_step_state_dim": 14,
            "per_step_action_dim": 7,
            "max_edge_steps": int(args.max_edge_steps),
            "edge_length": (
                "variable; min(max_edge_steps, remaining source intervals)"
            ),
            "condition_fields": [
                *[f"{name}_position_norm" for name in JOINT_NAMES],
                *[f"{name}_velocity_norm" for name in JOINT_NAMES],
            ],
            "normalization": {
                "q_lower": Q_LOWER.tolist(),
                "q_upper": Q_UPPER.tolist(),
                "dq_max_abs": DQ_MAX.tolist(),
                "ddq_max_abs": DDQ_MAX.tolist(),
                "dddq_max_abs": DDDQ_MAX.tolist(),
                "q_formula": "2 * (q - q_lower) / (q_upper - q_lower) - 1",
                "dq_formula": "dq / dq_max_abs",
            },
            "dt": float(raw["dt"]),
            "max_suffix_duration": float(raw["max_duration"]),
            "radius": float(radius),
            "radius_was_measured_median": args.radius is None,
            "radius_analysis": radius_report,
            "query_sampling": (
                "new q and dq sampled independently and uniformly inside the Franka "
                "joint/velocity limits; self-colliding q rejected; accept only when "
                "the raw radius-neighbor pool yields set_size complete dynamically "
                "valid and self-collision-free query-executed edges"
                if args.query_sampling == "uniform-valid"
                else "unique raw-edge start states from dataset support"
            ),
            "candidate_selection": (
                "all raw-edge starts within radius; uniformly cap at candidate_pool; "
                "re-integrate every capped candidate from the exact query; reject "
                "complete rollouts violating acceleration, intra-edge jerk, joint "
                "position, joint velocity, or self-collision constraints before FPS"
            ),
            "fps_features": [
                "0.5 * duration / max_duration",
                "query-executed delta_q / (q_upper - q_lower) [7]",
                "query-executed delta_dq / (2 * dq_max_abs) [7]",
            ],
            "set_selection": f"farthest_point_sample_k{args.set_size}",
            "stored_set_order": "ascending raw edge ID",
            "duration_policy": (
                "num_steps=min(max_edge_steps, remaining source intervals); "
                "duration=num_steps*dt"
            ),
            "outcome_storage": (
                "raw_edges stores original-start relative and exact outcomes; "
                "bundle_outcomes stores relative and exact outcomes obtained by "
                "float64 integration from each bundle query condition"
            ),
            "additional_trajectory_filtering": (
                "whole-edge query-rollout filtering at every executed waypoint; "
                "queries that cannot supply set_size fully valid members are resampled"
            ),
            "full_edge_validity": {
                "required_for_every_selected_edge": True,
                "execution_horizon": "all stored num_steps; no validity truncation",
                "checks": [
                    "finite acceleration and state values",
                    "acceleration limits",
                    "intra-edge jerk limits",
                    "joint-position limits at every waypoint",
                    "joint-velocity limits at every waypoint",
                    "PyBullet self-collision at every waypoint",
                ],
            },
            "bundle_generation": bundle_stats,
                "seed": int(args.seed),
                "query_sampling_mode": args.query_sampling,
                "query_urdf": (
                    str(args.urdf.resolve())
                    if args.query_sampling == "uniform-valid"
                    else None
                ),
            }
            write_output(
                args.output,
                source,
                raw,
                query_conditions,
                bundle_ids,
                val_fraction=args.val_fraction,
                metadata=metadata,
                rng=rng,
            )
        finally:
            if collision_checker is not None:
                collision_checker.close()

    elapsed = time.time() - total_started
    print(f"saved {args.output}", flush=True)
    print(f"total elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
