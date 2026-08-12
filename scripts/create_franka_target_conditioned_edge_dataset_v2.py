#!/usr/bin/env python3
"""Build the v2 Franka target-conditioned dataset from validated pool128 edges.

Unlike v1, this generator does not merely reorder the 32 edges already stored
in the source bundle. For every source query it reconstructs a fresh pool of
128 nearby raw acceleration sequences, integrates each sequence from the exact
query state, rejects complete rollouts that violate dynamics or MorphIt/cuRobo
self-collision, and retains every valid member of the accepted pool.

Two target views are generated per base state: a position-only target drawn
from the fixed 100-problem benchmark distribution and a reachable full-state
target anchored at a validated candidate endpoint. Each final 32-edge bundle
contains the eight candidates with best actual target progress plus 24 diverse
candidates selected with farthest-point sampling. Failed FlowEBRRT path states
can be appended as explicitly tagged hard examples paired with their original
benchmark goals.

The output is a compact, resumable companion HDF5. Controls and source
trajectories remain in the immutable v4 source HDF5, which is referenced by
basename and SHA-256 rather than duplicated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_franka_edge_bundle_dataset import (  # noqa: E402
    DQ_MAX,
    Q_LOWER,
    Q_UPPER,
    denormalize_conditions,
    farthest_point_sample,
    integrate_candidate_rollouts,
    normalize_conditions,
)
from FrankaPanda import (  # noqa: E402
    DEFAULT_CUROBO_CONFIG,
    FrankaCuroboCollisionChecker,
)


FORMAT_NAME = "franka_target_conditioned_valid_pool_bundle"
FORMAT_VERSION = 2
DEFAULT_SOURCE = (
    Path.home()
    / "Downloads"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "Downloads"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v2.h5"
)
DEFAULT_PROBLEMS = (
    ROOT_DIR
    / "data"
    / "benchmarks"
    / "franka_reachable_morphit295_ab_100_goal_r040.npz"
)
DEFAULT_HARD_RUN = (
    ROOT_DIR
    / "results"
    / "franka_flow_eb_rrt"
    / "franka_reachable_morphit295_100_goal_r040_best249_20260805"
)


def parse_args() -> argparse.Namespace:
    """Define full, smoke, resume, and validation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--hard-run", type=Path, default=DEFAULT_HARD_RUN)
    parser.add_argument("--curobo-config", type=Path, default=DEFAULT_CUROBO_CONFIG)
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--set-size", type=int, default=32)
    parser.add_argument("--directed-count", type=int, default=8)
    parser.add_argument("--pool-retries", type=int, default=8)
    parser.add_argument(
        "--radius-multipliers",
        type=float,
        nargs="+",
        default=(1.0, 1.25, 1.5, 2.0),
        help="Deterministic radius expansion for sparse current-valid neighborhoods.",
    )
    parser.add_argument("--collision-batch-size", type=int, default=32768)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--hard-states-per-failed-problem", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--skip-hard-states", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an incomplete output; refuse completed/existing incompatible files.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_json(dataset: h5py.Dataset) -> dict[str, object]:
    """Decode a scalar UTF-8 JSON HDF5 dataset."""
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def load_raw_source(source: h5py.File) -> dict[str, object]:
    """Load the raw edge index and contiguous acceleration pool into RAM."""
    raw_group = source["raw_edges"]
    raw: dict[str, object] = {
        "trajectory_index": raw_group["trajectory_index"][:],
        "start_index": raw_group["start_index"][:],
        "num_steps": raw_group["num_steps"][:],
        "dt": float(source.attrs["dt"]),
    }
    names_raw = source["source_trajectories/trajectory_names"][:]
    names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in names_raw]
    lengths = np.asarray(
        [source["source_trajectories"][name]["accelerations"].shape[0] for name in names],
        dtype=np.int64,
    )
    offsets = np.zeros(len(names), dtype=np.int64)
    if len(names) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    accelerations = np.empty((int(lengths.sum()), 7), dtype=np.float32)
    for index, (name, offset, length) in enumerate(zip(names, offsets, lengths), 1):
        accelerations[offset : offset + length] = source["source_trajectories"][name][
            "accelerations"
        ][:]
        if index % 10_000 == 0:
            print(f"loaded accelerations {index:,}/{len(names):,}", flush=True)
    raw["acceleration_offsets"] = offsets
    raw["accelerations"] = accelerations
    raw["max_duration"] = float(np.max(np.asarray(raw["num_steps"]))) * float(
        raw["dt"]
    )
    return raw


def load_search_conditions(source: h5py.File) -> np.ndarray:
    """Load and normalize all 24M raw-edge start states for the KD-tree."""
    raw = source["raw_edges"]
    count = int(raw["start_q"].shape[0])
    conditions = np.empty((count, 14), dtype=np.float32)
    chunk = 500_000
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        conditions[start:stop] = normalize_conditions(
            raw["start_q"][start:stop], raw["start_dq"][start:stop]
        )
    return conditions


def load_benchmark_goals(path: Path) -> np.ndarray:
    """Load the exact 100-problem goal distribution used by evaluation."""
    with np.load(path, allow_pickle=False) as data:
        goals = data["goals"].astype(np.float32)
    if goals.ndim != 2 or goals.shape[1] != 14:
        raise ValueError(f"Expected benchmark goals shaped (N,14), got {goals.shape}")
    return goals


def load_hard_states(run_dir: Path, per_failed_problem: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample intermediate states from saved failed FlowEBRRT paths."""
    states: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    problem_ids: list[int] = []
    if per_failed_problem <= 0 or not run_dir.is_dir():
        return (
            np.empty((0, 14), dtype=np.float32),
            np.empty((0, 14), dtype=np.float32),
            np.empty((0,), dtype=np.int16),
        )
    for path in sorted((run_dir / "paths").glob("problem_*.npz")):
        problem_id = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=False) as data:
            if bool(data["success"]):
                continue
            path_states = data["states"].astype(np.float32)
            goal = data["goal"].astype(np.float32)
        if len(path_states) == 0:
            continue
        indices = np.unique(
            np.linspace(0, len(path_states) - 1, min(per_failed_problem, len(path_states))).round().astype(int)
        )
        for index in indices:
            states.append(path_states[index])
            goals.append(goal)
            problem_ids.append(problem_id)
    return (
        np.asarray(states, dtype=np.float32).reshape(-1, 14),
        np.asarray(goals, dtype=np.float32).reshape(-1, 14),
        np.asarray(problem_ids, dtype=np.int16),
    )


def collision_free_candidates(
    checker: FrankaCuroboCollisionChecker,
    q_paths: np.ndarray,
    counts: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return candidates whose every executed waypoint is self-collision-free."""
    candidate_count = len(q_paths)
    flat_parts = [q_paths[i, : int(counts[i])] for i in range(candidate_count)]
    lengths = np.asarray([len(part) for part in flat_parts], dtype=np.int64)
    flat = np.concatenate(flat_parts, axis=0).astype(np.float32, copy=False)
    valid = np.empty(len(flat), dtype=bool)
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        valid[start:stop] = checker.collision_free_mask(flat[start:stop])
    offsets = np.zeros(candidate_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    return np.asarray(
        [np.all(valid[offsets[i] : offsets[i + 1]]) for i in range(candidate_count)],
        dtype=bool,
    )


def validate_pool(
    query: np.ndarray,
    candidate_ids: np.ndarray,
    raw: dict[str, object],
    checker: FrankaCuroboCollisionChecker,
    *,
    collision_batch_size: int,
) -> dict[str, np.ndarray]:
    """Integrate and fully validate one tested pool, retaining all valid edges."""
    dynamic, q_paths, dq_paths, counts = integrate_candidate_rollouts(query, candidate_ids, raw)
    if not np.any(dynamic):
        return {}
    ids = candidate_ids[dynamic]
    q_valid_paths = q_paths[dynamic]
    dq_valid_paths = dq_paths[dynamic]
    valid_counts = counts[dynamic]
    collision_free = collision_free_candidates(
        checker, q_valid_paths, valid_counts, batch_size=collision_batch_size
    )
    if not np.any(collision_free):
        return {}
    ids = ids[collision_free]
    q_valid_paths = q_valid_paths[collision_free]
    dq_valid_paths = dq_valid_paths[collision_free]
    valid_counts = valid_counts[collision_free]
    rows = np.arange(len(ids))
    finals = valid_counts - 1
    query_q, query_dq = denormalize_conditions(query[None, :])
    final_q = q_valid_paths[rows, finals].astype(np.float32)
    final_dq = dq_valid_paths[rows, finals].astype(np.float32)
    return {
        "edge_ids": ids.astype(np.int64),
        "num_steps": valid_counts.astype(np.int32),
        "delta_q": (final_q - query_q[0]).astype(np.float32),
        "delta_dq": (final_dq - query_dq[0]).astype(np.float32),
        "final_q": final_q,
        "final_dq": final_dq,
    }


def find_valid_pool(
    query: np.ndarray,
    hits: np.ndarray,
    mandatory_ids: np.ndarray,
    raw: dict[str, object],
    checker: FrankaCuroboCollisionChecker,
    *,
    candidate_pool: int,
    set_size: int,
    retries: int,
    collision_batch_size: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Try deterministic pool128 draws until at least 32 edges survive.

    Regular rows seed each tested pool with their 32 source members because the
    original generator did not retain its rejected pool or per-row RNG state.
    The remaining slots are fresh radius neighbors, so v2 can retain useful
    candidates that were not present in the stored source bundle.
    """
    mandatory_ids = np.unique(np.asarray(mandatory_ids, dtype=np.int64))
    mandatory_ids = mandatory_ids[mandatory_ids >= 0]
    if len(mandatory_ids) > candidate_pool:
        raise ValueError("Mandatory source edges exceed candidate-pool size")
    available = np.setdiff1d(hits, mandatory_ids, assume_unique=False)
    fresh_count = candidate_pool - len(mandatory_ids)
    if len(available) < fresh_count:
        return {}, 0
    rng = np.random.default_rng(seed)
    for attempt in range(1, retries + 1):
        fresh_ids = (
            rng.choice(available, size=fresh_count, replace=False)
            if len(available) > fresh_count
            else available.copy()
        )
        candidate_ids = np.concatenate((mandatory_ids, fresh_ids))
        pool = validate_pool(
            query,
            np.asarray(candidate_ids, dtype=np.int64),
            raw,
            checker,
            collision_batch_size=collision_batch_size,
        )
        if pool and len(pool["edge_ids"]) >= set_size:
            order = np.argsort(pool["edge_ids"], kind="stable")
            return {key: value[order] for key, value in pool.items()}, attempt
    return {}, retries


def validate_pools_batch(
    queries: np.ndarray,
    candidate_lists: list[np.ndarray],
    raw: dict[str, object],
    checker: FrankaCuroboCollisionChecker,
    *,
    collision_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    """Validate many state-specific candidate pools in shared GPU batches."""
    dynamic_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    candidate_offsets = [0]
    for query, candidate_ids in zip(queries, candidate_lists):
        dynamic, q_paths, dq_paths, counts = integrate_candidate_rollouts(
            query, candidate_ids, raw
        )
        ids = candidate_ids[dynamic]
        dynamic_parts.append(
            (ids, q_paths[dynamic], dq_paths[dynamic], counts[dynamic])
        )
        candidate_offsets.append(candidate_offsets[-1] + len(ids))
    total_dynamic = candidate_offsets[-1]
    if total_dynamic:
        all_q_paths = np.concatenate([part[1] for part in dynamic_parts], axis=0)
        all_counts = np.concatenate([part[3] for part in dynamic_parts], axis=0)
        all_collision_free = collision_free_candidates(
            checker,
            all_q_paths,
            all_counts,
            batch_size=collision_batch_size,
        )
    else:
        all_collision_free = np.empty((0,), dtype=bool)

    results: list[dict[str, np.ndarray]] = []
    for index, (query, part) in enumerate(zip(queries, dynamic_parts)):
        ids, q_paths, dq_paths, counts = part
        keep = all_collision_free[candidate_offsets[index] : candidate_offsets[index + 1]]
        if not np.any(keep):
            results.append({})
            continue
        ids = ids[keep]
        q_paths = q_paths[keep]
        dq_paths = dq_paths[keep]
        counts = counts[keep]
        rows = np.arange(len(ids))
        finals = counts - 1
        query_q, query_dq = denormalize_conditions(query[None, :])
        final_q = q_paths[rows, finals].astype(np.float32)
        final_dq = dq_paths[rows, finals].astype(np.float32)
        results.append(
            {
                "edge_ids": ids.astype(np.int64),
                "num_steps": counts.astype(np.int32),
                "delta_q": (final_q - query_q[0]).astype(np.float32),
                "delta_dq": (final_dq - query_dq[0]).astype(np.float32),
                "final_q": final_q,
                "final_dq": final_dq,
            }
        )
    return results


def merge_pools(
    accumulated: dict[str, np.ndarray], new: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Union valid candidates from repeated pool128 draws by raw edge ID."""
    if not new:
        return accumulated
    if not accumulated:
        order = np.argsort(new["edge_ids"], kind="stable")
        return {key: value[order] for key, value in new.items()}
    combined = {
        key: np.concatenate((accumulated[key], new[key]), axis=0)
        for key in accumulated
    }
    _, unique_indices = np.unique(combined["edge_ids"], return_index=True)
    order = unique_indices[np.argsort(combined["edge_ids"][unique_indices], kind="stable")]
    return {key: value[order] for key, value in combined.items()}


def find_valid_pools_batch(
    queries: np.ndarray,
    hit_lists: list[np.ndarray],
    mandatory_rows: np.ndarray,
    raw: dict[str, object],
    checker: FrankaCuroboCollisionChecker,
    *,
    candidate_pool: int,
    set_size: int,
    retries: int,
    collision_batch_size: int,
    seeds: np.ndarray,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Accumulate valid candidates over deterministic pool128 draws in batch."""
    row_count = len(queries)
    accumulated: list[dict[str, np.ndarray]] = [{} for _ in range(row_count)]
    retry_counts = np.zeros(row_count, dtype=np.uint8)
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    mandatory = []
    available = []
    for hits, mandatory_ids in zip(hit_lists, mandatory_rows):
        required = np.unique(np.asarray(mandatory_ids, dtype=np.int64))
        required = required[required >= 0]
        mandatory.append(required)
        available.append(np.setdiff1d(hits, required, assume_unique=False))

    unresolved = np.arange(row_count, dtype=np.int64)
    for attempt in range(1, retries + 1):
        if len(unresolved) == 0:
            break
        candidate_lists: list[np.ndarray] = []
        active_rows: list[int] = []
        for row in unresolved:
            fresh_count = candidate_pool - len(mandatory[row])
            if fresh_count < 0 or len(available[row]) < fresh_count:
                continue
            fresh = (
                rngs[row].choice(available[row], size=fresh_count, replace=False)
                if len(available[row]) > fresh_count
                else available[row].copy()
            )
            candidate_lists.append(np.concatenate((mandatory[row], fresh)))
            active_rows.append(int(row))
        if not active_rows:
            break
        validated = validate_pools_batch(
            queries[np.asarray(active_rows)],
            candidate_lists,
            raw,
            checker,
            collision_batch_size=collision_batch_size,
        )
        for row, pool in zip(active_rows, validated):
            accumulated[row] = merge_pools(accumulated[row], pool)
            retry_counts[row] = attempt
        unresolved = np.asarray(
            [row for row in unresolved if len(accumulated[row].get("edge_ids", ())) < set_size],
            dtype=np.int64,
        )
    return accumulated, retry_counts


def normalized_endpoint_features(pool: dict[str, np.ndarray]) -> np.ndarray:
    """Encode duration and terminal deltas for diversity selection."""
    return np.concatenate(
        (
            (pool["num_steps"].astype(np.float32) / 50.0 * 0.5)[:, None],
            pool["delta_q"] / (Q_UPPER - Q_LOWER)[None, :],
            pool["delta_dq"] / (2.0 * DQ_MAX[None, :]),
        ),
        axis=1,
    ).astype(np.float32)


def make_variant(
    query: np.ndarray,
    target_raw: np.ndarray,
    goal_mode: bool,
    pool: dict[str, np.ndarray],
    *,
    directed_count: int,
    set_size: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Select eight best-progress plus 24 diverse edges for one target."""
    query_q, query_dq = denormalize_conditions(query[None, :])
    target_delta_q = target_raw[:7] - query_q[0]
    target_delta_dq = target_raw[7:] - query_dq[0]
    relative = np.concatenate(
        (
            2.0 * target_delta_q / (Q_UPPER - Q_LOWER),
            target_delta_dq / DQ_MAX,
        )
    ).astype(np.float32)
    if goal_mode:
        relative[7:] = 0.0
    current_distance = float(np.linalg.norm(relative[:7] if goal_mode else relative))
    edge_relative = np.concatenate(
        (
            2.0 * pool["delta_q"] / (Q_UPPER - Q_LOWER)[None, :],
            pool["delta_dq"] / DQ_MAX[None, :],
        ),
        axis=1,
    ).astype(np.float32)
    difference = edge_relative - relative[None, :]
    distances = np.linalg.norm(difference[:, :7] if goal_mode else difference, axis=1)
    progress = current_distance - distances
    directed = np.argsort(-progress, kind="stable")[:directed_count]
    remaining = np.setdiff1d(np.arange(len(distances)), directed, assume_unique=False)
    diverse_count = set_size - directed_count
    if diverse_count:
        diverse_local = farthest_point_sample(
            normalized_endpoint_features(pool)[remaining],
            diverse_count,
            np.random.default_rng(seed),
        )
        selected = np.concatenate((directed, remaining[diverse_local]))
    else:
        selected = directed
    cond = np.concatenate(
        (query, relative, np.asarray([float(goal_mode)], dtype=np.float32))
    ).astype(np.float32)
    return {
        "cond": cond,
        "target_raw": target_raw.astype(np.float32),
        "relative": relative,
        "selected_pool_slots": selected.astype(np.uint16),
        "selected_edge_ids": pool["edge_ids"][selected],
        "selected_num_steps": pool["num_steps"][selected],
        "selected_delta_q": pool["delta_q"][selected],
        "selected_delta_dq": pool["delta_dq"][selected],
        "selected_progress": progress[selected].astype(np.float32),
        "best_available_progress": float(np.max(progress)),
    }


POOL_FIELDS = {
    "edge_ids": (np.int64, ()),
    "num_steps": (np.int32, ()),
    "delta_q": (np.float32, (7,)),
    "delta_dq": (np.float32, (7,)),
    "final_q": (np.float32, (7,)),
    "final_dq": (np.float32, (7,)),
}


def create_split(group: h5py.Group, rows: int, set_size: int) -> None:
    """Create fixed base/variant tables and extendable valid-pool tables."""
    group.attrs["completed_rows"] = 0
    group.attrs["input_cursor"] = 0
    group.attrs["complete"] = False
    group.create_dataset("base_conditions", shape=(rows, 14), dtype=np.float32)
    group.create_dataset("base_conditions_raw", shape=(rows, 14), dtype=np.float32)
    group.create_dataset("base_source_index", shape=(rows,), dtype=np.int64)
    group.create_dataset("hard_state", shape=(rows,), dtype=np.uint8)
    group.create_dataset("benchmark_problem_id", shape=(rows,), dtype=np.int16)
    group.create_dataset("pool_offsets", shape=(rows + 1,), dtype=np.int64)
    group.create_dataset("pool_retry_count", shape=(rows,), dtype=np.uint8)
    group.create_dataset("pool_radius_multiplier", shape=(rows,), dtype=np.float32)
    group.create_dataset(
        "rejected_source_indices",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=(1024,),
        compression="lzf",
    )
    for name, (dtype, tail) in POOL_FIELDS.items():
        group.create_dataset(
            f"pool_{name}",
            shape=(0, *tail),
            maxshape=(None, *tail),
            dtype=dtype,
            chunks=(4096, *tail),
            compression="lzf",
        )
    variants = rows * 2
    group.create_dataset("conds", shape=(variants, 29), dtype=np.float32)
    group.create_dataset("target_states_raw", shape=(variants, 14), dtype=np.float32)
    group.create_dataset("target_relative_normalized", shape=(variants, 14), dtype=np.float32)
    group.create_dataset("goal_mode", shape=(variants,), dtype=np.uint8)
    group.create_dataset("target_source", shape=(variants,), dtype=np.uint8)
    group.create_dataset("selected_pool_slots", shape=(variants, set_size), dtype=np.uint16)
    group.create_dataset("selected_edge_ids", shape=(variants, set_size), dtype=np.int64)
    group.create_dataset("selected_num_steps", shape=(variants, set_size), dtype=np.int32)
    group.create_dataset("selected_delta_q", shape=(variants, set_size, 7), dtype=np.float32)
    group.create_dataset("selected_delta_dq", shape=(variants, set_size, 7), dtype=np.float32)
    group.create_dataset("selected_progress", shape=(variants, set_size), dtype=np.float32)
    group.create_dataset("best_available_progress", shape=(variants,), dtype=np.float32)


def append_pool(group: h5py.Group, pool: dict[str, np.ndarray]) -> tuple[int, int]:
    """Append one variable-length pool consistently across all flat datasets."""
    start = int(group["pool_edge_ids"].shape[0])
    stop = start + len(pool["edge_ids"])
    for name in POOL_FIELDS:
        dataset = group[f"pool_{name}"]
        dataset.resize((stop, *dataset.shape[1:]))
        dataset[start:stop] = pool[name]
    return start, stop


def repair_for_resume(group: h5py.Group) -> int:
    """Truncate any uncommitted pool tail after an interrupted row write."""
    completed = int(group.attrs["completed_rows"])
    committed = int(group["pool_offsets"][completed]) if completed else 0
    for name in POOL_FIELDS:
        dataset = group[f"pool_{name}"]
        if dataset.shape[0] != committed:
            dataset.resize((committed, *dataset.shape[1:]))
    return completed


def ensure_resumable_schema(group: h5py.Group, rows: int) -> None:
    """Add backward-compatible fields when resuming an earlier v2 partial file."""
    if "pool_radius_multiplier" not in group:
        dataset = group.create_dataset(
            "pool_radius_multiplier", shape=(rows,), dtype=np.float32
        )
        completed = int(group.attrs.get("completed_rows", 0))
        if completed:
            dataset[:completed] = 1.0
    if "base_source_index" not in group:
        dataset = group.create_dataset("base_source_index", shape=(rows,), dtype=np.int64)
        completed = int(group.attrs.get("completed_rows", 0))
        if completed:
            dataset[:completed] = np.arange(completed, dtype=np.int64)
    if "input_cursor" not in group.attrs:
        group.attrs["input_cursor"] = int(group.attrs.get("completed_rows", 0))
    if "rejected_source_indices" not in group:
        group.create_dataset(
            "rejected_source_indices",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1024,),
            compression="lzf",
        )


def append_rejected_source_index(group: h5py.Group, source_index: int) -> None:
    """Record one source state that cannot supply a current-valid K=32 pool."""
    dataset = group["rejected_source_indices"]
    index = int(dataset.shape[0])
    dataset.resize((index + 1,))
    dataset[index] = int(source_index)


def compact_completed_split(group: h5py.Group) -> None:
    """Trim fixed-capacity row tables to the final number of accepted states."""
    accepted = int(group.attrs["completed_rows"])
    base_datasets = (
        "base_conditions",
        "base_conditions_raw",
        "base_source_index",
        "hard_state",
        "benchmark_problem_id",
        "pool_retry_count",
        "pool_radius_multiplier",
    )
    variant_datasets = (
        "conds",
        "target_states_raw",
        "target_relative_normalized",
        "goal_mode",
        "target_source",
        "selected_pool_slots",
        "selected_edge_ids",
        "selected_num_steps",
        "selected_delta_q",
        "selected_delta_dq",
        "selected_progress",
        "best_available_progress",
    )
    resize_plan = (
        [(name, accepted) for name in base_datasets]
        + [("pool_offsets", accepted + 1)]
        + [(name, accepted * 2) for name in variant_datasets]
    )
    for name, count in resize_plan:
        source = group[name]
        if source.shape[0] == count:
            continue
        temporary = f"_compact_{name}"
        if temporary in group:
            del group[temporary]
        kwargs: dict[str, object] = {}
        if count and source.ndim:
            chunk_rows = min(4096, max(1, count))
            kwargs = {
                "chunks": (chunk_rows, *source.shape[1:]),
                "compression": "lzf",
            }
        compact = group.create_dataset(
            temporary,
            shape=(count, *source.shape[1:]),
            dtype=source.dtype,
            **kwargs,
        )
        copy_chunk = 4096
        for start in range(0, count, copy_chunk):
            stop = min(start + copy_chunk, count)
            compact[start:stop] = source[start:stop]
        del group[name]
        group.move(temporary, name)


def write_variant(group: h5py.Group, index: int, variant: dict[str, object], *, goal_mode: bool, target_source: int) -> None:
    """Write one final target-conditioned K=32 training row."""
    group["conds"][index] = variant["cond"]
    group["target_states_raw"][index] = variant["target_raw"]
    group["target_relative_normalized"][index] = variant["relative"]
    group["goal_mode"][index] = int(goal_mode)
    group["target_source"][index] = int(target_source)
    for name in (
        "selected_pool_slots",
        "selected_edge_ids",
        "selected_num_steps",
        "selected_delta_q",
        "selected_delta_dq",
        "selected_progress",
        "best_available_progress",
    ):
        group[name][index] = variant[name]


def process_split(
    output: h5py.File,
    split: str,
    queries: np.ndarray,
    source_indices: np.ndarray,
    mandatory_edge_ids: np.ndarray,
    forced_goals: np.ndarray,
    problem_ids: np.ndarray,
    hard_flags: np.ndarray,
    tree: cKDTree,
    raw: dict[str, object],
    benchmark_goals: np.ndarray,
    checker: FrankaCuroboCollisionChecker,
    args: argparse.Namespace,
    radius: float,
) -> None:
    """Generate one resumable split and flush progress at safe row boundaries."""
    group = output[split]
    ensure_resumable_schema(group, len(queries))
    completed_rows = repair_for_resume(group)
    start_input = int(group.attrs.get("input_cursor", completed_rows))
    started = time.perf_counter()
    # Radius queries over the 24.3M-point, 14D KD-tree dominate CPU time. Run
    # one checkpoint-sized query batch across all logical CPU cores, while row
    # validation remains deterministic and commits in source order.
    for batch_start in range(start_input, len(queries), args.checkpoint_every):
        batch_stop = min(batch_start + args.checkpoint_every, len(queries))
        raw_hit_lists = tree.query_ball_point(
            queries[batch_start:batch_stop], r=radius, workers=-1
        )
        hit_lists = [np.asarray(value, dtype=np.int64) for value in raw_hit_lists]
        batch_rows = np.arange(batch_start, batch_stop, dtype=np.int64)
        batch_pools, batch_retries = find_valid_pools_batch(
            queries[batch_start:batch_stop],
            hit_lists,
            mandatory_edge_ids[batch_start:batch_stop],
            raw,
            checker,
            candidate_pool=args.candidate_pool,
            set_size=args.set_size,
            retries=args.pool_retries,
            collision_batch_size=args.collision_batch_size,
            seeds=(
                args.seed
                + (0 if split == "train" else 1_000_000_000)
                + batch_rows * 17
            ),
        )
        for input_row, pool, retry_count in zip(
            range(batch_start, batch_stop), batch_pools, batch_retries
        ):
            query = queries[input_row]
            radius_multiplier = 1.0
            if len(pool.get("edge_ids", ())) < args.set_size:
                # Some configurations are valid under the active robot model
                # but the legacy radius contains too few current-valid motion
                # primitives. Expand only those sparse neighborhoods and keep
                # every dynamics/collision requirement unchanged.
                for multiplier_index, multiplier in enumerate(
                    args.radius_multipliers[1:], start=1
                ):
                    expanded_hits = np.asarray(
                        tree.query_ball_point(query, r=radius * multiplier),
                        dtype=np.int64,
                    )
                    expanded_pools, expanded_retries = find_valid_pools_batch(
                        query[None, :],
                        [expanded_hits],
                        mandatory_edge_ids[input_row : input_row + 1],
                        raw,
                        checker,
                        candidate_pool=args.candidate_pool,
                        set_size=args.set_size,
                        retries=args.pool_retries,
                        collision_batch_size=args.collision_batch_size,
                        seeds=np.asarray(
                            [
                                args.seed
                                + (0 if split == "train" else 1_000_000_000)
                                + input_row * 17
                                + multiplier_index * 10_000_019
                            ],
                            dtype=np.int64,
                        ),
                    )
                    pool = merge_pools(pool, expanded_pools[0])
                    retry_count = max(int(retry_count), int(expanded_retries[0]))
                    radius_multiplier = float(multiplier)
                    if len(pool.get("edge_ids", ())) >= args.set_size:
                        break
            if len(pool.get("edge_ids", ())) < args.set_size:
                append_rejected_source_index(
                    group, int(source_indices[input_row])
                )
                group.attrs["input_cursor"] = input_row + 1
                print(
                    f"{split}: skipped source index {int(source_indices[input_row])}; "
                    f"only {len(pool.get('edge_ids', ()))} current-valid candidates",
                    flush=True,
                )
                continue
            output_row = int(group.attrs["completed_rows"])
            raw_q, raw_dq = denormalize_conditions(query[None, :])
            query_raw = np.concatenate((raw_q[0], raw_dq[0])).astype(np.float32)
            pool_start, pool_stop = append_pool(group, pool)
            group["base_conditions"][output_row] = query
            group["base_conditions_raw"][output_row] = query_raw
            group["base_source_index"][output_row] = source_indices[input_row]
            group["hard_state"][output_row] = hard_flags[input_row]
            group["benchmark_problem_id"][output_row] = problem_ids[input_row]
            group["pool_offsets"][output_row] = pool_start
            group["pool_offsets"][output_row + 1] = pool_stop
            group["pool_retry_count"][output_row] = retry_count
            group["pool_radius_multiplier"][output_row] = radius_multiplier

            if hard_flags[input_row]:
                benchmark_target = forced_goals[input_row]
                benchmark_problem_id = int(problem_ids[input_row])
            else:
                benchmark_problem_id = int(
                    np.random.default_rng(args.seed + input_row * 31).integers(
                        len(benchmark_goals)
                    )
                )
                benchmark_target = benchmark_goals[benchmark_problem_id]
                group["benchmark_problem_id"][output_row] = benchmark_problem_id
            goal_variant = make_variant(
                query,
                benchmark_target,
                True,
                pool,
                directed_count=args.directed_count,
                set_size=args.set_size,
                seed=args.seed + input_row * 43,
            )
            anchor = int(
                np.random.default_rng(args.seed + input_row * 59).integers(
                    len(pool["edge_ids"])
                )
            )
            full_target = np.concatenate(
                (pool["final_q"][anchor], pool["final_dq"][anchor])
            )
            full_variant = make_variant(
                query,
                full_target,
                False,
                pool,
                directed_count=args.directed_count,
                set_size=args.set_size,
                seed=args.seed + input_row * 71,
            )
            write_variant(
                group,
                2 * output_row,
                goal_variant,
                goal_mode=True,
                target_source=0,
            )
            write_variant(
                group,
                2 * output_row + 1,
                full_variant,
                goal_mode=False,
                target_source=1,
            )
            group.attrs["completed_rows"] = output_row + 1
            group.attrs["input_cursor"] = input_row + 1
        output.flush()
        elapsed = time.perf_counter() - started
        rate = (batch_stop - start_input) / max(elapsed, 1e-9)
        accepted = int(group.attrs["completed_rows"])
        print(
            f"{split}: scanned={batch_stop:,}/{len(queries):,}; "
            f"accepted={accepted:,}; "
            f"valid_pool={pool_stop:,}; {rate:.2f} rows/s",
            flush=True,
        )
    compact_completed_split(group)
    group.attrs["complete"] = True
    output.flush()


def main() -> None:
    """Create or resume the v2 HDF5 and process both deterministic splits."""
    args = parse_args()
    if not 0 < args.directed_count <= args.set_size <= args.candidate_pool:
        raise ValueError("Require 0 < directed-count <= set-size <= candidate-pool")
    if args.pool_retries <= 0 or args.checkpoint_every <= 0:
        raise ValueError("pool-retries and checkpoint-every must be positive")
    if (
        not args.radius_multipliers
        or args.radius_multipliers[0] != 1.0
        or any(value <= 0 for value in args.radius_multipliers)
        or any(
            right <= left
            for left, right in zip(args.radius_multipliers, args.radius_multipliers[1:])
        )
    ):
        raise ValueError("radius-multipliers must be strictly increasing and start at 1.0")
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file() or not args.problems.is_file():
        raise FileNotFoundError("Source HDF5 and benchmark NPZ are required")
    source_hash = None if args.skip_source_hash else file_sha256(source_path)
    benchmark_goals = load_benchmark_goals(args.problems.resolve())
    hard_states, hard_goals, hard_problem_ids = (
        (np.empty((0, 14), np.float32), np.empty((0, 14), np.float32), np.empty(0, np.int16))
        if args.skip_hard_states
        else load_hard_states(args.hard_run.resolve(), args.hard_states_per_failed_problem)
    )

    with h5py.File(source_path, "r") as source:
        source_metadata = decode_json(source["metadata_json"])
        radius = float(source_metadata["radius"])
        checker = FrankaCuroboCollisionChecker(args.curobo_config.resolve())
        regular_queries: dict[str, np.ndarray] = {}
        regular_mandatory_ids: dict[str, np.ndarray] = {}
        regular_source_indices: dict[str, np.ndarray] = {}
        initial_collision_filter: dict[str, dict[str, int]] = {}
        for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
            all_queries = source[f"conds_{split}"][:].astype(np.float32)
            all_ids = source[f"source_edge_ids_{split}"][:].astype(np.int64)
            if limit is not None:
                all_queries = all_queries[:limit]
                all_ids = all_ids[:limit]
            all_q, _ = denormalize_conditions(all_queries)
            keep = np.empty(len(all_queries), dtype=bool)
            for start in range(0, len(all_queries), args.collision_batch_size):
                stop = min(start + args.collision_batch_size, len(all_queries))
                keep[start:stop] = checker.collision_free_mask(all_q[start:stop])
            regular_queries[split] = all_queries[keep]
            regular_mandatory_ids[split] = all_ids[keep]
            regular_source_indices[split] = np.flatnonzero(keep).astype(np.int64)
            initial_collision_filter[split] = {
                "checked": int(len(keep)),
                "retained": int(keep.sum()),
                "rejected": int((~keep).sum()),
            }
            print(
                f"{split} initial-state CuRobo filter: retained "
                f"{int(keep.sum()):,}/{len(keep):,}",
                flush=True,
            )

        if len(hard_states):
            hard_keep = np.empty(len(hard_states), dtype=bool)
            for start in range(0, len(hard_states), args.collision_batch_size):
                stop = min(start + args.collision_batch_size, len(hard_states))
                hard_keep[start:stop] = checker.collision_free_mask(
                    hard_states[start:stop, :7]
                )
            hard_states = hard_states[hard_keep]
            hard_goals = hard_goals[hard_keep]
            hard_problem_ids = hard_problem_ids[hard_keep]
            initial_collision_filter["hard"] = {
                "checked": int(len(hard_keep)),
                "retained": int(hard_keep.sum()),
                "rejected": int((~hard_keep).sum()),
            }
        else:
            initial_collision_filter["hard"] = {
                "checked": 0,
                "retained": 0,
                "rejected": 0,
            }
        # Hard states belong only to training so validation remains independent.
        train_queries = np.concatenate((regular_queries["train"], normalize_conditions(hard_states[:, :7], hard_states[:, 7:])), axis=0)
        split_queries = {"train": train_queries, "val": regular_queries["val"]}
        hard_count = len(hard_states)
        split_source_indices = {
            "train": np.concatenate(
                (regular_source_indices["train"], np.full(hard_count, -1, np.int64))
            ),
            "val": regular_source_indices["val"],
        }
        split_mandatory_ids = {
            "train": np.concatenate(
                (
                    regular_mandatory_ids["train"],
                    np.full((hard_count, args.set_size), -1, dtype=np.int64),
                ),
                axis=0,
            ),
            "val": regular_mandatory_ids["val"],
        }
        split_hard_flags = {
            "train": np.concatenate((np.zeros(len(regular_queries["train"]), np.uint8), np.ones(hard_count, np.uint8))),
            "val": np.zeros(len(regular_queries["val"]), np.uint8),
        }
        split_problem_ids = {
            "train": np.concatenate((np.full(len(regular_queries["train"]), -1, np.int16), hard_problem_ids)),
            "val": np.full(len(regular_queries["val"]), -1, np.int16),
        }
        split_forced_goals = {
            "train": np.concatenate((np.zeros((len(regular_queries["train"]), 14), np.float32), hard_goals), axis=0),
            "val": np.zeros((len(regular_queries["val"]), 14), np.float32),
        }

        print("loading raw source arrays", flush=True)
        raw = load_raw_source(source)
        print("loading 24M normalized search states", flush=True)
        search_conditions = load_search_conditions(source)
        print("building 14D KD-tree", flush=True)
        tree = cKDTree(search_conditions)

        mode = "r+" if output_path.exists() and args.resume else "w"
        if output_path.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, mode) as output:
            if mode == "w":
                output.attrs["format_name"] = FORMAT_NAME
                output.attrs["format_version"] = FORMAT_VERSION
                output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
                output.attrs["source_dataset_basename"] = source_path.name
                output.attrs["source_dataset_original_path"] = str(source_path)
                if source_hash is not None:
                    output.attrs["source_dataset_sha256"] = source_hash
                output.attrs["complete"] = False
                for split in ("train", "val"):
                    create_split(output.create_group(split), len(split_queries[split]), args.set_size)
                metadata = {
                    "format_name": FORMAT_NAME,
                    "format_version": FORMAT_VERSION,
                    "source_dataset_basename": source_path.name,
                    "source_dataset_sha256": source_hash,
                    "source_dataset_metadata": source_metadata,
                    "benchmark_problems": str(args.problems.resolve()),
                    "hard_example_run": None if args.skip_hard_states else str(args.hard_run.resolve()),
                    "hard_state_count": hard_count,
                    "initial_state_curobo_filter": initial_collision_filter,
                    "candidate_pool_tested": args.candidate_pool,
                    "radius_multipliers": list(args.radius_multipliers),
                    "pool_storage": "all complete dynamically valid and MorphIt/cuRobo self-collision-free candidates from the accepted tested pool",
                    "condition_dim": 29,
                    "variants_per_base_state": 2,
                    "target_sources": {"0": "fixed 100-problem benchmark goal", "1": "reachable validated candidate endpoint"},
                    "goal_metric": "normalized 7D joint-position L2; radius 0.40",
                    "selection": f"{args.directed_count} best actual goal-progress plus {args.set_size - args.directed_count} FPS-diverse candidates",
                    "seed": args.seed,
                }
                output.create_dataset("metadata_json", data=json.dumps(metadata, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
                output.flush()
            elif output.attrs.get("format_name") != FORMAT_NAME:
                raise ValueError(f"Cannot resume incompatible output {output_path}")
            else:
                # Keep metadata synchronized when a resumable generator gains
                # a backward-compatible safety feature between checkpoints.
                resume_metadata = decode_json(output["metadata_json"])
                resume_metadata["radius_multipliers"] = list(
                    args.radius_multipliers
                )
                resume_metadata["dead_state_policy"] = (
                    "record source index and skip when no tested neighborhood "
                    "supplies set_size current-valid candidates"
                )
                del output["metadata_json"]
                output.create_dataset(
                    "metadata_json",
                    data=json.dumps(resume_metadata, sort_keys=True),
                    dtype=h5py.string_dtype("utf-8"),
                )
                output.flush()

            try:
                for split in ("train", "val"):
                    process_split(
                        output,
                        split,
                        split_queries[split],
                        split_source_indices[split],
                        split_mandatory_ids[split],
                        split_forced_goals[split],
                        split_problem_ids[split],
                        split_hard_flags[split],
                        tree,
                        raw,
                        benchmark_goals,
                        checker,
                        args,
                        radius,
                    )
            finally:
                checker.close()
            output.attrs["complete"] = True
            output.attrs["completed_utc"] = datetime.now(timezone.utc).isoformat()
            output.flush()
    print(f"completed v2 dataset: {output_path}", flush=True)


if __name__ == "__main__":
    main()
