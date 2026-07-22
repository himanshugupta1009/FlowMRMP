#!/usr/bin/env python3
"""Create a self-contained Franka edge-bundle dataset from CuRobo trajectories.

Every eligible trajectory timestep is a raw edge start.  Its edge is the full
variable-length suffix from that timestep to the end of the source trajectory.
Bundles are conditioned on normalized (q, dq), use a fixed-radius neighborhood,
cap the candidate pool at 256, and select 32 diverse suffixes with FPS.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "dataset50k.h5"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "franka_edge_bundle_50k_k32_n100000.h5"

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
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--set-size", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=256)
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Normalized 14D radius. Default: median 256th-neighbor distance.",
    )
    parser.add_argument("--radius-samples", type=int, default=20_000)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def normalize_conditions(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    q_norm = 2.0 * (q - Q_LOWER) / (Q_UPPER - Q_LOWER) - 1.0
    dq_norm = dq / DQ_MAX
    return np.concatenate((q_norm, dq_norm), axis=-1).astype(np.float32)


def load_raw_edge_index(source: h5py.File) -> dict[str, np.ndarray | list[str] | float]:
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
    final_q = np.empty((total, 7), dtype=np.float32)
    final_dq = np.empty((total, 7), dtype=np.float32)
    trajectory_index = np.empty(total, dtype=np.int32)
    start_index = np.empty(total, dtype=np.int32)
    trajectory_length = np.empty(total, dtype=np.int32)
    duration = np.empty(total, dtype=np.float32)

    dt_values: list[float] = []
    cursor = 0
    for traj_idx, name in enumerate(names):
        group = source[name]
        q = group["positions"][:].astype(np.float32, copy=False)
        dq = group["velocities"][:].astype(np.float32, copy=False)
        edge_count = max(0, q.shape[0] - 1)
        if edge_count == 0:
            continue

        dt = float(group.attrs.get("dt", source.attrs.get("interpolation_dt", 0.02)))
        dt_values.append(dt)
        sl = slice(cursor, cursor + edge_count)
        indices = np.arange(edge_count, dtype=np.int32)
        lengths = q.shape[0] - indices

        start_q[sl] = q[:-1]
        start_dq[sl] = dq[:-1]
        delta_q[sl] = q[-1] - q[:-1]
        final_q[sl] = q[-1]
        final_dq[sl] = dq[-1]
        trajectory_index[sl] = traj_idx
        start_index[sl] = indices
        trajectory_length[sl] = lengths
        duration[sl] = (lengths - 1) * dt
        cursor += edge_count

    unique_dt = np.unique(np.asarray(dt_values, dtype=np.float64))
    if unique_dt.size != 1:
        raise ValueError(f"Expected one shared dt; found {unique_dt.tolist()}")

    conditions_norm = normalize_conditions(start_q, start_dq)
    max_duration = float(duration.max())
    diversity_features = np.concatenate(
        (
            (duration / max_duration)[:, None] * 0.5,
            delta_q / (Q_UPPER - Q_LOWER),
            final_dq / DQ_MAX,
        ),
        axis=1,
    ).astype(np.float32)

    return {
        "trajectory_names": names,
        "trajectory_index": trajectory_index,
        "start_index": start_index,
        "trajectory_length": trajectory_length,
        "duration": duration,
        "start_q": start_q,
        "start_dq": start_dq,
        "delta_q": delta_q,
        "final_q": final_q,
        "final_dq": final_dq,
        "conditions_norm": conditions_norm,
        "diversity_features": diversity_features,
        "dt": float(unique_dt[0]),
        "max_duration": max_duration,
    }


def measure_radius(
    tree: cKDTree,
    conditions: np.ndarray,
    *,
    candidate_pool: int,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[float, dict[str, object]]:
    sample_count = min(sample_count, conditions.shape[0])
    query_ids = rng.choice(conditions.shape[0], size=sample_count, replace=False)
    distances, _ = tree.query(
        conditions[query_ids], k=candidate_pool, workers=-1
    )
    kth_distances = np.asarray(distances)[:, -1]
    percentiles = {
        str(p): float(np.percentile(kth_distances, p))
        for p in (5, 25, 50, 75, 90, 95, 99)
    }
    radius = float(np.median(kth_distances))
    counts = tree.query_ball_point(
        conditions[query_ids], r=radius, return_length=True, workers=-1
    )
    report = {
        "sample_count": int(sample_count),
        "neighbor_rank": int(candidate_pool),
        "distance_percentiles": percentiles,
        "median_radius": radius,
        "coverage_at_least_set_size": None,
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


def build_bundles(
    tree: cKDTree,
    conditions: np.ndarray,
    diversity_features: np.ndarray,
    *,
    num_samples: int,
    candidate_pool: int,
    set_size: int,
    radius: float,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if candidate_pool < set_size:
        raise ValueError("candidate_pool must be at least set_size")

    query_order = rng.permutation(conditions.shape[0])
    query_edge_ids = np.empty(num_samples, dtype=np.int64)
    bundle_edge_ids = np.empty((num_samples, set_size), dtype=np.int64)
    accepted = 0
    attempted = 0
    candidate_counts: list[int] = []
    started = time.time()

    for offset in range(0, query_order.size, batch_size):
        batch_ids = query_order[offset : offset + batch_size]
        hit_lists = tree.query_ball_point(
            conditions[batch_ids], r=radius, workers=-1
        )
        for query_id, hits in zip(batch_ids, hit_lists):
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

            local_ids = farthest_point_sample(
                diversity_features[candidate_ids], set_size, rng
            )
            chosen = candidate_ids[local_ids]
            # Deterministic order for storage. Training can later choose its own set loss.
            chosen.sort()
            query_edge_ids[accepted] = query_id
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
                    "candidate_count_before_cap_percentiles": {
                        str(p): float(np.percentile(candidate_counts, p))
                        for p in (5, 50, 95)
                    },
                    "elapsed_seconds": float(time.time() - started),
                }
                return query_edge_ids, bundle_edge_ids, stats

    raise RuntimeError(
        f"Only found {accepted:,} eligible queries among {attempted:,} unique starts."
    )


def _create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    kwargs: dict[str, object] = {}
    if data.ndim > 0 and data.size > 0:
        kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=data, **kwargs)


def write_output(
    output_path: Path,
    source: h5py.File,
    raw: dict[str, object],
    query_ids: np.ndarray,
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

    sample_perm = rng.permutation(query_ids.shape[0])
    num_val = int(round(query_ids.shape[0] * val_fraction))
    val_idx = sample_perm[:num_val]
    train_idx = sample_perm[num_val:]
    conditions = raw["conditions_norm"]
    start_q = raw["start_q"]
    start_dq = raw["start_dq"]
    assert isinstance(conditions, np.ndarray)
    assert isinstance(start_q, np.ndarray)
    assert isinstance(start_dq, np.ndarray)

    with h5py.File(temp_path, "w") as output:
        output.attrs["format_name"] = "franka_variable_length_edge_bundle"
        output.attrs["format_version"] = 1
        output.attrs["self_contained"] = True
        output.attrs["created_unix_time"] = time.time()

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
            "trajectory_length",
            "duration",
            "start_q",
            "start_dq",
            "delta_q",
            "final_q",
            "final_dq",
        ):
            value = raw[field]
            assert isinstance(value, np.ndarray)
            _create_dataset(raw_group, field, value)
        raw_group.attrs["variable_length_resolution"] = (
            "For raw edge e: name=source_trajectories/trajectory_names["
            "raw_edges/trajectory_index[e]]; start=raw_edges/start_index[e]; "
            "q, dq, ddq are source_trajectories/{name}/positions[start:], "
            "velocities[start:], accelerations[start:]."
        )

        for split, indices in (("train", train_idx), ("val", val_idx)):
            _create_dataset(output, f"conds_{split}", conditions[query_ids[indices]])
            raw_conditions = np.concatenate(
                (start_q[query_ids[indices]], start_dq[query_ids[indices]]), axis=1
            )
            _create_dataset(output, f"conds_raw_{split}", raw_conditions)
            _create_dataset(
                output, f"query_source_edge_ids_{split}", query_ids[indices]
            )
            _create_dataset(
                output, f"source_edge_ids_{split}", bundle_ids[indices]
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

    rng = np.random.default_rng(args.seed)
    total_started = time.time()
    with h5py.File(args.input, "r") as source:
        print(f"loading raw-edge index from {args.input}", flush=True)
        raw = load_raw_edge_index(source)
        conditions = raw["conditions_norm"]
        diversity = raw["diversity_features"]
        assert isinstance(conditions, np.ndarray)
        assert isinstance(diversity, np.ndarray)
        print(f"eligible raw edges: {conditions.shape[0]:,}", flush=True)
        print("building 14D condition KD-tree", flush=True)
        tree = cKDTree(conditions)

        measured_radius, radius_report = measure_radius(
            tree,
            conditions,
            candidate_pool=args.candidate_pool,
            sample_count=args.radius_samples,
            rng=rng,
        )
        radius_report["coverage_at_least_set_size"] = float(
            np.mean(
                tree.query_ball_point(
                    conditions[
                        rng.choice(
                            conditions.shape[0],
                            size=min(args.radius_samples, conditions.shape[0]),
                            replace=False,
                        )
                    ],
                    r=measured_radius,
                    return_length=True,
                    workers=-1,
                )
                >= args.set_size
            )
        )
        radius = measured_radius if args.radius is None else float(args.radius)
        print(json.dumps(radius_report, indent=2, sort_keys=True), flush=True)
        print(f"using radius: {radius:.12f}", flush=True)

        query_ids, bundle_ids, bundle_stats = build_bundles(
            tree,
            conditions,
            diversity,
            num_samples=args.num_samples,
            candidate_pool=args.candidate_pool,
            set_size=args.set_size,
            radius=radius,
            batch_size=args.query_batch_size,
            rng=rng,
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
            "edge_length": "variable; full source-trajectory suffix",
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
                "unique raw-edge start states from dataset support; accept only when "
                "at least candidate_pool starts fall within the fixed radius"
            ),
            "candidate_selection": (
                "all raw-edge starts within radius, uniformly cap at candidate_pool"
            ),
            "fps_features": [
                "0.5 * duration / max_duration",
                "delta_q / (q_upper - q_lower) [7]",
                "final_dq / dq_max_abs [7]",
            ],
            "set_selection": "farthest_point_sample_k32",
            "stored_set_order": "ascending raw edge ID",
            "duration_policy": "full remaining suffix; duration bias intentionally retained",
            "additional_trajectory_filtering": "none",
            "bundle_generation": bundle_stats,
            "seed": int(args.seed),
        }
        write_output(
            args.output,
            source,
            raw,
            query_ids,
            bundle_ids,
            val_fraction=args.val_fraction,
            metadata=metadata,
            rng=rng,
        )

    elapsed = time.time() - total_started
    print(f"saved {args.output}", flush=True)
    print(f"total elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
