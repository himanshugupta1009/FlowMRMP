#!/usr/bin/env python3
"""Audit the self-contained Franka edge-bundle HDF5 before training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
if str(MRMP_SRC) not in sys.path:
    sys.path.insert(0, str(MRMP_SRC))

from Agents.FrankaPanda import FrankaSelfCollisionChecker  # noqa: E402


DEFAULT_DATASET = (
    ROOT_DIR
    / "data"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5"
)
DEFAULT_URDF = ROOT_DIR / "assets" / "robots" / "panda" / "panda.urdf"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


def finite_max_abs(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all():
        return float("inf")
    return float(np.max(np.abs(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--neighbor-samples", type=int, default=5_000)
    parser.add_argument("--collision-samples", type=int, default=5_000)
    parser.add_argument(
        "--rollout-collision-samples",
        type=int,
        default=5_000,
        help="Number of complete selected edge rollouts to independently self-collision check.",
    )
    parser.add_argument("--seed", type=int, default=8841)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--expected-candidate-pool",
        type=int,
        default=None,
        help=(
            "Optionally require an exact candidate-pool value. The audit always "
            "checks the stored value against the radius analysis."
        ),
    )
    parser.add_argument(
        "--expected-num-queries",
        type=int,
        default=None,
        help="Optionally require an exact total bundle/query count.",
    )
    parser.add_argument(
        "--require-full-edge-validity",
        action="store_true",
        help=(
            "Require the dataset metadata and independent audit to prove that "
            "every selected complete edge was validity-filtered."
        ),
    )
    return parser.parse_args()


def percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def normalize(q: np.ndarray, dq: np.ndarray, metadata: dict) -> np.ndarray:
    norm = metadata["normalization"]
    lower = np.asarray(norm["q_lower"], dtype=np.float32)
    upper = np.asarray(norm["q_upper"], dtype=np.float32)
    dq_max = np.asarray(norm["dq_max_abs"], dtype=np.float32)
    return np.concatenate(
        (2.0 * (q - lower) / (upper - lower) - 1.0, dq / dq_max), axis=-1
    ).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []
    dataset_path = args.dataset.resolve()
    with h5py.File(dataset_path, "r") as source:
        if source.attrs.get("format_name") != "franka_variable_length_edge_bundle":
            failures.append("unexpected format_name")
        format_version = int(source.attrs.get("format_version", -1))
        if format_version != 4:
            failures.append(f"unexpected format_version {format_version}")
        if not bool(source.attrs.get("self_contained", False)):
            failures.append("dataset is not marked self-contained")
        link_counts = {"hard": 0, "soft": 0, "external": 0}

        def count_link(_name: str, link: h5py.HardLink) -> None:
            if isinstance(link, h5py.ExternalLink):
                link_counts["external"] += 1
            elif isinstance(link, h5py.SoftLink):
                link_counts["soft"] += 1
            else:
                link_counts["hard"] += 1

        source.visititems_links(count_link)
        if link_counts["external"] or link_counts["soft"]:
            failures.append(
                "self-contained file contains external or soft HDF5 links"
            )
        external_storage = {"virtual_datasets": 0, "external_raw_datasets": 0}

        def inspect_dataset_storage(_name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.is_virtual:
                external_storage["virtual_datasets"] += 1
            if obj.external:
                external_storage["external_raw_datasets"] += 1

        source.visititems(inspect_dataset_storage)
        if any(external_storage.values()):
            failures.append(
                "self-contained file uses a virtual dataset or external raw storage"
            )
        metadata_value = source["metadata_json"][()]
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        metadata = json.loads(str(metadata_value))
        required_metadata = {
            "set_size": 32,
            "max_edge_steps": 50,
            "cond_dim": 14,
            "query_sampling_mode": "uniform-valid",
            "radius_was_measured_median": True,
            "stored_set_order": "ascending raw edge ID",
        }
        if args.expected_num_queries is not None:
            required_metadata["num_samples"] = int(args.expected_num_queries)
        full_edge_metadata = metadata.get("full_edge_validity", {})
        full_edge_filter_declared = bool(
            full_edge_metadata.get("required_for_every_selected_edge", False)
        )
        if args.require_full_edge_validity and not full_edge_filter_declared:
            failures.append("dataset does not declare complete selected-edge validity")
        candidate_pool_value = metadata.get("candidate_pool")
        candidate_pool = (
            int(candidate_pool_value)
            if isinstance(candidate_pool_value, int)
            and not isinstance(candidate_pool_value, bool)
            else -1
        )
        set_size = int(metadata.get("set_size", -1))
        if candidate_pool < set_size or set_size <= 0:
            failures.append(
                "candidate_pool must be an integer greater than or equal to set_size"
            )
        analyzed_neighbor_rank = metadata.get("radius_analysis", {}).get(
            "neighbor_rank"
        )
        if analyzed_neighbor_rank != candidate_pool:
            failures.append(
                "candidate_pool does not match radius_analysis.neighbor_rank"
            )
        if (
            args.expected_candidate_pool is not None
            and candidate_pool != args.expected_candidate_pool
        ):
            failures.append(
                "candidate_pool "
                f"{candidate_pool} does not equal requested "
                f"{args.expected_candidate_pool}"
            )
        metadata_mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required_metadata.items()
            if metadata.get(key) != expected
        }
        if metadata_mismatches:
            failures.append(
                f"required pipeline metadata differs: {metadata_mismatches}"
            )
        train_cond = source["conds_train"][:].astype(np.float32, copy=False)
        val_cond = source["conds_val"][:].astype(np.float32, copy=False)
        train_raw = source["conds_raw_train"][:].astype(np.float32, copy=False)
        val_raw = source["conds_raw_val"][:].astype(np.float32, copy=False)
        train_ids = source["source_edge_ids_train"][:]
        val_ids = source["source_edge_ids_val"][:]
        conditions = np.concatenate((train_cond, val_cond), axis=0)
        raw_conditions = np.concatenate((train_raw, val_raw), axis=0)
        bundle_ids = np.concatenate((train_ids, val_ids), axis=0)
        raw_group = source["raw_edges"]
        raw_q = raw_group["start_q"][:].astype(np.float32, copy=False)
        raw_dq = raw_group["start_dq"][:].astype(np.float32, copy=False)
        raw_num_steps = raw_group["num_steps"][:]
        raw_length = raw_group["trajectory_length"][:]
        raw_duration = raw_group["duration"][:]
        raw_trajectory = raw_group["trajectory_index"][:]
        raw_start = raw_group["start_index"][:]
        trajectory_names = source["source_trajectories/trajectory_names"][:]
        raw_core_values_finite = bool(
            np.isfinite(raw_q).all()
            and np.isfinite(raw_dq).all()
            and np.isfinite(raw_duration).all()
        )
        if not raw_core_values_finite:
            failures.append("raw edge start/duration arrays contain non-finite values")
        decoded_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in trajectory_names
        ]
        source_lengths = np.asarray(
            [
                source["source_trajectories"][name]["positions"].shape[0]
                for name in decoded_names
            ],
            dtype=np.int32,
        )
        trajectory_offsets = np.zeros(len(decoded_names), dtype=np.int64)
        if len(decoded_names) > 1:
            trajectory_offsets[1:] = np.cumsum(source_lengths[:-1], dtype=np.int64)
        total_source_states = int(source_lengths.sum(dtype=np.int64))
        source_q = np.empty((total_source_states, 7), dtype=np.float32)
        source_dq = np.empty_like(source_q)
        source_ddq = np.empty_like(source_q)
        source_shapes_valid = True
        source_values_finite = True
        for trajectory_index, (name, length) in enumerate(
            zip(decoded_names, source_lengths)
        ):
            group = source["source_trajectories"][name]
            expected = (int(length), 7)
            if (
                group["positions"].shape != expected
                or group["velocities"].shape != expected
                or group["accelerations"].shape != expected
            ):
                source_shapes_valid = False
                continue
            offset = int(trajectory_offsets[trajectory_index])
            q_values = group["positions"][:].astype(np.float32, copy=False)
            dq_values = group["velocities"][:].astype(np.float32, copy=False)
            acceleration_values = group["accelerations"][:]
            source_values_finite &= bool(
                np.isfinite(q_values).all()
                and np.isfinite(dq_values).all()
                and np.isfinite(acceleration_values).all()
            )
            source_q[offset : offset + int(length)] = q_values
            source_dq[offset : offset + int(length)] = dq_values
            source_ddq[offset : offset + int(length)] = acceleration_values
        if not source_shapes_valid:
            failures.append("a copied source trajectory has an inconsistent array shape")
        if not source_values_finite:
            failures.append("a copied source trajectory contains a non-finite value")

        expected_num_queries = int(metadata.get("num_samples", -1))
        expected_num_train = int(metadata.get("num_train", -1))
        expected_num_val = int(metadata.get("num_val", -1))
        expected_set_size = int(metadata.get("set_size", -1))
        expected_shape = (expected_num_queries, 14)
        if (
            train_cond.shape != (expected_num_train, 14)
            or val_cond.shape != (expected_num_val, 14)
        ):
            failures.append("actual condition split does not match metadata")
        if (
            train_ids.shape != (expected_num_train, expected_set_size)
            or val_ids.shape != (expected_num_val, expected_set_size)
        ):
            failures.append("actual bundle split does not match metadata")
        if conditions.shape != expected_shape:
            failures.append(
                f"condition shape {conditions.shape} does not equal {expected_shape}"
            )
        expected_bundle_shape = (expected_num_queries, expected_set_size)
        if bundle_ids.shape != expected_bundle_shape:
            failures.append(
                f"bundle shape {bundle_ids.shape} does not equal {expected_bundle_shape}"
            )
        if not np.isfinite(conditions).all() or not np.isfinite(raw_conditions).all():
            failures.append("query conditions contain non-finite values")
        if np.any(np.abs(conditions) > 1.0 + 2e-6):
            failures.append("normalized query lies outside Franka limits")
        reconstructed = normalize(raw_conditions[:, :7], raw_conditions[:, 7:], metadata)
        roundtrip_max_error = finite_max_abs(reconstructed - conditions)
        if roundtrip_max_error > 3e-6:
            failures.append("raw/normalized query round trip is inconsistent")

        raw_count = len(raw_q)
        bundle_ids_in_range = bool(
            np.all(bundle_ids >= 0) and np.all(bundle_ids < raw_count)
        )
        if not bundle_ids_in_range:
            failures.append("bundle contains an out-of-range raw-edge ID")
        bundle_order_valid = bool(np.all(bundle_ids[:, 1:] >= bundle_ids[:, :-1]))
        if not bundle_order_valid:
            failures.append("stored bundle members are not in ascending raw-edge-ID order")
        duplicate_rows = int(
            np.sum(np.any(np.diff(np.sort(bundle_ids, axis=1), axis=1) == 0, axis=1))
        )
        if duplicate_rows:
            failures.append(f"{duplicate_rows} bundles contain duplicate members")
        sorted_bundle_ids = np.sort(bundle_ids, axis=1)
        duplicate_member_sets = int(
            len(sorted_bundle_ids)
            - len(np.unique(sorted_bundle_ids, axis=0))
        )
        if duplicate_member_sets:
            failures.append(
                f"{duplicate_member_sets} bundle rows repeat an existing member set"
            )
        duplicate_conditions = int(
            len(conditions) - len(np.unique(conditions, axis=0))
        )
        if duplicate_conditions:
            failures.append(f"{duplicate_conditions} query conditions are exact duplicates")
        if bundle_ids_in_range:
            unique_train_ids = np.unique(train_ids)
            unique_val_ids = np.unique(val_ids)
            val_occurrence_overlap = int(np.isin(val_ids, unique_train_ids).sum())
            unique_val_overlap = int(np.isin(unique_val_ids, unique_train_ids).sum())
            train_trajectories = np.unique(raw_trajectory[unique_train_ids])
            val_trajectories = np.unique(raw_trajectory[unique_val_ids])
            val_trajectory_overlap = int(
                np.isin(val_trajectories, train_trajectories).sum()
            )
            train_validation_overlap = {
                "validation_edge_occurrences": int(val_ids.size),
                "validation_edge_occurrences_also_in_train": val_occurrence_overlap,
                "validation_edge_occurrence_overlap_fraction": float(
                    val_occurrence_overlap / val_ids.size
                ),
                "unique_validation_raw_edge_ids": int(len(unique_val_ids)),
                "unique_validation_raw_edge_ids_also_in_train": unique_val_overlap,
                "unique_validation_raw_edge_overlap_fraction": float(
                    unique_val_overlap / len(unique_val_ids)
                ),
                "validation_contributing_trajectories": int(len(val_trajectories)),
                "validation_contributing_trajectories_also_in_train": val_trajectory_overlap,
                "scope": (
                    "Validation is a held-out bundle split over a shared raw-edge and "
                    "source-trajectory library, not an unseen-trajectory split."
                ),
            }
        else:
            train_validation_overlap = {"scope": "not evaluated: invalid raw-edge IDs"}
        max_edge_steps = int(metadata.get("max_edge_steps", -1))
        if (
            raw_count == 0
            or raw_num_steps.min() < 1
            or raw_num_steps.max() > max_edge_steps
            or not np.array_equal(raw_length, raw_num_steps + 1)
        ):
            failures.append("a raw suffix has no executable acceleration interval")
        duration_error_max = finite_max_abs(
            raw_duration - raw_num_steps.astype(np.float64) * float(metadata["dt"])
        )
        if duration_error_max > 1e-5:
            failures.append("raw duration does not equal num_steps * dt")
        trajectory_indices_valid = bool(
            raw_count
            and raw_trajectory.min() >= 0
            and raw_trajectory.max() < len(trajectory_names)
        )
        if not trajectory_indices_valid:
            failures.append("raw trajectory index is out of range")
        start_indices_nonnegative = bool(np.all(raw_start >= 0))
        if not start_indices_nonnegative:
            failures.append("raw start index is negative")
        suffix_indices_valid = trajectory_indices_valid and start_indices_nonnegative
        if suffix_indices_valid:
            remaining_steps = source_lengths[raw_trajectory] - 1 - raw_start
            expected_num_steps = np.minimum(remaining_steps, max_edge_steps)
            suffix_indices_valid = bool(
                np.all(raw_start < source_lengths[raw_trajectory])
                and np.all(raw_start + raw_num_steps < source_lengths[raw_trajectory])
                and np.array_equal(raw_num_steps, expected_num_steps)
            )
        if not suffix_indices_valid:
            failures.append(
                "raw capped edge length does not resolve into source trajectory"
            )

        raw_library_complete = False
        if trajectory_indices_valid:
            expected_counts = source_lengths.astype(np.int64) - 1
            actual_counts = np.bincount(
                raw_trajectory.astype(np.int64), minlength=len(source_lengths)
            )
            if np.array_equal(actual_counts, expected_counts):
                expected_trajectory = np.repeat(
                    np.arange(len(source_lengths), dtype=raw_trajectory.dtype),
                    expected_counts,
                )
                expected_offsets = np.repeat(
                    np.cumsum(
                        np.concatenate(([0], expected_counts[:-1])),
                        dtype=np.int64,
                    ),
                    expected_counts,
                )
                expected_start = (
                    np.arange(raw_count, dtype=np.int64) - expected_offsets
                )
                raw_library_complete = bool(
                    np.array_equal(raw_trajectory, expected_trajectory)
                    and np.array_equal(raw_start, expected_start)
                )
                del expected_trajectory, expected_offsets, expected_start
        if not raw_library_complete:
            failures.append(
                "raw library is not exactly one suffix start 0..L-2 per source trajectory"
            )

        raw_source_max_errors: dict[str, float | None] = {
            key: 0.0
            for key in (
                "start_q",
                "start_dq",
                "final_q",
                "final_dq",
                "delta_q",
                "delta_dq",
            )
        }
        raw_source_values_finite = raw_core_values_finite
        source_resolution_ready = (
            suffix_indices_valid and source_shapes_valid and source_values_finite
        )
        if source_resolution_ready:
            for offset in range(0, raw_count, 200_000):
                stop = min(offset + 200_000, raw_count)
                trajectory_slice = raw_trajectory[offset:stop]
                start_slice = raw_start[offset:stop]
                start_flat_index = trajectory_offsets[trajectory_slice] + start_slice
                final_flat_index = (
                    trajectory_offsets[trajectory_slice]
                    + start_slice
                    + raw_num_steps[offset:stop]
                )
                expected_start_q = source_q[start_flat_index]
                expected_start_dq = source_dq[start_flat_index]
                expected_final_q = source_q[final_flat_index]
                expected_final_dq = source_dq[final_flat_index]
                stored_final_q = raw_group["final_q"][offset:stop]
                stored_final_dq = raw_group["final_dq"][offset:stop]
                stored_delta_q = raw_group["delta_q"][offset:stop]
                stored_delta_dq = raw_group["delta_dq"][offset:stop]
                raw_source_values_finite &= bool(
                    np.isfinite(stored_final_q).all()
                    and np.isfinite(stored_final_dq).all()
                    and np.isfinite(stored_delta_q).all()
                    and np.isfinite(stored_delta_dq).all()
                )
                differences = {
                    "start_q": raw_q[offset:stop] - expected_start_q,
                    "start_dq": raw_dq[offset:stop] - expected_start_dq,
                    "final_q": stored_final_q - expected_final_q,
                    "final_dq": stored_final_dq - expected_final_dq,
                    "delta_q": stored_delta_q
                    - (expected_final_q - expected_start_q),
                    "delta_dq": stored_delta_dq
                    - (expected_final_dq - expected_start_dq),
                }
                for key, difference in differences.items():
                    raw_source_max_errors[key] = max(
                        float(raw_source_max_errors[key]), finite_max_abs(difference)
                    )
        else:
            raw_source_max_errors = {key: None for key in raw_source_max_errors}
        if not raw_source_values_finite:
            failures.append("raw edge outcome arrays contain non-finite values")
        if (
            not source_resolution_ready
            or any(value is None or value > 2e-6 for value in raw_source_max_errors.values())
        ):
            failures.append(
                "raw start/final/outcome arrays do not resolve exactly into copied trajectories"
            )

        bundle_outcome_max_errors = {
            key: 0.0
            for key in (
                "num_steps",
                "duration",
                "delta_q",
                "delta_dq",
                "final_q",
                "final_dq",
            )
        }
        bundle_outcome_values_finite = True
        selected_edge_count = 0
        selected_dynamic_invalid_edges = 0
        selected_acceleration_invalid_edges = 0
        selected_jerk_invalid_edges = 0
        selected_position_invalid_edges = 0
        selected_velocity_invalid_edges = 0
        if "bundle_outcomes" not in source:
            failures.append("bundle_outcomes group is missing")
            bundle_outcome_max_errors = {
                key: None for key in bundle_outcome_max_errors
            }
            bundle_outcome_values_finite = False
        else:
            outcome_group = source["bundle_outcomes"]
            for split, split_raw_conditions, split_ids in (
                ("train", train_raw, train_ids),
                ("val", val_raw, val_ids),
            ):
                expected_scalar_shape = split_ids.shape
                expected_vector_shape = (*split_ids.shape, 7)
                for field in ("num_steps", "duration"):
                    if outcome_group[f"{field}_{split}"].shape != expected_scalar_shape:
                        failures.append(
                            f"bundle_outcomes/{field}_{split} has wrong shape"
                        )
                for field in ("delta_q", "delta_dq", "final_q", "final_dq"):
                    if outcome_group[f"{field}_{split}"].shape != expected_vector_shape:
                        failures.append(
                            f"bundle_outcomes/{field}_{split} has wrong shape"
                        )

                for offset in range(0, len(split_ids), 256):
                    stop = min(offset + 256, len(split_ids))
                    ids = split_ids[offset:stop]
                    stored_counts = outcome_group[f"num_steps_{split}"][
                        offset:stop
                    ]
                    stored_duration = outcome_group[f"duration_{split}"][
                        offset:stop
                    ]
                    stored = {
                        field: outcome_group[f"{field}_{split}"][offset:stop]
                        for field in ("delta_q", "delta_dq", "final_q", "final_dq")
                    }
                    bundle_outcome_values_finite &= bool(
                        np.isfinite(stored_duration).all()
                        and all(np.isfinite(value).all() for value in stored.values())
                    )
                    expected_counts = raw_num_steps[ids]
                    bundle_outcome_max_errors["num_steps"] = max(
                        float(bundle_outcome_max_errors["num_steps"]),
                        finite_max_abs(stored_counts - expected_counts),
                    )
                    bundle_outcome_max_errors["duration"] = max(
                        float(bundle_outcome_max_errors["duration"]),
                        finite_max_abs(
                            stored_duration - expected_counts * float(metadata["dt"])
                        ),
                    )

                    flat_starts = (
                        trajectory_offsets[raw_trajectory[ids]] + raw_start[ids]
                    )
                    steps = np.arange(max_edge_steps, dtype=np.int64)[None, None, :]
                    mask = steps < expected_counts[:, :, None]
                    safe_steps = np.minimum(
                        steps, expected_counts[:, :, None] - 1
                    )
                    actions = source_ddq[
                        flat_starts[:, :, None] + safe_steps
                    ].astype(np.float64)
                    actions *= mask[:, :, :, None]
                    selected_edge_count += int(ids.size)
                    acceleration_valid = np.all(
                        ~mask[:, :, :, None]
                        | (
                            np.abs(actions)
                            <= np.asarray(
                                metadata["normalization"]["ddq_max_abs"],
                                dtype=np.float64,
                            )[None, None, None, :]
                            + 1e-10
                        ),
                        axis=(2, 3),
                    )
                    if max_edge_steps > 1:
                        jerk_mask = (
                            steps[:, :, 1:] < expected_counts[:, :, None]
                        )
                        jerk = np.diff(actions, axis=2) / float(metadata["dt"])
                        jerk_valid = np.all(
                            ~jerk_mask[:, :, :, None]
                            | (
                                np.abs(jerk)
                                <= np.asarray(
                                    metadata["normalization"]["dddq_max_abs"],
                                    dtype=np.float64,
                                )[None, None, None, :]
                                + 1e-10
                            ),
                            axis=(2, 3),
                        )
                    else:
                        jerk_valid = np.ones(ids.shape, dtype=bool)
                    cumulative = np.cumsum(actions, axis=2)
                    delta_dq_path = float(metadata["dt"]) * cumulative
                    query_dq = split_raw_conditions[offset:stop, 7:].astype(
                        np.float64
                    )
                    position_increment = (
                        query_dq[:, None, None, :] * float(metadata["dt"])
                        + float(metadata["dt"]) ** 2
                        * (cumulative - 0.5 * actions)
                    )
                    delta_q_path = np.cumsum(position_increment, axis=2)
                    query_q = split_raw_conditions[offset:stop, :7].astype(
                        np.float64
                    )
                    q_path = query_q[:, None, None, :] + delta_q_path
                    dq_path = query_dq[:, None, None, :] + delta_dq_path
                    q_lower = np.asarray(
                        metadata["normalization"]["q_lower"], dtype=np.float64
                    )
                    q_upper = np.asarray(
                        metadata["normalization"]["q_upper"], dtype=np.float64
                    )
                    dq_max = np.asarray(
                        metadata["normalization"]["dq_max_abs"], dtype=np.float64
                    )
                    position_valid = np.all(
                        ~mask[:, :, :, None]
                        | (
                            (q_path >= q_lower[None, None, None, :] - 1e-10)
                            & (q_path <= q_upper[None, None, None, :] + 1e-10)
                        ),
                        axis=(2, 3),
                    )
                    velocity_valid = np.all(
                        ~mask[:, :, :, None]
                        | (
                            np.abs(dq_path)
                            <= dq_max[None, None, None, :] + 1e-10
                        ),
                        axis=(2, 3),
                    )
                    finite_valid = np.all(
                        ~mask[:, :, :, None]
                        | (np.isfinite(q_path) & np.isfinite(dq_path)),
                        axis=(2, 3),
                    )
                    dynamic_valid = (
                        acceleration_valid
                        & jerk_valid
                        & position_valid
                        & velocity_valid
                        & finite_valid
                    )
                    selected_acceleration_invalid_edges += int(
                        (~acceleration_valid).sum()
                    )
                    selected_jerk_invalid_edges += int((~jerk_valid).sum())
                    selected_position_invalid_edges += int(
                        (~position_valid).sum()
                    )
                    selected_velocity_invalid_edges += int(
                        (~velocity_valid).sum()
                    )
                    selected_dynamic_invalid_edges += int(
                        (~dynamic_valid).sum()
                    )
                    batch_indices = np.arange(stop - offset)[:, None]
                    edge_indices = np.arange(ids.shape[1])[None, :]
                    final_indices = expected_counts - 1
                    expected_delta_q = delta_q_path[
                        batch_indices, edge_indices, final_indices
                    ]
                    expected_delta_dq = delta_dq_path[
                        batch_indices, edge_indices, final_indices
                    ]
                    expected_values = {
                        "delta_q": expected_delta_q,
                        "delta_dq": expected_delta_dq,
                        "final_q": query_q[:, None, :] + expected_delta_q,
                        "final_dq": query_dq[:, None, :] + expected_delta_dq,
                    }
                    for field, expected_value in expected_values.items():
                        bundle_outcome_max_errors[field] = max(
                            float(bundle_outcome_max_errors[field]),
                            finite_max_abs(stored[field] - expected_value),
                        )
        if not bundle_outcome_values_finite:
            failures.append("bundle outcome arrays contain non-finite values")
        if any(
            value is None or value > 3e-6
            for value in bundle_outcome_max_errors.values()
        ):
            failures.append(
                "bundle outcomes do not match float64 integration from query states"
            )
        if args.require_full_edge_validity and selected_dynamic_invalid_edges:
            failures.append(
                f"{selected_dynamic_invalid_edges} selected complete edges violate "
                "acceleration, jerk, position, or velocity validity"
            )

        radius = float(metadata.get("radius", float("nan")))
        analyzed_median_radius = float(
            metadata.get("radius_analysis", {}).get("median_radius", float("nan"))
        )
        radius_valid = bool(
            np.isfinite(radius)
            and radius > 0.0
            and np.isclose(radius, analyzed_median_radius, rtol=0.0, atol=1e-12)
            and metadata.get("radius_was_measured_median") is True
        )
        if not radius_valid:
            failures.append("fixed radius is not a finite positive measured median")

        rollout_collision_sample_count = min(
            max(0, args.rollout_collision_samples), int(bundle_ids.size)
        )
        rollout_collision_edges = 0
        rollout_collision_waypoints_checked = 0
        if (
            rollout_collision_sample_count
            and bundle_ids_in_range
            and source_resolution_ready
        ):
            occurrence_indices = rng.choice(
                bundle_ids.size,
                size=rollout_collision_sample_count,
                replace=False,
            )
            bundle_row_indices = occurrence_indices // bundle_ids.shape[1]
            bundle_slot_indices = occurrence_indices % bundle_ids.shape[1]
            sampled_edge_ids = bundle_ids[
                bundle_row_indices, bundle_slot_indices
            ]
            checker = FrankaSelfCollisionChecker(args.urdf)
            try:
                dt = float(metadata["dt"])
                for bundle_row, edge_id in zip(
                    bundle_row_indices, sampled_edge_ids
                ):
                    edge_id = int(edge_id)
                    count = int(raw_num_steps[edge_id])
                    flat_start = int(
                        trajectory_offsets[raw_trajectory[edge_id]]
                        + raw_start[edge_id]
                    )
                    actions = source_ddq[flat_start : flat_start + count].astype(
                        np.float64, copy=False
                    )
                    cumulative = np.cumsum(actions, axis=0)
                    query_state = raw_conditions[int(bundle_row)].astype(
                        np.float64, copy=False
                    )
                    q_path = query_state[None, :7] + np.cumsum(
                        query_state[None, 7:] * dt
                        + dt * dt * (cumulative - 0.5 * actions),
                        axis=0,
                    )
                    collided = False
                    for configuration in q_path:
                        rollout_collision_waypoints_checked += 1
                        if checker.in_collision(configuration):
                            collided = True
                            break
                    rollout_collision_edges += int(collided)
            finally:
                checker.close()
        if args.require_full_edge_validity and rollout_collision_edges:
            failures.append(
                f"{rollout_collision_edges}/{rollout_collision_sample_count} "
                "sampled selected complete edges self-collide"
            )

        raw_normalized = normalize(raw_q, raw_dq, metadata).astype(
            np.float64, copy=False
        )
        del source_q, source_dq, source_ddq, raw_q, raw_dq, sorted_bundle_ids
        gc.collect()
        locality_max: float | None = None
        locality_violations: int | None = None
        if bundle_ids_in_range and radius_valid and np.isfinite(raw_normalized).all():
            locality_max = 0.0
            locality_violations = 0
            for offset in range(0, len(bundle_ids), 1_000):
                query = conditions[offset : offset + 1_000]
                selected = raw_normalized[bundle_ids[offset : offset + 1_000]]
                distance = np.linalg.norm(selected - query[:, None, :], axis=-1)
                locality_max = max(locality_max, float(distance.max()))
                locality_violations += int(np.sum(distance > radius + 2e-6))
            if locality_violations:
                failures.append(
                    f"{locality_violations} selected edges lie outside the fixed radius"
                )
        else:
            failures.append("selected-edge locality could not be evaluated safely")

        neighbor_counts = np.empty(0, dtype=np.int64)
        nearest_distance = np.empty(0, dtype=np.float64)
        conditions_finite = bool(np.isfinite(conditions).all())
        if radius_valid and conditions_finite and np.isfinite(raw_normalized).all():
            tree = cKDTree(raw_normalized, copy_data=False)
            neighbor_sample_count = min(args.neighbor_samples, len(conditions))
            neighbor_indices = rng.choice(
                len(conditions), size=neighbor_sample_count, replace=False
            )
            neighbor_counts = tree.query_ball_point(
                conditions[neighbor_indices],
                r=radius,
                return_length=True,
                workers=-1,
            )
            if np.any(neighbor_counts < candidate_pool):
                failures.append(
                    "a sampled accepted query has fewer than "
                    f"{candidate_pool} radius neighbors"
                )
            nearest_distance, _ = tree.query(
                conditions[neighbor_indices], k=1, workers=-1
            )
        else:
            failures.append("sampled neighbor checks could not be evaluated safely")

    collision_sample_count = min(args.collision_samples, len(raw_conditions))
    collision_indices = rng.choice(
        len(raw_conditions), size=collision_sample_count, replace=False
    )
    checker = FrankaSelfCollisionChecker(args.urdf)
    collision_count = 0
    try:
        for index in collision_indices:
            collision_count += int(checker.in_collision(raw_conditions[index, :7]))
    finally:
        checker.close()
    if collision_count:
        failures.append(
            f"{collision_count}/{collision_sample_count} sampled queries self-collide"
        )

    summary = {
        "dataset": portable_path(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": file_sha256(dataset_path),
        "format_version": format_version,
        "hdf5_link_counts": link_counts,
        "hdf5_external_storage_counts": external_storage,
        "required_pipeline_metadata_mismatches": metadata_mismatches,
        "all_checks_passed": not failures,
        "failures": failures,
        "num_queries": int(len(conditions)),
        "num_train": int(len(train_cond)),
        "num_validation": int(len(val_cond)),
        "set_size": int(bundle_ids.shape[1]),
        "candidate_pool": candidate_pool,
        "max_edge_steps": max_edge_steps,
        "raw_num_steps": percentiles(raw_num_steps),
        "raw_edge_count": int(raw_count),
        "source_trajectory_count": int(len(trajectory_names)),
        "query_sampling_mode": metadata.get("query_sampling_mode"),
        "fixed_radius": radius,
        "selected_edge_max_distance": locality_max,
        "selected_edge_locality_violations": locality_violations,
        "query_roundtrip_max_abs_error": roundtrip_max_error,
        "bundles_with_duplicate_member_ids": duplicate_rows,
        "duplicate_bundle_member_sets": duplicate_member_sets,
        "duplicate_query_conditions": duplicate_conditions,
        "train_validation_overlap": train_validation_overlap,
        "stored_bundle_order_valid": bundle_order_valid,
        "raw_library_complete": raw_library_complete,
        "duration_max_abs_error_seconds": duration_error_max,
        "raw_source_resolution_max_abs_errors": raw_source_max_errors,
        "bundle_outcome_resolution_max_abs_errors": bundle_outcome_max_errors,
        "bundle_outcome_values_finite": bundle_outcome_values_finite,
        "full_edge_filter_declared": full_edge_filter_declared,
        "selected_complete_edge_dynamic_validity": {
            "edges_checked": int(selected_edge_count),
            "invalid_edges": int(selected_dynamic_invalid_edges),
            "acceleration_invalid_edges": int(
                selected_acceleration_invalid_edges
            ),
            "jerk_invalid_edges": int(selected_jerk_invalid_edges),
            "position_invalid_edges": int(selected_position_invalid_edges),
            "velocity_invalid_edges": int(selected_velocity_invalid_edges),
        },
        "sampled_selected_complete_edge_self_collision": {
            "edges_checked": int(rollout_collision_sample_count),
            "colliding_edges": int(rollout_collision_edges),
            "waypoints_checked": int(rollout_collision_waypoints_checked),
        },
        "raw_edge_values_finite": raw_source_values_finite,
        "source_trajectory_shapes_valid": source_shapes_valid,
        "source_trajectory_values_finite": source_values_finite,
        "sampled_neighbor_count": percentiles(neighbor_counts),
        "sampled_nearest_raw_start_distance": percentiles(nearest_distance),
        "sampled_query_self_collisions": collision_count,
        "sampled_query_self_collision_sample_size": collision_sample_count,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": [
            (
                "HDF5 v4 schema, metadata-consistent train/validation/k32/max50 "
                "pipeline dimensions, "
                f"internally consistent pool{candidate_pool}, and no "
                "external/virtual storage"
            ),
            "finite normalized and raw 14D query states within limits",
            "raw/normalized condition round trip",
            "32 unique sorted in-range raw-edge references per bundle; no repeated member sets or conditions",
            "every selected member lies within the fixed radius",
            (
                "sampled accepted queries have at least "
                f"{candidate_pool} radius neighbors"
            ),
            "complete capped-edge library plus finite N/duration/start/final/relative-outcome resolution into copied source trajectories",
            "bundle-specific relative and exact outcomes match float64 integration from each query condition",
            "sampled query configurations are self-collision free",
            (
                "every selected complete edge satisfies acceleration, intra-edge "
                "jerk, joint-position, and joint-velocity limits"
            ),
            "sampled selected complete edge rollouts are self-collision free",
        ],
    }
    output_path = args.output or dataset_path.with_suffix(".audit.json")
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"audit saved to {output_path}")
    if failures:
        raise RuntimeError(f"Dataset audit failed with {len(failures)} issue(s)")


if __name__ == "__main__":
    main()
