#!/usr/bin/env python3
"""Audit a target-conditioned Franka companion HDF5 against its source.

The audit verifies source identity, the 29D condition layout, goal-mode
semantics, reachable target anchors, complete edge permutations, and the
target-nearest prefix. It writes a JSON report and exits nonzero on failure.
No HDF5 data is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a potentially large file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Define the read-only audit command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Target-conditioned companion HDF5.")
    parser.add_argument("--output", type=Path, default=None, help="Audit JSON path.")
    parser.add_argument(
        "--chunk-size", type=int, default=4096, help="Rows checked per chunk."
    )
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Skip the expensive source hash only for disposable smoke data.",
    )
    return parser.parse_args()


def decode_json(dataset: h5py.Dataset) -> dict[str, object]:
    """Decode one scalar UTF-8 JSON HDF5 dataset."""
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def resolve_source(dataset_path: Path, target: h5py.File) -> Path:
    """Find the immutable source HDF5 beside the companion or at its old path."""
    local = dataset_path.parent / str(target.attrs["source_dataset_basename"])
    original = Path(str(target.attrs["source_dataset_original_path"]))
    if local.is_file():
        return local.resolve()
    if original.is_file():
        return original.resolve()
    raise FileNotFoundError(f"Missing source dataset: {local} or {original}")


def main() -> None:
    """Run every audit check, write the JSON report, and signal pass/failure."""
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    dataset_path = args.dataset.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else dataset_path.with_suffix(dataset_path.suffix + ".audit.json")
    )
    errors: list[str] = []
    split_reports = {}
    with h5py.File(dataset_path, "r") as target:
        if target.attrs.get("format_name") != "franka_target_conditioned_edge_bundle":
            raise ValueError(f"Unexpected target dataset format: {dataset_path}")
        metadata = decode_json(target["metadata_json"])
        source_path = resolve_source(dataset_path, target)
        expected_source_hash = metadata.get("source_dataset_sha256")
        actual_source_hash = None
        if not args.skip_source_hash:
            actual_source_hash = file_sha256(source_path)
            if expected_source_hash is None:
                errors.append("target metadata omits source_dataset_sha256")
            elif actual_source_hash != expected_source_hash:
                errors.append("source dataset SHA-256 does not match target metadata")

        with h5py.File(source_path, "r") as source:
            source_metadata = decode_json(source["metadata_json"])
            normalization = source_metadata["normalization"]
            q_range = (
                np.asarray(normalization["q_upper"], dtype=np.float32)
                - np.asarray(normalization["q_lower"], dtype=np.float32)
            )
            dq_max = np.asarray(normalization["dq_max_abs"], dtype=np.float32)
            directed_count = int(metadata["target_directed_count"])
            outcomes = source["bundle_outcomes"]
            for split in ("train", "val"):
                row_count = int(target[f"conds_{split}"].shape[0])
                set_size = int(target[f"edge_order_{split}"].shape[1])
                split_errors = 0
                goal_rows = 0
                max_anchor_distance = 0.0
                for start in range(0, row_count, args.chunk_size):
                    stop = min(start + args.chunk_size, row_count)
                    base_indices = target[f"base_indices_{split}"][start:stop]
                    order = target[f"edge_order_{split}"][start:stop].astype(
                        np.int64
                    )
                    cond = target[f"conds_{split}"][start:stop]
                    relative = target[f"target_relative_normalized_{split}"][
                        start:stop
                    ]
                    goal_mode = target[f"goal_mode_{split}"][start:stop].astype(
                        bool
                    )
                    anchor = target[f"target_anchor_source_slot_{split}"][
                        start:stop
                    ].astype(np.int64)
                    target_raw = target[f"target_states_raw_{split}"][start:stop]
                    goal_rows += int(goal_mode.sum())
                    unique, inverse = np.unique(base_indices, return_inverse=True)
                    source_cond = source[f"conds_{split}"][unique][inverse]
                    source_raw = source[f"conds_raw_{split}"][unique][inverse]
                    delta_q = outcomes[f"delta_q_{split}"][unique][inverse]
                    delta_dq = outcomes[f"delta_dq_{split}"][unique][inverse]
                    rows = np.arange(len(base_indices))
                    target_delta_q = delta_q[rows, anchor]
                    target_delta_dq = delta_dq[rows, anchor]
                    expected_relative = np.concatenate(
                        (
                            2.0 * target_delta_q / q_range[None, :],
                            target_delta_dq / dq_max[None, :],
                        ),
                        axis=1,
                    ).astype(np.float32)
                    expected_relative[goal_mode, 7:] = 0.0
                    expected_target_raw = source_raw.copy()
                    expected_target_raw[:, :7] += target_delta_q
                    expected_target_raw[:, 7:] += target_delta_dq

                    checks = (
                        np.allclose(cond[:, :14], source_cond, atol=1e-6),
                        np.allclose(cond[:, 14:28], relative, atol=1e-6),
                        np.allclose(cond[:, 28], goal_mode.astype(np.float32)),
                        np.allclose(relative, expected_relative, atol=1e-6),
                        np.allclose(target_raw, expected_target_raw, atol=1e-6),
                    )
                    if not all(checks):
                        split_errors += 1

                    sorted_order = np.sort(order, axis=1)
                    expected_slots = np.arange(set_size)[None, :]
                    if not np.array_equal(
                        sorted_order, np.repeat(expected_slots, len(order), axis=0)
                    ):
                        split_errors += 1

                    edge_relative = np.concatenate(
                        (
                            2.0 * delta_q / q_range[None, None, :],
                            delta_dq / dq_max[None, None, :],
                        ),
                        axis=2,
                    )
                    q_distance = np.linalg.norm(
                        edge_relative[:, :, :7] - relative[:, None, :7], axis=2
                    )
                    full_distance = np.linalg.norm(
                        edge_relative - relative[:, None, :], axis=2
                    )
                    distance = np.where(goal_mode[:, None], q_distance, full_distance)
                    expected_nearest = np.argsort(
                        distance, axis=1, kind="stable"
                    )[:, :directed_count]
                    if not np.array_equal(order[:, :directed_count], expected_nearest):
                        split_errors += 1
                    anchor_distance = distance[rows, anchor]
                    max_anchor_distance = max(
                        max_anchor_distance, float(anchor_distance.max(initial=0.0))
                    )
                if split_errors:
                    errors.append(
                        f"{split} failed {split_errors} chunk-level schema checks"
                    )
                split_reports[split] = {
                    "rows": row_count,
                    "goal_mode_rows": goal_rows,
                    "full_state_target_rows": row_count - goal_rows,
                    "set_size": set_size,
                    "target_directed_count": directed_count,
                    "max_anchor_target_distance": max_anchor_distance,
                    "chunk_check_failures": split_errors,
                }

    report = {
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "source_dataset": str(source_path),
        "expected_source_dataset_sha256": expected_source_hash,
        "actual_source_dataset_sha256": actual_source_hash,
        "source_hash_checked": not args.skip_source_hash,
        "splits": split_reports,
        "errors": errors,
        "passed": not errors,
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
