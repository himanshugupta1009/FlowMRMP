#!/usr/bin/env python3
"""Audit the v2 variable-pool Franka target-conditioned companion HDF5.

The audit verifies completion, source identity, variable-length pool offsets,
pool uniqueness, selected-slot referential integrity, the 29D condition layout,
goal/full-state target semantics, and that the first eight selected edges are
exactly the best actual goal-progress candidates available in each pool.
It is read-only and emits a JSON report beside the dataset by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


FORMAT_NAME = "franka_target_conditioned_valid_pool_bundle"
Q_RANGE = np.array(
    [5.7946, 3.5256, 5.7946, 3.0020, 5.7946, 3.7700, 5.7946],
    dtype=np.float32,
)
DQ_MAX = np.array(
    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    """Define audit path, hashing, and chunk controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_json(dataset: h5py.Dataset) -> dict[str, object]:
    """Decode a scalar UTF-8 JSON dataset."""
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def resolve_source(path: Path, dataset: h5py.File) -> Path:
    """Resolve the immutable source beside the companion or at its old path."""
    local = path.parent / str(dataset.attrs["source_dataset_basename"])
    original = Path(str(dataset.attrs["source_dataset_original_path"]))
    if local.is_file():
        return local.resolve()
    if original.is_file():
        return original.resolve()
    raise FileNotFoundError(f"Missing source HDF5: {local} or {original}")


def audit_split(group: h5py.Group, *, chunk_size: int) -> dict[str, object]:
    """Check one split's pools, selections, conditions, and progress ranking."""
    rows = int(group["base_conditions"].shape[0])
    variants = int(group["conds"].shape[0])
    errors: list[str] = []
    if variants != rows * 2:
        errors.append(f"variant count {variants} != 2 * base rows {rows}")
    if int(group.attrs.get("completed_rows", -1)) != rows:
        errors.append("completed_rows does not equal allocated base rows")
    if not bool(group.attrs.get("complete", False)):
        errors.append("split complete flag is false")
    offsets = group["pool_offsets"][:]
    counts = np.diff(offsets)
    flat_count = int(group["pool_edge_ids"].shape[0])
    if len(offsets) != rows + 1 or offsets[0] != 0 or offsets[-1] != flat_count:
        errors.append("pool offsets do not cover the flat pool datasets")
    if np.any(counts < 32):
        errors.append(f"{int(np.sum(counts < 32))} pools contain fewer than 32 edges")
    for name in ("num_steps", "delta_q", "delta_dq", "final_q", "final_dq"):
        if int(group[f"pool_{name}"].shape[0]) != flat_count:
            errors.append(f"pool_{name} length differs from pool_edge_ids")

    chunk_failures = 0
    max_full_anchor_distance = 0.0
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        cond = group["conds"][2 * start : 2 * stop]
        targets = group["target_relative_normalized"][2 * start : 2 * stop]
        goal_mode = group["goal_mode"][2 * start : 2 * stop].astype(bool)
        source = group["target_source"][2 * start : 2 * stop]
        selected_slots = group["selected_pool_slots"][2 * start : 2 * stop].astype(
            np.int64
        )
        selected_ids = group["selected_edge_ids"][2 * start : 2 * stop]
        selected_steps = group["selected_num_steps"][2 * start : 2 * stop]
        selected_dq = group["selected_delta_q"][2 * start : 2 * stop]
        selected_ddq = group["selected_delta_dq"][2 * start : 2 * stop]
        selected_progress = group["selected_progress"][2 * start : 2 * stop]
        best_progress = group["best_available_progress"][2 * start : 2 * stop]

        checks = [
            cond.shape[1] == 29,
            np.allclose(cond[:, 14:28], targets, atol=1e-6),
            np.allclose(cond[:, 28], goal_mode.astype(np.float32), atol=1e-6),
            np.all(goal_mode[0::2]),
            np.all(~goal_mode[1::2]),
            np.all(source[0::2] == 0),
            np.all(source[1::2] == 1),
            np.allclose(targets[goal_mode, 7:], 0.0, atol=1e-7),
        ]
        for local_base, base_row in enumerate(range(start, stop)):
            pool_start = int(offsets[base_row])
            pool_stop = int(offsets[base_row + 1])
            pool_ids = group["pool_edge_ids"][pool_start:pool_stop]
            pool_steps = group["pool_num_steps"][pool_start:pool_stop]
            pool_dq = group["pool_delta_q"][pool_start:pool_stop]
            pool_ddq = group["pool_delta_dq"][pool_start:pool_stop]
            if len(np.unique(pool_ids)) != len(pool_ids):
                checks.append(False)
                continue
            for variant_local in (2 * local_base, 2 * local_base + 1):
                slots = selected_slots[variant_local]
                if np.any(slots < 0) or np.any(slots >= len(pool_ids)):
                    checks.append(False)
                    continue
                checks.extend(
                    (
                        np.array_equal(selected_ids[variant_local], pool_ids[slots]),
                        np.array_equal(selected_steps[variant_local], pool_steps[slots]),
                        np.allclose(selected_dq[variant_local], pool_dq[slots], atol=1e-6),
                        np.allclose(selected_ddq[variant_local], pool_ddq[slots], atol=1e-6),
                    )
                )
                relative = np.concatenate(
                    (
                        2.0 * pool_dq / Q_RANGE[None, :],
                        pool_ddq / DQ_MAX[None, :],
                    ),
                    axis=1,
                )
                difference = relative - targets[variant_local][None, :]
                distance = np.linalg.norm(
                    difference[:, :7] if goal_mode[variant_local] else difference,
                    axis=1,
                )
                current_distance = np.linalg.norm(
                    targets[variant_local, :7]
                    if goal_mode[variant_local]
                    else targets[variant_local]
                )
                progress = current_distance - distance
                expected_directed = np.argsort(-progress, kind="stable")[:8]
                checks.extend(
                    (
                        np.array_equal(slots[:8], expected_directed),
                        np.allclose(
                            selected_progress[variant_local], progress[slots], atol=1e-5
                        ),
                        np.isclose(best_progress[variant_local], np.max(progress), atol=1e-5),
                    )
                )
                if not goal_mode[variant_local]:
                    max_full_anchor_distance = max(
                        max_full_anchor_distance, float(np.min(distance))
                    )
        if not all(bool(value) for value in checks):
            chunk_failures += 1

    if chunk_failures:
        errors.append(f"{chunk_failures} chunks failed semantic checks")
    return {
        "base_rows": rows,
        "variant_rows": variants,
        "hard_rows": int(group["hard_state"][:].sum()),
        "flat_valid_pool_edges": flat_count,
        "pool_count_min": int(counts.min()) if rows else 0,
        "pool_count_mean": float(counts.mean()) if rows else 0.0,
        "pool_count_max": int(counts.max()) if rows else 0,
        "max_full_state_anchor_distance": max_full_anchor_distance,
        "chunk_failures": chunk_failures,
        "errors": errors,
    }


def main() -> None:
    """Run both split audits, hash files, write JSON, and exit nonzero on errors."""
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    path = args.dataset.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else path.with_suffix(path.suffix + ".audit.json")
    )
    errors: list[str] = []
    with h5py.File(path, "r") as dataset:
        if dataset.attrs.get("format_name") != FORMAT_NAME:
            raise ValueError(f"Unexpected dataset format: {path}")
        if not bool(dataset.attrs.get("complete", False)):
            errors.append("top-level complete flag is false")
        metadata = decode_json(dataset["metadata_json"])
        source_path = resolve_source(path, dataset)
        expected_hash = metadata.get("source_dataset_sha256")
        actual_source_hash = None if args.skip_hashes else file_sha256(source_path)
        if not args.skip_hashes and expected_hash != actual_source_hash:
            errors.append("source SHA-256 differs from metadata")
        splits = {
            split: audit_split(dataset[split], chunk_size=args.chunk_size)
            for split in ("train", "val")
        }
        for split, report in splits.items():
            errors.extend(f"{split}: {message}" for message in report["errors"])
    report = {
        "dataset": str(path),
        "dataset_sha256": None if args.skip_hashes else file_sha256(path),
        "source_dataset": str(source_path),
        "expected_source_sha256": expected_hash,
        "actual_source_sha256": actual_source_hash,
        "splits": splits,
        "errors": errors,
        "passed": not errors,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
