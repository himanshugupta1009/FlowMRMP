#!/usr/bin/env python3
"""Build a compact target-conditioned view of a Franka edge-bundle HDF5.

The source file remains immutable and owns controls, outcomes, and trajectories.
This companion file adds two planner inputs for every bundle:

1. the normalized relative requested target; and
2. a goal-mode flag that removes target-velocity requirements.

Multiple variants of the same source query use different reachable target
anchors and target-dependent slot orders. The first slots are the candidates
whose physically integrated outcomes are closest to the requested target; the
remaining slots preserve the source bundle's diverse canonical order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


# Companion-file identity. Increment FORMAT_VERSION if the stored schema or
# normalization contract changes incompatibly.
FORMAT_NAME = "franka_target_conditioned_edge_bundle"
FORMAT_VERSION = 1


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the HDF5 into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Define the reproducible dataset-generation command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original edge HDF5.")
    parser.add_argument("--output", type=Path, required=True, help="New companion HDF5.")
    parser.add_argument(
        "--variants-per-bundle",
        type=int,
        default=2,
        help="Distinct reachable targets created for every source bundle.",
    )
    parser.add_argument(
        "--target-directed-count",
        type=int,
        default=8,
        help="Number of target-nearest edges placed at the front of each set.",
    )
    parser.add_argument("--seed", type=int, default=20260811, help="Target-anchor seed.")
    parser.add_argument(
        "--chunk-size", type=int, default=4096, help="Source rows processed per chunk."
    )
    parser.add_argument(
        "--train-limit", type=int, default=None, help="Optional smoke-data row limit."
    )
    parser.add_argument(
        "--val-limit", type=int, default=None, help="Optional smoke-data row limit."
    )
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Useful only for disposable smoke datasets; full datasets should be hashed.",
    )
    return parser.parse_args()


def decode_json_dataset(source: h5py.File, name: str) -> dict[str, object]:
    """Decode one scalar UTF-8 JSON dataset from an open HDF5 file."""
    value = source[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def create_dataset(
    group: h5py.File, name: str, shape: tuple[int, ...], dtype
) -> h5py.Dataset:
    """Create a row-chunked, LZF-compressed companion dataset."""
    row_chunk = min(max(1, 4096), max(1, shape[0]))
    chunks = (row_chunk, *shape[1:])
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression="lzf",
    )


def target_dependent_order(
    edge_relative: np.ndarray,
    target_relative: np.ndarray,
    goal_mode: np.ndarray,
    directed_count: int,
) -> np.ndarray:
    """Put target-nearest candidates first and retain source order afterward."""
    q_distance = np.linalg.norm(
        edge_relative[:, :, :7] - target_relative[:, None, :7], axis=2
    )
    full_distance = np.linalg.norm(
        edge_relative - target_relative[:, None, :], axis=2
    )
    distance = np.where(goal_mode[:, None], q_distance, full_distance)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :directed_count]
    set_size = edge_relative.shape[1]
    order = np.empty((len(edge_relative), set_size), dtype=np.uint8)
    order[:, :directed_count] = nearest.astype(np.uint8)
    for row_index, selected in enumerate(nearest):
        selected_mask = np.zeros(set_size, dtype=bool)
        selected_mask[selected] = True
        order[row_index, directed_count:] = np.flatnonzero(~selected_mask).astype(
            np.uint8
        )
    return order


def write_split(
    source: h5py.File,
    output: h5py.File,
    *,
    split: str,
    base_count: int,
    variants: int,
    directed_count: int,
    chunk_size: int,
    seed: int,
    q_range: np.ndarray,
    dq_max: np.ndarray,
) -> dict[str, object]:
    """Write one train/validation split and return its row-count summary.

    Each source bundle is repeated ``variants`` times. Every repeated row gets
    a distinct, reachable edge endpoint as its requested target, alternates
    between full-state and position-only goal semantics, and stores a
    target-dependent permutation of the original edge slots.
    """
    set_size = int(source[f"source_edge_ids_{split}"].shape[1])
    row_count = base_count * variants
    base_ids_out = create_dataset(output, f"base_indices_{split}", (row_count,), np.int64)
    conds_out = create_dataset(output, f"conds_{split}", (row_count, 29), np.float32)
    target_out = create_dataset(
        output, f"target_states_raw_{split}", (row_count, 14), np.float32
    )
    relative_out = create_dataset(
        output, f"target_relative_normalized_{split}", (row_count, 14), np.float32
    )
    goal_out = create_dataset(output, f"goal_mode_{split}", (row_count,), np.uint8)
    anchor_out = create_dataset(
        output, f"target_anchor_source_slot_{split}", (row_count,), np.uint8
    )
    order_out = create_dataset(
        output, f"edge_order_{split}", (row_count, set_size), np.uint8
    )

    rng = np.random.default_rng(seed)
    outcomes = source["bundle_outcomes"]
    written = 0
    goal_rows = 0
    for start in range(0, base_count, chunk_size):
        stop = min(start + chunk_size, base_count)
        base_indices = np.arange(start, stop, dtype=np.int64)
        batch_size = len(base_indices)
        source_cond = source[f"conds_{split}"][start:stop].astype(
            np.float32, copy=False
        )
        source_raw = source[f"conds_raw_{split}"][start:stop].astype(
            np.float32, copy=False
        )
        delta_q = outcomes[f"delta_q_{split}"][start:stop].astype(
            np.float32, copy=False
        )
        delta_dq = outcomes[f"delta_dq_{split}"][start:stop].astype(
            np.float32, copy=False
        )

        # Distinct target anchors for each variant force the condition to carry
        # information even when multiple rows share the exact same start state.
        random_scores = rng.random((batch_size, set_size))
        anchors = np.argsort(random_scores, axis=1)[:, :variants]
        flat_anchor = anchors.reshape(-1)
        row_ids = np.repeat(np.arange(batch_size), variants)
        flat_base = np.repeat(base_indices, variants)
        flat_cond = np.repeat(source_cond, variants, axis=0)
        flat_raw = np.repeat(source_raw, variants, axis=0)
        target_delta_q = delta_q[row_ids, flat_anchor]
        target_delta_dq = delta_dq[row_ids, flat_anchor]

        # With two variants every source query has one full-state steering row
        # and one position-only goal row. More variants remain deterministically
        # balanced by source row and variant index.
        variant_ids = np.tile(np.arange(variants), batch_size)
        flat_goal_mode = ((flat_base + variant_ids) % 2 == 1)
        goal_rows += int(flat_goal_mode.sum())
        target_relative = np.concatenate(
            (
                2.0 * target_delta_q / q_range[None, :],
                target_delta_dq / dq_max[None, :],
            ),
            axis=1,
        ).astype(np.float32)
        model_relative = target_relative.copy()
        model_relative[flat_goal_mode, 7:] = 0.0
        model_cond = np.concatenate(
            (flat_cond, model_relative, flat_goal_mode[:, None].astype(np.float32)),
            axis=1,
        ).astype(np.float32)
        target_raw = flat_raw.copy()
        target_raw[:, :7] += target_delta_q
        target_raw[:, 7:] += target_delta_dq

        edge_relative = np.concatenate(
            (
                2.0 * delta_q[row_ids] / q_range[None, None, :],
                delta_dq[row_ids] / dq_max[None, None, :],
            ),
            axis=2,
        ).astype(np.float32)
        order = target_dependent_order(
            edge_relative,
            model_relative,
            flat_goal_mode,
            directed_count,
        )
        out_slice = slice(written, written + len(flat_base))
        base_ids_out[out_slice] = flat_base
        conds_out[out_slice] = model_cond
        target_out[out_slice] = target_raw
        relative_out[out_slice] = model_relative
        goal_out[out_slice] = flat_goal_mode.astype(np.uint8)
        anchor_out[out_slice] = flat_anchor.astype(np.uint8)
        order_out[out_slice] = order
        written += len(flat_base)
        print(
            f"{split}: {written:,}/{row_count:,} target-conditioned rows",
            flush=True,
        )
    return {
        "source_bundle_rows": base_count,
        "target_conditioned_rows": row_count,
        "goal_mode_rows": goal_rows,
        "full_state_target_rows": row_count - goal_rows,
    }


def main() -> None:
    """Validate inputs and atomically create the companion HDF5."""
    args = parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    if args.variants_per_bundle <= 0 or args.chunk_size <= 0:
        raise ValueError("variants-per-bundle and chunk-size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_hash = None if args.skip_source_hash else file_sha256(source_path)
    with h5py.File(source_path, "r") as source:
        if source.attrs.get("format_name") != "franka_variable_length_edge_bundle":
            raise ValueError(f"Unexpected source format: {source_path}")
        source_metadata = decode_json_dataset(source, "metadata_json")
        set_size = int(source[f"source_edge_ids_train"].shape[1])
        if args.variants_per_bundle > set_size:
            raise ValueError("variants-per-bundle cannot exceed source set size")
        if not 0 < args.target_directed_count <= set_size:
            raise ValueError("target-directed-count must be within the source set size")
        normalization = source_metadata["normalization"]
        q_range = (
            np.asarray(normalization["q_upper"], dtype=np.float32)
            - np.asarray(normalization["q_lower"], dtype=np.float32)
        )
        dq_max = np.asarray(normalization["dq_max_abs"], dtype=np.float32)
        available = {
            split: int(source[f"conds_{split}"].shape[0])
            for split in ("train", "val")
        }
        limits = {"train": args.train_limit, "val": args.val_limit}
        counts = {
            split: available[split]
            if limits[split] is None
            else min(int(limits[split]), available[split])
            for split in ("train", "val")
        }

        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with h5py.File(temporary, "w") as output:
                output.attrs["format_name"] = FORMAT_NAME
                output.attrs["format_version"] = FORMAT_VERSION
                output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
                output.attrs["source_dataset_basename"] = source_path.name
                output.attrs["source_dataset_original_path"] = str(source_path)
                if source_hash is not None:
                    output.attrs["source_dataset_sha256"] = source_hash
                split_reports = {}
                for split, split_seed in (("train", args.seed), ("val", args.seed + 1)):
                    split_reports[split] = write_split(
                        source,
                        output,
                        split=split,
                        base_count=counts[split],
                        variants=args.variants_per_bundle,
                        directed_count=args.target_directed_count,
                        chunk_size=args.chunk_size,
                        seed=split_seed,
                        q_range=q_range,
                        dq_max=dq_max,
                    )
                metadata = {
                    "format_name": FORMAT_NAME,
                    "format_version": FORMAT_VERSION,
                    "source_dataset_basename": source_path.name,
                    "source_dataset_original_path": str(source_path),
                    "source_dataset_sha256": source_hash,
                    "source_dataset_metadata": source_metadata,
                    "seed": args.seed,
                    "variants_per_source_bundle": args.variants_per_bundle,
                    "target_directed_count": args.target_directed_count,
                    "condition_dim": 29,
                    "condition_layout": {
                        "current_normalized_state": [0, 14],
                        "relative_target_normalized": [14, 28],
                        "goal_mode_flag": 28,
                    },
                    "relative_target_normalization": {
                        "delta_q": "2 * delta_q / (q_upper - q_lower)",
                        "delta_dq": "delta_dq / dq_max_abs",
                        "goal_mode_delta_dq": "zeroed because goal success is position-only",
                    },
                    "target_policy": (
                        "reachable anchors are physically integrated members of the "
                        "source bundle; variants use distinct anchors"
                    ),
                    "edge_order": (
                        "target-nearest candidates first, then unselected members in "
                        "the source canonical order"
                    ),
                    "split_reports": split_reports,
                }
                text_dtype = h5py.string_dtype(encoding="utf-8")
                output.create_dataset(
                    "metadata_json",
                    data=json.dumps(metadata, sort_keys=True),
                    dtype=text_dtype,
                )
            temporary.replace(output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    print(f"saved target-conditioned dataset: {output_path}")


if __name__ == "__main__":
    main()
