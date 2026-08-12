#!/usr/bin/env python3
"""Train conditional flow matching on variable-length Franka edge bundles.

The HDF5 file remains the source of truth: every bundle contains references to
32 trajectory segments containing at most 50 executed acceleration intervals.
Each segment is encoded as a fixed vector containing zero-padded accelerations,
normalized step count, relative configuration change, and relative velocity
change. Padding is an encoding detail; the decoded planner edge retains only
its generated number of executed steps.

End-to-end data flow
--------------------
1. Load the 32 raw-edge references assigned to each condition. The original
   dataset uses ``c = [q_normalized(7), dq_normalized(7)]``. The target-aware
   companion dataset uses ``c = [current_state(14), relative_target(14),
   goal_mode(1)]`` and stores a target-dependent ordering of those edges.
2. Materialize each referenced edge as one 365D vector:
   ``50 * 7`` padded accelerations, one normalized step count, ``delta_q(7)``,
   and ``delta_dq(7)``.
3. Train a conditional velocity field from Gaussian noise at flow time 0 to
   the encoded edge-set distribution at flow time 1.
4. Mask padded acceleration steps in the main action loss, while applying a
   small separate regularizer that teaches padded outputs to stay near zero.
5. Save full training checkpoints and a smaller inference-only checkpoint.

Tensor symbols used below: ``B`` is batch size, ``K=32`` is bundle size,
``L=50`` is maximum edge length, ``A=7`` is action dimension, and ``D=365``
is encoded edge dimension.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT_DIR
    / "dataset"
    / "franka_eb_dataset"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v1.h5"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "trained_models" / "franka_edge_flow"

ACTION_DIM = 7
DEFAULT_MAX_ACTIONS = 50


# ---------------------------------------------------------------------------
# Reproducibility and experiment configuration
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a large file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(path: Path) -> dict[str, object]:
    """Record the repository revision and whether uncommitted edits are present."""
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


class TeeStream:
    """Mirror console output to both the original stream and a persistent log."""

    def __init__(self, terminal, log_stream):
        """Store the console and persistent-log destinations."""
        self.terminal = terminal
        self.log_stream = log_stream

    def write(self, value: str) -> int:
        """Write identical text to both destinations."""
        self.terminal.write(value)
        self.log_stream.write(value)
        return len(value)

    def flush(self) -> None:
        """Flush both destinations together."""
        self.terminal.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        """Preserve terminal detection for libraries that inspect stdout."""
        return bool(self.terminal.isatty())

    @property
    def encoding(self):
        """Expose the wrapped terminal's text encoding."""
        return self.terminal.encoding


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with microsecond uniqueness."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def utc_isoformat() -> str:
    """Return the current UTC time in an unambiguous machine-readable form."""
    return datetime.now(timezone.utc).isoformat()


def description_slug(value: str) -> str:
    """Convert a human experiment description into a safe short path component."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug[:80] or "run"


def available_memory_bytes() -> int | None:
    """Return Linux MemAvailable when exposed by the host."""
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1_024
    except (OSError, ValueError, IndexError):
        return None
    return None


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace one JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary_path, path)


def resolve_run_directory(args: argparse.Namespace) -> tuple[Path, bool]:
    """Create a unique fresh run directory or recover it from a resume checkpoint."""
    if args.resume is not None:
        checkpoint = args.resume.resolve()
        parent = checkpoint.parent
        run_dir = parent.parent if parent.name == "checkpoints" else parent
        return run_dir, True

    label = args.description or args.experiment_name or "baseline"
    stem = f"{utc_timestamp()}_{description_slug(label)}"
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, False


def dependency_versions() -> dict[str, str | None]:
    """Capture key package versions without requiring every optional package."""
    versions: dict[str, str | None] = {}
    for package in ("torch", "numpy", "h5py", "tqdm", "matplotlib"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_provenance(device: torch.device) -> dict[str, object]:
    """Describe the code and machine used for this training session."""
    gpu = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": int(index),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "gpu": gpu,
        "git": git_snapshot(ROOT_DIR),
    }


def write_git_diff(path: Path) -> None:
    """Save tracked working-tree changes used by the run for reproducibility."""
    result = subprocess.run(
        ["git", "-C", str(ROOT_DIR), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    path.write_bytes(result.stdout)


@dataclass
class TrainConfig:
    """Serializable record of every setting needed to reproduce a run."""
    dataset: str
    output_dir: str
    experiment_name: str
    seed: int
    device: str
    batch_size: int
    epochs: int
    lr: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    num_workers: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    time_embed_dim: int
    cond_embed_dim: int
    checkpoint_every: int
    sample_every: int
    sample_steps: int
    padded_action_weight: float
    target_progress_weight: float
    target_recall_radius: float
    cache_encoded_dataset: bool
    cache_build_batch_size: int
    amp: bool
    max_train_batches: int | None
    max_val_batches: int | None
    resume: str | None
    debug: bool


class SinusoidalTimeEmbedding(nn.Module):
    """Encode scalar flow time ``t in [0,1]`` with fixed sine/cosine features."""

    def __init__(self, dim: int):
        """Validate and store the even embedding width."""
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Convert ``t`` from shape ``(B,)`` to a feature tensor ``(B, dim)``."""
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t[:, None] * frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


# ---------------------------------------------------------------------------
# Conditional flow model
# ---------------------------------------------------------------------------


class EdgeSetFlowModel(nn.Module):
    """Predict the flow velocity for an entire conditional 32-edge set.

    Each edge is treated as a Transformer token. Attention lets the model
    coordinate all 32 outputs so it learns a diverse bundle rather than 32
    unrelated samples. Slot embeddings preserve either the source canonical
    ordering or the target-conditioned ordering supplied by the dataset.
    """

    def __init__(
        self,
        *,
        edge_dim: int,
        cond_dim: int = 14,
        set_size: int = 32,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
    ):
        """Construct embeddings, Transformer encoder, residual path, and head."""
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.cond_dim = int(cond_dim)
        self.set_size = int(set_size)
        # Flow time is global to a sample, so one embedding is broadcast to all
        # K edge tokens in that bundle.
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The edge vector is wider than a Transformer token. This scalar,
        # time-conditioned residual gives every input coordinate a direct path
        # to the output while attention learns the structured correction.
        self.skip_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, hidden_dim),
        )
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        # Slot i always represents the i-th member after canonical sorting.
        self.slot_embed = nn.Parameter(torch.zeros(1, set_size, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        nn.init.trunc_normal_(self.slot_embed, std=0.02)
        nn.init.zeros_(self.skip_scale[-1].weight)
        nn.init.constant_(self.skip_scale[-1].bias, -1.0)

    def forward(
        self, noisy_edges: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Return velocity predictions with shape ``(B,K,D)``.

        ``noisy_edges`` is the point currently moving along the flow path,
        ``t`` is its interpolation time, and ``cond`` is either the normalized
        14D Franka start state or the 29D target-aware condition shared by all
        K edges.
        """
        if noisy_edges.ndim != 3:
            raise ValueError(f"Expected (B,K,D), received {tuple(noisy_edges.shape)}")
        if noisy_edges.shape[1:] != (self.set_size, self.edge_dim):
            raise ValueError(
                f"Expected (*,{self.set_size},{self.edge_dim}), received "
                f"{tuple(noisy_edges.shape)}"
            )
        # Project each D-dimensional edge into one Transformer token.
        h = self.edge_in(noisy_edges)
        time_token = self.time_embed(t.reshape(-1).to(noisy_edges.dtype))
        skip_scale = self.skip_scale(time_token)[:, None, :]
        # Time and robot state are global context; slot identity is local.
        global_token = time_token + self.cond_embed(cond)
        h = h + global_token[:, None, :] + self.slot_embed
        return skip_scale * noisy_edges + self.out(self.backbone(h))


# ---------------------------------------------------------------------------
# HDF5 reference decoding and fixed-width edge encoding
# ---------------------------------------------------------------------------


class FrankaEdgeBundleStore:
    """Load the self-contained HDF5 and materialize training batches in RAM.

    The HDF5 stores each source trajectory once and represents bundles with raw
    edge IDs. This class flattens all source accelerations into one array, then
    reconstructs only the requested ``(B,K,L,A)`` control blocks per batch.
    """

    def __init__(self, path: Path):
        """Read immutable dataset arrays and derive normalization/offset tables."""
        self.path = Path(path).resolve()
        started = time.perf_counter()
        target_view = None
        with h5py.File(self.path, "r") as dataset:
            format_name = dataset.attrs.get("format_name")
            if format_name == "franka_target_conditioned_edge_bundle":
                metadata_value = dataset["metadata_json"][()]
                if isinstance(metadata_value, bytes):
                    metadata_value = metadata_value.decode("utf-8")
                target_metadata = json.loads(str(metadata_value))
                source_basename = str(dataset.attrs["source_dataset_basename"])
                local_source = self.path.parent / source_basename
                original_source = Path(
                    str(dataset.attrs["source_dataset_original_path"])
                )
                if local_source.is_file():
                    source_path = local_source.resolve()
                elif original_source.is_file():
                    source_path = original_source.resolve()
                else:
                    raise FileNotFoundError(
                        "Target-conditioned dataset requires its source HDF5. "
                        f"Expected {local_source} or {original_source}"
                    )
                target_view = {
                    "metadata": target_metadata,
                    "conds": {
                        split: dataset[f"conds_{split}"][:].astype(
                            np.float32, copy=False
                        )
                        for split in ("train", "val")
                    },
                    "base_indices": {
                        split: dataset[f"base_indices_{split}"][:]
                        for split in ("train", "val")
                    },
                    "edge_order": {
                        split: dataset[f"edge_order_{split}"][:].astype(
                            np.int64, copy=False
                        )
                        for split in ("train", "val")
                    },
                }
                self.preserve_bundle_order = True
            elif format_name == "franka_variable_length_edge_bundle":
                source_path = self.path
                self.preserve_bundle_order = False
            else:
                raise ValueError(f"Not a Franka edge-bundle dataset: {self.path}")

        self.source_path = source_path
        with h5py.File(source_path, "r") as source:
            if source.attrs.get("format_name") != "franka_variable_length_edge_bundle":
                raise ValueError(f"Invalid source Franka edge dataset: {source_path}")
            metadata_value = source["metadata_json"][()]
            if isinstance(metadata_value, bytes):
                metadata_value = metadata_value.decode("utf-8")
            self.metadata = json.loads(str(metadata_value))
            if target_view is None:
                # Conditions are normalized start states, shape (num_bundles, 14).
                self.conds = {
                    split: source[f"conds_{split}"][:].astype(
                        np.float32, copy=False
                    )
                    for split in ("train", "val")
                }
                # Each row stores the K raw-edge IDs selected for one bundle.
                self.bundle_ids = {
                    split: source[f"source_edge_ids_{split}"][:]
                    for split in ("train", "val")
                }
            else:
                self.conds = target_view["conds"]
                self.bundle_ids = {}
                for split in ("train", "val"):
                    base_indices = target_view["base_indices"][split]
                    unique_indices, inverse = np.unique(
                        base_indices, return_inverse=True
                    )
                    base_ids = source[f"source_edge_ids_{split}"][unique_indices][
                        inverse
                    ]
                    order = target_view["edge_order"][split]
                    self.bundle_ids[split] = np.take_along_axis(
                        base_ids, order, axis=1
                    )
            # Outcomes were computed during dataset construction and are used
            # both for training and fast endpoint ranking during planning.
            outcomes = source["bundle_outcomes"]
            if target_view is None:
                self.bundle_num_steps = {
                    split: outcomes[f"num_steps_{split}"][:]
                    for split in ("train", "val")
                }
                self.bundle_delta_q = {
                    split: outcomes[f"delta_q_{split}"][:]
                    for split in ("train", "val")
                }
                self.bundle_delta_dq = {
                    split: outcomes[f"delta_dq_{split}"][:]
                    for split in ("train", "val")
                }
            else:
                self.bundle_num_steps = {}
                self.bundle_delta_q = {}
                self.bundle_delta_dq = {}
                for split in ("train", "val"):
                    base_indices = target_view["base_indices"][split]
                    unique_indices, inverse = np.unique(
                        base_indices, return_inverse=True
                    )
                    order = target_view["edge_order"][split]
                    for destination, name in (
                        (self.bundle_num_steps, "num_steps"),
                        (self.bundle_delta_q, "delta_q"),
                        (self.bundle_delta_dq, "delta_dq"),
                    ):
                        values = outcomes[f"{name}_{split}"][unique_indices][inverse]
                        destination[split] = np.take_along_axis(
                            values,
                            order if values.ndim == 2 else order[:, :, None],
                            axis=1,
                        )
            # A raw edge is identified by source trajectory, start index, and
            # number of acceleration intervals to execute.
            raw = source["raw_edges"]
            raw_edge_count = int(raw["trajectory_index"].shape[0])
            selected_raw_ids = None
            if target_view is not None:
                selected_raw_ids = np.unique(
                    np.concatenate(
                        (
                            self.bundle_ids["train"].reshape(-1),
                            self.bundle_ids["val"].reshape(-1),
                        )
                    )
                )
                # A small smoke companion should not force all 24M raw-edge
                # records and 200k trajectories into RAM. Full companions use
                # most of the source and retain the faster contiguous load.
                if len(selected_raw_ids) >= 0.20 * raw_edge_count:
                    selected_raw_ids = None
            if selected_raw_ids is None:
                self.raw_trajectory = raw["trajectory_index"][:]
                self.raw_start = raw["start_index"][:]
                self.raw_num_steps = raw["num_steps"][:]
                self.raw_length = raw["trajectory_length"][:]
            else:
                self.raw_trajectory = raw["trajectory_index"][selected_raw_ids]
                self.raw_start = raw["start_index"][selected_raw_ids]
                self.raw_num_steps = raw["num_steps"][selected_raw_ids]
                self.raw_length = raw["trajectory_length"][selected_raw_ids]
                for split in ("train", "val"):
                    self.bundle_ids[split] = np.searchsorted(
                        selected_raw_ids, self.bundle_ids[split]
                    )

            names = source["source_trajectories/trajectory_names"][:]
            decoded_names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in names
            ]
            used_trajectory_ids = (
                np.arange(len(decoded_names), dtype=np.int64)
                if selected_raw_ids is None
                else np.unique(self.raw_trajectory)
            )
            trajectory_lengths = np.asarray(
                [
                    source["source_trajectories"][decoded_names[index]][
                        "accelerations"
                    ].shape[0]
                    for index in used_trajectory_ids
                ],
                dtype=np.int64,
            )
            # Offsets make every (trajectory, timestep) pair addressable in one
            # contiguous acceleration pool without duplicating HDF5 trajectories.
            self.trajectory_offsets = np.full(
                len(decoded_names), -1, dtype=np.int64
            )
            used_offsets = np.zeros(len(used_trajectory_ids), dtype=np.int64)
            if len(used_trajectory_ids) > 1:
                used_offsets[1:] = np.cumsum(trajectory_lengths[:-1])
            self.trajectory_offsets[used_trajectory_ids] = used_offsets
            self.encoded_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            self.accelerations = np.empty(
                (int(trajectory_lengths.sum()), ACTION_DIM), dtype=np.float32
            )
            for trajectory_id, length, offset in zip(
                used_trajectory_ids, trajectory_lengths, used_offsets
            ):
                name = decoded_names[int(trajectory_id)]
                self.accelerations[offset : offset + int(length)] = source[
                    "source_trajectories"
                ][name]["accelerations"][:]

        if target_view is not None:
            target_metadata = target_view["metadata"]
            self.metadata.update(
                {
                    "cond_dim": int(target_metadata["condition_dim"]),
                    "target_conditioning": {
                        "format_name": target_metadata["format_name"],
                        "format_version": target_metadata["format_version"],
                        "condition_layout": target_metadata["condition_layout"],
                        "relative_target_normalization": target_metadata[
                            "relative_target_normalization"
                        ],
                        "target_policy": target_metadata["target_policy"],
                        "edge_order": target_metadata["edge_order"],
                        "target_directed_count": target_metadata[
                            "target_directed_count"
                        ],
                        "source_dataset_sha256": target_metadata.get(
                            "source_dataset_sha256"
                        ),
                    },
                }
            )

        norm = self.metadata["normalization"]
        self.q_range = np.asarray(norm["q_upper"], dtype=np.float32) - np.asarray(
            norm["q_lower"], dtype=np.float32
        )
        self.dq_max = np.asarray(norm["dq_max_abs"], dtype=np.float32)
        self.ddq_max = np.asarray(norm["ddq_max_abs"], dtype=np.float32)
        self.dt = float(self.metadata["dt"])
        self.max_duration = float(self.metadata["max_suffix_duration"])
        self.set_size = int(self.metadata["set_size"])
        self.max_actions = int(
            self.metadata.get("max_edge_steps", np.max(self.raw_num_steps))
        )
        if self.max_actions != DEFAULT_MAX_ACTIONS:
            raise ValueError(
                f"Expected {DEFAULT_MAX_ACTIONS} maximum actions, found {self.max_actions}"
            )
        if not np.array_equal(self.raw_length, self.raw_num_steps + 1):
            raise ValueError("trajectory_length must equal num_steps + 1")
        # Fixed vector layout: [L*A actions | N | delta_q(7) | delta_dq(7)].
        self.action_block_dim = self.max_actions * ACTION_DIM
        self.step_count_index = self.action_block_dim
        self.delta_q_slice = slice(
            self.step_count_index + 1, self.step_count_index + 8
        )
        self.delta_dq_slice = slice(
            self.step_count_index + 8, self.step_count_index + 15
        )
        self.edge_dim = self.action_block_dim + 15
        self.metadata.update(
            {
                "edge_dim": self.edge_dim,
                "cond_dim": int(self.conds["train"].shape[1]),
                "set_size": self.set_size,
                "flow_model_architecture": (
                    "SOC 32-slot conditional Transformer with a time-conditioned "
                    f"coordinate residual for the {self.edge_dim}D Franka edge"
                ),
                "flow_encoding": {
                    "name": "zero_padded_piecewise_constant_acceleration_v2",
                    "max_actions": self.max_actions,
                    "action_dim": ACTION_DIM,
                    "action_block_dim": self.action_block_dim,
                    "step_count_index": self.step_count_index,
                    "delta_q_slice": [self.delta_q_slice.start, self.delta_q_slice.stop],
                    "delta_dq_slice": [
                        self.delta_dq_slice.start,
                        self.delta_dq_slice.stop,
                    ],
                    "acceleration_normalization": "ddq / ddq_max_abs",
                    "step_count_normalization": "num_steps / max_actions",
                    "duration_reconstruction": "num_steps * dt",
                    "delta_q_normalization": "delta_q / (q_upper - q_lower)",
                    "delta_dq_normalization": "delta_dq / (2 * dq_max_abs)",
                    "outcome_reference_frame": (
                        "bundle query condition; terminal state comes from float64 "
                        "integration of the selected acceleration sequence"
                    ),
                    "executed_action_count": "num_steps",
                    "padding": (
                        "zeros to max_actions; ground-truth num_steps mask gives "
                        "each edge equal valid-action loss weight"
                    ),
                    "canonical_order": (
                        "target-nearest first, then source canonical order"
                        if self.preserve_bundle_order
                        else "lexicographic normalized delta_q, then normalized num_steps"
                    ),
                },
            }
        )
        self.load_seconds = time.perf_counter() - started

    def encoded_cache_bytes(self) -> int:
        """Return bytes needed to cache every encoded edge and integer count."""
        sample_count = sum(len(values) for values in self.conds.values())
        edge_bytes = sample_count * self.set_size * self.edge_dim * np.dtype(np.float32).itemsize
        count_bytes = sample_count * self.set_size * np.dtype(np.int64).itemsize
        return int(edge_bytes + count_bytes)

    def build_encoded_cache(self, batch_size: int = 1_024) -> dict[str, float]:
        """Materialize the exact existing edge encoding once for repeated epochs."""
        if batch_size <= 0:
            raise ValueError("cache build batch size must be positive")
        if self.encoded_cache:
            raise RuntimeError("encoded cache has already been built")
        started = time.perf_counter()
        for split in ("train", "val"):
            sample_count = len(self.conds[split])
            cached_edges = np.empty(
                (sample_count, self.set_size, self.edge_dim), dtype=np.float32
            )
            cached_counts = np.empty(
                (sample_count, self.set_size), dtype=np.int64
            )
            progress = tqdm(
                range(0, sample_count, batch_size),
                desc=f"cache {split}",
                leave=False,
                disable=not sys.stderr.isatty(),
            )
            for start in progress:
                stop = min(start + batch_size, sample_count)
                indices = np.arange(start, stop, dtype=np.int64)
                _, edges, _, counts = self.encode_batch(split, indices)
                cached_edges[start:stop] = edges
                cached_counts[start:stop] = counts
            self.encoded_cache[split] = (cached_edges, cached_counts)
        return {
            "seconds": time.perf_counter() - started,
            "bytes": float(self.encoded_cache_bytes()),
        }

    def __len__(self) -> int:
        """Return the total number of train and validation bundles."""
        return len(self.conds["train"]) + len(self.conds["val"])

    def encode_batch(
        self, split: str, sample_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode selected bundles into model-ready NumPy arrays.

        Returns ``cond (B,14)``, ``edges (B,K,D)``, ``action_mask (B,K,L)``,
        and integer ``action_counts (B,K)``. Values are normalized so common
        feature blocks have comparable numerical scale.
        """
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if split in self.encoded_cache:
            cached_edges, cached_counts = self.encoded_cache[split]
            action_counts = cached_counts[sample_indices]
            steps = np.arange(self.max_actions, dtype=np.int64)[None, None, :]
            action_mask = steps < action_counts[:, :, None]
            return (
                self.conds[split][sample_indices],
                cached_edges[sample_indices],
                action_mask,
                action_counts,
            )
        edge_ids = self.bundle_ids[split][sample_indices]
        batch_size = edge_ids.shape[0]
        action_counts = self.bundle_num_steps[split][sample_indices].astype(
            np.int64
        )
        # Convert source-local acceleration indices into the flattened pool.
        starts = (
            self.trajectory_offsets[self.raw_trajectory[edge_ids]]
            + self.raw_start[edge_ids]
        )
        steps = np.arange(self.max_actions, dtype=np.int64)[None, None, :]
        action_mask = steps < action_counts[:, :, None]
        # Clamp padded gathers to the last real index, then multiply them away.
        # This avoids out-of-bounds indexing for short edges.
        safe_steps = np.minimum(steps, action_counts[:, :, None] - 1)
        pool_indices = starts[:, :, None] + safe_steps
        actions = self.accelerations[pool_indices]
        actions = actions / self.ddq_max[None, None, None, :]
        actions *= action_mask[:, :, :, None]

        # Store duration as normalized discrete step count, not floating time.
        step_count = action_counts.astype(np.float32) / float(self.max_actions)
        delta_q = self.bundle_delta_q[split][sample_indices] / self.q_range[
            None, None, :
        ]
        delta_dq = self.bundle_delta_dq[split][sample_indices] / (
            2.0 * self.dq_max[None, None, :]
        )
        edges = np.concatenate(
            (
                actions.reshape(batch_size, self.set_size, self.action_block_dim),
                step_count[:, :, None],
                delta_q,
                delta_dq,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)

        if not self.preserve_bundle_order:
            # Legacy source files have deterministic raw-edge-ID order. Reorder
            # them geometrically so learned slots have consistent meaning.
            keys = [step_count]
            keys.extend(delta_q[:, :, joint] for joint in range(6, -1, -1))
            order = np.lexsort(tuple(keys), axis=1)
            edges = np.take_along_axis(edges, order[:, :, None], axis=1)
            action_mask = np.take_along_axis(action_mask, order[:, :, None], axis=1)
            action_counts = np.take_along_axis(action_counts, order, axis=1)
        # Target-conditioned views already store target-nearest candidates in
        # the first slots, so re-sorting them here would erase the new signal.
        cond = self.conds[split][sample_indices]
        return cond, edges, action_mask, action_counts


class IndexDataset(Dataset):
    """Tiny Dataset that lets DataLoader shuffle integer bundle indices."""

    def __init__(self, size: int):
        """Store the number of addressable bundle rows."""
        self.size = int(size)

    def __len__(self) -> int:
        """Return the number of bundle indices in this split."""
        return self.size

    def __getitem__(self, index: int) -> int:
        """Return the requested integer for later vectorized collation."""
        return int(index)


class EncodeBatch:
    """DataLoader collator that decodes a list of bundle IDs in one vectorized call."""

    def __init__(self, store: FrankaEdgeBundleStore, split: str):
        """Bind one in-memory store and its train/validation split."""
        self.store = store
        self.split = split

    def __call__(self, indices: list[int]):
        """Return CPU tensors; the epoch runner transfers them to the device."""
        cond, edges, mask, counts = self.store.encode_batch(
            self.split, np.asarray(indices, dtype=np.int64)
        )
        return (
            torch.from_numpy(cond),
            torch.from_numpy(edges),
            torch.from_numpy(mask),
            torch.from_numpy(counts),
        )


# ---------------------------------------------------------------------------
# Command-line configuration and deterministic runtime setup
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Define model, optimizer, checkpoint, and debug command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--description",
        default=None,
        help="Short human label appended to the timestamped run directory.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--cond-embed-dim", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument(
        "--padded-action-weight",
        type=float,
        default=0.01,
        help=(
            "Small separate regularizer that teaches unused padded action "
            "coordinates to flow to zero. Valid-step loss is always normalized "
            "per edge using the ground-truth num_steps mask."
        ),
    )
    parser.add_argument(
        "--target-progress-weight",
        type=float,
        default=0.25,
        help=(
            "Weight for best-of-K normalized target-distance loss. Requires a "
            "29D target-conditioned dataset when greater than zero."
        ),
    )
    parser.add_argument(
        "--target-recall-radius",
        type=float,
        default=0.40,
        help="Normalized radius used for target recall metrics.",
    )
    parser.add_argument(
        "--cache-encoded-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache exact fixed-width edge encodings in RAM for faster epochs.",
    )
    parser.add_argument("--cache-build-batch-size", type=int, default=1_024)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when training on CUDA.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def cli_option_present(*names: str) -> bool:
    """Return whether any exact or equals-style CLI option was supplied."""
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
        for name in names
    )


def apply_resume_configuration(args: argparse.Namespace) -> None:
    """Recover omitted run settings and reject incompatible resume overrides."""
    if args.resume is None:
        return
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    required_training_state = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
    }
    missing_training_state = required_training_state.difference(checkpoint)
    if missing_training_state:
        raise ValueError(
            "Resume requires a full training checkpoint; missing "
            f"{sorted(missing_training_state)}. Use best.pt, last.pt, or an "
            "epoch_XXXX.pt checkpoint instead of best_inference.pt."
        )
    saved = checkpoint.get("config", {})
    options = {
        "dataset": ("--dataset",),
        "seed": ("--seed",),
        "batch_size": ("--batch-size",),
        "epochs": ("--epochs",),
        "lr": ("--lr",),
        "min_lr_ratio": ("--min-lr-ratio",),
        "warmup_steps": ("--warmup-steps",),
        "weight_decay": ("--weight-decay",),
        "grad_clip": ("--grad-clip",),
        "hidden_dim": ("--hidden-dim",),
        "depth": ("--depth",),
        "num_heads": ("--num-heads",),
        "mlp_ratio": ("--mlp-ratio",),
        "dropout": ("--dropout",),
        "time_embed_dim": ("--time-embed-dim",),
        "cond_embed_dim": ("--cond-embed-dim",),
        "checkpoint_every": ("--checkpoint-every",),
        "sample_every": ("--sample-every",),
        "sample_steps": ("--sample-steps",),
        "padded_action_weight": ("--padded-action-weight",),
        "target_progress_weight": ("--target-progress-weight",),
        "target_recall_radius": ("--target-recall-radius",),
        "max_train_batches": ("--max-train-batches",),
        "max_val_batches": ("--max-val-batches",),
        "cache_encoded_dataset": (
            "--cache-encoded-dataset",
            "--no-cache-encoded-dataset",
        ),
        "cache_build_batch_size": ("--cache-build-batch-size",),
        "amp": ("--amp", "--no-amp"),
    }
    strict_fields = {
        "batch_size",
        "lr",
        "min_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "hidden_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "time_embed_dim",
        "cond_embed_dim",
        "padded_action_weight",
        "target_progress_weight",
        "target_recall_radius",
        "max_train_batches",
    }
    for field, names in options.items():
        if field not in saved:
            continue
        saved_value = Path(saved[field]) if field == "dataset" else saved[field]
        supplied = cli_option_present(*names)
        if supplied and field in strict_fields and getattr(args, field) != saved_value:
            raise ValueError(
                f"Cannot change --{field.replace('_', '-')} when resuming: "
                f"checkpoint={saved_value!r}, requested={getattr(args, field)!r}"
            )
        if not supplied:
            setattr(args, field, saved_value)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` as CUDA, then Apple MPS, then CPU, with strict overrides."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable to this process")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every available CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    """Capture every RNG used by training so a resumed run is reproducible."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore RNG state saved in a full training checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def cosine_schedule_factor(
    step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> float:
    """Return linear-warmup then cosine-decay multiplier for one optimizer step."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Conditional flow-matching objective
# ---------------------------------------------------------------------------


def target_condition_metrics(
    estimated_edges: torch.Tensor,
    cond: torch.Tensor,
    *,
    action_block_dim: int,
    recall_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best-of-K target distance and recall for a 29D condition.

    Edge deltas use half the normalized-state scale used by the target
    condition, hence the factor of two. Goal-mode rows ignore velocity exactly
    like the Franka planner's position-only goal test.
    """
    if cond.ndim != 2 or cond.shape[1] != 29:
        raise ValueError(
            "Target-aware training requires condition shape (B,29): current "
            "state, relative target, and goal-mode flag"
        )
    target_relative = cond[:, 14:28]
    goal_mode = cond[:, 28] >= 0.5
    edge_relative = torch.cat(
        (
            2.0 * estimated_edges[:, :, action_block_dim + 1 : action_block_dim + 8],
            2.0 * estimated_edges[:, :, action_block_dim + 8 : action_block_dim + 15],
        ),
        dim=2,
    )
    difference = edge_relative - target_relative[:, None, :]
    difference = difference.clone()
    difference[goal_mode, :, 7:] = 0.0
    distances = torch.linalg.vector_norm(difference, dim=2)
    best_distance = distances.min(dim=1).values
    recall = (best_distance <= float(recall_radius)).to(estimated_edges.dtype)
    return best_distance, recall


def flow_matching_loss(
    model: nn.Module,
    edges: torch.Tensor,
    cond: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_block_dim: int,
    max_actions: int,
    padded_action_weight: float,
    target_progress_weight: float,
    target_recall_radius: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute balanced conditional flow-matching loss for ragged edges.

    The straight probability path is ``x_t=(1-t)*noise+t*data`` and therefore
    has constant target velocity ``data-noise``. Valid action steps are averaged
    per edge before averaging the batch, so a 50-step edge does not receive more
    weight than a 5-step edge. Step count and both relative-outcome blocks are
    always supervised. Padded controls receive only the small auxiliary term.
    """
    # Draw one Gaussian source edge-set and one flow time per bundle.
    noise = torch.randn(
        edges.shape,
        device=edges.device,
        dtype=edges.dtype,
        generator=generator,
    )
    t = torch.rand(
        edges.shape[0],
        device=edges.device,
        dtype=edges.dtype,
        generator=generator,
    )
    t_view = t[:, None, None]
    # Interpolate from noise (t=0) to the encoded data example (t=1).
    noisy_edges = (1.0 - t_view) * noise + t_view * edges
    target = edges - noise
    prediction = model(noisy_edges, t, cond)
    squared = (prediction - target).square()

    # Reshape the action block so the ground-truth variable-length mask can be
    # applied at the timestep level.
    action_error = squared[:, :, :action_block_dim].reshape(
        edges.shape[0], edges.shape[1], max_actions, ACTION_DIM
    )
    valid_mask = action_mask[:, :, :, None].to(action_error.dtype)
    valid_count = valid_mask.sum(dim=(2, 3)) * ACTION_DIM
    valid_action_per_edge = (action_error * valid_mask).sum(dim=(2, 3)) / (
        valid_count.clamp_min(1.0)
    )
    action_loss = valid_action_per_edge.mean()

    # Padding is not part of the physical edge; this low-weight term merely
    # makes inference padding well behaved.
    padding_mask = 1.0 - valid_mask
    padding_count = padding_mask.sum(dim=(2, 3)) * ACTION_DIM
    padded_action_per_edge = (action_error * padding_mask).sum(dim=(2, 3)) / (
        padding_count.clamp_min(1.0)
    )
    has_padding = padding_count > 0
    if bool(has_padding.any()):
        padding_loss = padded_action_per_edge[has_padding].mean()
    else:
        padding_loss = torch.zeros((), device=edges.device, dtype=edges.dtype)

    step_count_loss = squared[:, :, action_block_dim].mean()
    delta_q_loss = squared[:, :, action_block_dim + 1 : action_block_dim + 8].mean()
    delta_dq_loss = squared[:, :, action_block_dim + 8 : action_block_dim + 15].mean()
    # Give the four semantic targets equal top-level influence even though
    # their raw coordinate counts differ substantially.
    base_flow_loss = (
        action_loss + step_count_loss + delta_q_loss + delta_dq_loss
    ) / 4.0 + padded_action_weight * padding_loss
    if target_progress_weight > 0.0:
        # Under the straight flow path, x_1 = x_t + (1-t) * velocity. This
        # converts the velocity prediction into an estimated clean edge set so
        # target progress can supervise the generated outcome, not just the
        # raw condition embedding.
        estimated_edges = noisy_edges + (1.0 - t_view) * prediction
        best_target_distance, target_recall = target_condition_metrics(
            estimated_edges,
            cond,
            action_block_dim=action_block_dim,
            recall_radius=target_recall_radius,
        )
        target_progress_loss = best_target_distance.mean()
    else:
        target_progress_loss = torch.zeros(
            (), device=edges.device, dtype=edges.dtype
        )
        target_recall = torch.zeros(
            (edges.shape[0],), device=edges.device, dtype=edges.dtype
        )
    total = base_flow_loss + target_progress_weight * target_progress_loss
    return total, {
        "base_flow": base_flow_loss.detach(),
        "action": action_loss.detach(),
        "padding": padding_loss.detach(),
        "step_count": step_count_loss.detach(),
        "delta_q": delta_q_loss.detach(),
        "delta_dq": delta_dq_loss.detach(),
        "target_progress": target_progress_loss.detach(),
        "target_recall": target_recall.mean().detach(),
    }


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    grad_clip: float,
    action_block_dim: int,
    max_actions: int,
    padded_action_weight: float,
    target_progress_weight: float,
    target_recall_radius: float,
    max_batches: int | None,
    description: str,
    validation_seed: int | None = None,
) -> dict[str, float]:
    """Run one train or validation epoch and return sample-weighted metrics.

    Passing an optimizer enables gradients and updates. Validation uses a fixed
    noise generator so epoch-to-epoch loss changes reflect the model rather
    than a newly sampled validation path.
    """
    training = optimizer is not None
    model.train(training)
    totals = {
        key: 0.0
        for key in (
            "loss",
            "base_flow",
            "action",
            "padding",
            "step_count",
            "delta_q",
            "delta_dq",
            "target_progress",
            "target_recall",
        )
    }
    sample_count = 0
    generator = None
    if not training and validation_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(validation_seed))
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(
            loader,
            desc=description,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for batch_index, (cond, edges, action_mask, _) in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            cond = cond.to(device=device, dtype=torch.float32)
            edges = edges.to(device=device, dtype=torch.float32)
            action_mask = action_mask.to(device=device)
            # Only the training pass owns an optimizer and gradient graph.
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                loss, blocks = flow_matching_loss(
                    model,
                    edges,
                    cond,
                    action_mask,
                    action_block_dim=action_block_dim,
                    max_actions=max_actions,
                    padded_action_weight=padded_action_weight,
                    target_progress_weight=target_progress_weight,
                    target_recall_radius=target_recall_radius,
                    generator=generator,
                )
            if optimizer is not None:
                optimizer_stepped = True
                if scaler is not None and scaler.is_enabled():
                    scale_before = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = scaler.get_scale() >= scale_before
                else:
                    loss.backward()
                    if grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None and optimizer_stepped:
                    scheduler.step()
            batch_samples = int(edges.shape[0])
            totals["loss"] += float(loss.item()) * batch_samples
            for key, value in blocks.items():
                totals[key] += float(value.item()) * batch_samples
            sample_count += batch_samples
            progress.set_postfix(loss=f"{loss.item():.5f}")
    denominator = max(sample_count, 1)
    return {key: value / denominator for key, value in totals.items()}


@torch.no_grad()
def sample_edge_sets(
    model: nn.Module,
    cond: torch.Tensor,
    *,
    steps: int,
    set_size: int,
    edge_dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate encoded edge sets by Euler-integrating the learned ODE 0 -> 1."""
    model.eval()
    edges = torch.randn(
        cond.shape[0],
        set_size,
        edge_dim,
        device=cond.device,
        generator=generator,
    )
    dt = 1.0 / float(steps)
    # Explicit Euler is intentionally identical to planner-time sampling.
    for step in range(steps):
        t = torch.full(
            (cond.shape[0],),
            step / float(steps),
            device=cond.device,
            dtype=edges.dtype,
        )
        edges = edges + dt * model(edges, t, cond)
    return edges


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    metadata: dict,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_validation_loss: float,
    best_epoch: int,
    inference_only: bool = False,
) -> None:
    """Atomically save model metadata, metrics, and optionally optimizer state."""
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "dataset_metadata": metadata,
        "epoch": int(epoch),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": int(best_epoch),
    }
    if not inference_only:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace prevents a partial checkpoint after interruption.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def append_metrics(path: Path, epoch: int, train: dict, val: dict) -> None:
    """Append one epoch of overall and per-feature losses to CSV."""
    fields = [
        "loss",
        "base_flow",
        "action",
        "padding",
        "step_count",
        "delta_q",
        "delta_dq",
        "target_progress",
        "target_recall",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(["epoch", *[f"train_{x}" for x in fields], *[f"val_{x}" for x in fields]])
        writer.writerow(
            [epoch]
            + [f"{train[x]:.10f}" for x in fields]
            + [f"{val[x]:.10f}" for x in fields]
        )


def plot_losses(path: Path, metrics_path: Path) -> None:
    """Render the overall and component validation curves when Matplotlib exists."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = np.genfromtxt(metrics_path, delimiter=",", names=True)
    if values.size == 0:
        return
    values = np.atleast_1d(values)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(values["epoch"], values["train_loss"], label="train")
    axes[0].plot(values["epoch"], values["val_loss"], label="validation")
    axes[0].set(title="Balanced flow-matching loss", xlabel="Epoch", ylabel="MSE")
    axes[0].legend()
    for block in (
        "base_flow",
        "action",
        "padding",
        "step_count",
        "delta_q",
        "delta_dq",
        "target_progress",
        "target_recall",
    ):
        axes[1].plot(values["epoch"], values[f"val_{block}"], label=block)
    axes[1].set(title="Validation loss by feature block", xlabel="Epoch", ylabel="MSE")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    status: dict[str, object],
) -> None:
    """Build the data/model stack, train, and persist artifacts for one run."""
    if args.num_workers != 0:
        raise ValueError("Use --num-workers 0; the RAM-backed store must not be duplicated")
    if args.padded_action_weight < 0.0:
        raise ValueError("padded-action-weight must be nonnegative")
    if args.target_progress_weight < 0.0:
        raise ValueError("target-progress-weight must be nonnegative")
    if args.target_recall_radius <= 0.0:
        raise ValueError("target-recall-radius must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be nonnegative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be between 0 and 1")
    if args.cache_build_batch_size <= 0:
        raise ValueError("cache-build-batch-size must be positive")
    # Debug mode keeps the full code path but shrinks every expensive dimension.
    if args.debug:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.hidden_dim = min(args.hidden_dim, 64)
        args.depth = min(args.depth, 2)
        args.num_heads = min(args.num_heads, 4)
        args.max_train_batches = args.max_train_batches or 2
        args.max_val_batches = args.max_val_batches or 1
        args.sample_every = 1
        args.checkpoint_every = 1
        args.sample_steps = min(args.sample_steps, 4)
        args.warmup_steps = min(args.warmup_steps, 1)

    set_seed(args.seed)
    # Loading once avoids repeated HDF5 random access during hundreds of
    # thousands of shuffled bundle lookups.
    store = FrankaEdgeBundleStore(args.dataset)
    store.metadata.update(
        {
            "dataset_sha256": file_sha256(store.path),
            "dataset_bytes": store.path.stat().st_size,
            "validation_protocol": (
                "held-out bundle split; fixed flow noise seed; metrics weighted by "
                "the number of samples in each batch"
            ),
            "training_runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "platform": platform.platform(),
                "flowmrmp_git": git_snapshot(ROOT_DIR),
            },
        }
    )
    cache_report: dict[str, object] = {
        "enabled": bool(args.cache_encoded_dataset),
        "estimated_bytes": store.encoded_cache_bytes(),
    }
    if args.cache_encoded_dataset:
        available_bytes = available_memory_bytes()
        required_bytes = store.encoded_cache_bytes()
        cache_report["available_memory_bytes_before_build"] = available_bytes
        if available_bytes is not None and required_bytes > 0.5 * available_bytes:
            raise MemoryError(
                "Encoded cache requires "
                f"{required_bytes / 2**30:.2f} GiB but only "
                f"{available_bytes / 2**30:.2f} GiB is available. "
                "Use --no-cache-encoded-dataset."
            )
        print(
            f"Building exact encoded cache ({required_bytes / 2**30:.2f} GiB)",
            flush=True,
        )
        cache_report.update(
            store.build_encoded_cache(batch_size=args.cache_build_batch_size)
        )
        print(
            "Encoded cache ready in {:.1f}s".format(cache_report["seconds"]),
            flush=True,
        )
    store.metadata["training_runtime"]["encoded_cache"] = cache_report
    experiment_name = run_dir.name
    config = TrainConfig(
        dataset=str(args.dataset.resolve()),
        output_dir=str(run_dir.parent),
        experiment_name=experiment_name,
        seed=args.seed,
        device=str(device),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
        cond_embed_dim=args.cond_embed_dim,
        checkpoint_every=args.checkpoint_every,
        sample_every=args.sample_every,
        sample_steps=args.sample_steps,
        padded_action_weight=args.padded_action_weight,
        target_progress_weight=args.target_progress_weight,
        target_recall_radius=args.target_recall_radius,
        cache_encoded_dataset=args.cache_encoded_dataset,
        cache_build_batch_size=args.cache_build_batch_size,
        amp=bool(args.amp and device.type == "cuda"),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=None if args.resume is None else str(args.resume.resolve()),
        debug=args.debug,
    )

    # The loader shuffles lightweight indices; EncodeBatch performs the actual
    # vectorized decode and keeps multiprocessing disabled to avoid RAM copies.
    train_loader = DataLoader(
        IndexDataset(len(store.conds["train"])),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=EncodeBatch(store, "train"),
    )
    val_loader = DataLoader(
        IndexDataset(len(store.conds["val"])),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=EncodeBatch(store, "val"),
    )
    # One forward call predicts all 32 edge velocities for every condition.
    model = EdgeSetFlowModel(
        edge_dim=store.edge_dim,
        cond_dim=int(store.metadata["cond_dim"]),
        set_size=store.set_size,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        time_embed_dim=config.time_embed_dim,
        cond_embed_dim=config.cond_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_steps_per_epoch = len(train_loader)
    if config.max_train_batches is not None:
        train_steps_per_epoch = min(train_steps_per_epoch, config.max_train_batches)
    total_training_steps = max(train_steps_per_epoch * config.epochs, 1)
    effective_warmup_steps = min(config.warmup_steps, max(total_training_steps - 1, 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_schedule_factor(
            step,
            warmup_steps=effective_warmup_steps,
            total_steps=total_training_steps,
            min_lr_ratio=config.min_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    # Resume restores the complete training state; a new run refuses to overwrite
    # an existing non-empty directory.
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        expected_dataset_hash = checkpoint.get("dataset_metadata", {}).get(
            "dataset_sha256"
        )
        actual_dataset_hash = store.metadata["dataset_sha256"]
        if (
            expected_dataset_hash is not None
            and expected_dataset_hash != actual_dataset_hash
        ):
            raise ValueError(
                "Resume checkpoint/dataset mismatch: "
                f"expected {expected_dataset_hash}, found {actual_dataset_hash}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get("val_metrics", {}).get("loss", float("inf")),
            )
        )
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))

    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    samples_dir = run_dir / "samples"
    manifest_path = run_dir / "run.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest.setdefault("resolved_config_history", []).append(
        {
            "session_id": status["active_session"],
            "config": asdict(config),
        }
    )
    manifest.update(
        {
            "resolved_config": asdict(config),
            "dataset_metadata": store.metadata,
            "artifacts": {
                "checkpoints": str(checkpoint_dir.resolve()),
                "metrics": str(metrics_dir.resolve()),
                "samples": str(samples_dir.resolve()),
                "console_log": str((run_dir / "console.log").resolve()),
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        run_dir / "config.json",
        {"config": asdict(config), "dataset_metadata": store.metadata},
    )

    print(f"Training: {config.experiment_name}", flush=True)
    print(f"Dataset: {store.path}", flush=True)
    print(f"RAM data store load: {store.load_seconds:.3f}s", flush=True)
    print(f"Train/validation: {len(store.conds['train']):,}/{len(store.conds['val']):,}", flush=True)
    print(f"Edge tensor: (32, {store.edge_dim})", flush=True)
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    print(f"Device: {device}; AMP: {config.amp}", flush=True)
    print(
        f"LR schedule: warmup={effective_warmup_steps:,} steps, "
        f"total={total_training_steps:,} steps, min_lr={config.lr * config.min_lr_ratio:.3g}",
        flush=True,
    )

    metrics_path = metrics_dir / "losses.csv"
    last_train = {}
    last_val = {}
    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        last_train = run_loader(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp=config.amp,
            grad_clip=config.grad_clip,
            action_block_dim=store.action_block_dim,
            max_actions=store.max_actions,
            padded_action_weight=config.padded_action_weight,
            target_progress_weight=config.target_progress_weight,
            target_recall_radius=config.target_recall_radius,
            max_batches=config.max_train_batches,
            description=f"train {epoch:03d}",
            validation_seed=None,
        )
        last_val = run_loader(
            model,
            val_loader,
            device,
            optimizer=None,
            scheduler=None,
            scaler=None,
            amp=config.amp,
            grad_clip=0.0,
            action_block_dim=store.action_block_dim,
            max_actions=store.max_actions,
            padded_action_weight=config.padded_action_weight,
            target_progress_weight=config.target_progress_weight,
            target_recall_radius=config.target_recall_radius,
            max_batches=config.max_val_batches,
            description=f"val {epoch:03d}",
            validation_seed=config.seed + 10_000,
        )
        elapsed = time.perf_counter() - epoch_started
        print(
            f"epoch={epoch:03d} train={last_train['loss']:.6f} "
            f"val={last_val['loss']:.6f} seconds={elapsed:.2f}",
            flush=True,
        )
        append_metrics(metrics_path, epoch, last_train, last_val)
        plot_losses(metrics_dir / "loss_curve.png", metrics_path)
        # Keep both a resumable best checkpoint and a compact planner checkpoint.
        if last_val["loss"] < best_val:
            best_val = last_val["loss"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
                best_validation_loss=best_val,
                best_epoch=best_epoch,
            )
            save_checkpoint(
                checkpoint_dir / "best_inference.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
                best_validation_loss=best_val,
                best_epoch=best_epoch,
                inference_only=True,
            )

        if config.checkpoint_every and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
                best_validation_loss=best_val,
                best_epoch=best_epoch,
            )
        # Periodic encoded samples are diagnostic artifacts, not planner paths.
        if config.sample_every and epoch % config.sample_every == 0:
            cond = torch.as_tensor(store.conds["val"][:4], device=device)
            sample_generator = torch.Generator(device=device)
            sample_generator.manual_seed(config.seed + 20_000 + epoch)
            generated = sample_edge_sets(
                model,
                cond,
                steps=config.sample_steps,
                set_size=store.set_size,
                edge_dim=store.edge_dim,
                generator=sample_generator,
            )
            np.savez_compressed(
                samples_dir / f"epoch_{epoch:04d}.npz",
                cond=cond.cpu().numpy(),
                edges=generated.cpu().numpy(),
            )
        status.update(
            {
                "last_completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val,
                "latest_train_loss": last_train["loss"],
                "latest_validation_loss": last_val["loss"],
                "updated_at": utc_isoformat(),
            }
        )
        atomic_write_json(run_dir / "status.json", status)

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        metadata=store.metadata,
        epoch=config.epochs,
        train_metrics=last_train,
        val_metrics=last_val,
        best_validation_loss=best_val,
        best_epoch=best_epoch,
    )
    total_seconds = time.perf_counter() - training_started
    inference_checkpoint = checkpoint_dir / "best_inference.pt"
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "completed_epochs": config.epochs,
                "best_validation_loss": best_val,
                "best_epoch": best_epoch,
                "total_training_seconds": total_seconds,
                "device": str(device),
                "dataset_sha256": store.metadata["dataset_sha256"],
                "checkpoint": str(inference_checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(inference_checkpoint),
            },
            stream,
            indent=2,
        )
    print(f"Finished in {total_seconds:.1f}s; artifacts: {run_dir}", flush=True)



def main() -> int:
    """Create or resume one managed run and capture its complete console output."""
    args = parse_args()
    apply_resume_configuration(args)
    device = resolve_device(args.device)
    run_dir, resumed = resolve_run_directory(args)
    for child in ("checkpoints", "metrics", "samples", "provenance"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    session_id = utc_timestamp()
    session_dir = run_dir / "provenance" / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    command = shlex.join([sys.executable, *sys.argv])
    (session_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    write_git_diff(session_dir / "git_diff.patch")

    cli_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    session = {
        "session_id": session_id,
        "started_at": utc_isoformat(),
        "resumed": resumed,
        "resume_checkpoint": (
            None if args.resume is None else str(args.resume.resolve())
        ),
        "command": command,
        "cli_arguments": cli_arguments,
        "runtime": runtime_provenance(device),
        "provenance_directory": str(session_dir.resolve()),
    }
    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    else:
        manifest = {
            "format_version": 1,
            "run_id": run_dir.name,
            "description": args.description or args.experiment_name or "baseline",
            "created_at": session["started_at"],
            "sessions": [],
        }
    manifest.setdefault("sessions", []).append(session)
    atomic_write_json(manifest_path, manifest)

    status_path = run_dir / "status.json"
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as stream:
            status = json.load(stream)
    else:
        status = {"run_id": run_dir.name}
    status.update(
        {
            "state": "running",
            "active_session": session_id,
            "started_at": session["started_at"],
            "updated_at": utc_isoformat(),
        }
    )
    status.pop("completed_at", None)
    status.pop("failed_at", None)
    status.pop("interrupted_at", None)
    atomic_write_json(status_path, status)

    log_path = run_dir / "console.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(original_stdout, log_stream)
    sys.stderr = TeeStream(original_stderr, log_stream)
    exit_code = 0
    try:
        print(
            f"Training session {session_id} ({'resume' if resumed else 'fresh'})",
            flush=True,
        )
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Command: {command}", flush=True)
        run_training(args, device, run_dir, status)
        status.update(
            {
                "state": "completed",
                "completed_at": utc_isoformat(),
                "updated_at": utc_isoformat(),
            }
        )
    except KeyboardInterrupt:
        traceback.print_exc()
        status.update(
            {
                "state": "interrupted",
                "interrupted_at": utc_isoformat(),
                "updated_at": utc_isoformat(),
            }
        )
        exit_code = 130
    except BaseException:
        traceback.print_exc()
        status.update(
            {
                "state": "failed",
                "failed_at": utc_isoformat(),
                "updated_at": utc_isoformat(),
            }
        )
        exit_code = 1
    finally:
        with manifest_path.open("r", encoding="utf-8") as stream:
            final_manifest = json.load(stream)
        for recorded_session in final_manifest.get("sessions", []):
            if recorded_session.get("session_id") == session_id:
                recorded_session.update(
                    {
                        "state": status["state"],
                        "ended_at": utc_isoformat(),
                        "exit_code": exit_code,
                    }
                )
                break
        atomic_write_json(manifest_path, final_manifest)
        atomic_write_json(status_path, status)
        print(f"Run state: {status['state']}", flush=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_stream.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
