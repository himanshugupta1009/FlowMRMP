#!/usr/bin/env python3
"""Train conditional flow matching on variable-length Franka edge bundles.

The HDF5 file remains the source of truth: every bundle contains references to
32 trajectory segments containing at most 50 executed acceleration intervals.
Each segment is encoded as a fixed vector containing zero-padded accelerations,
normalized step count, relative configuration change, and relative velocity
change. Padding is an encoding detail; the decoded planner edge retains only
its generated number of executed steps.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT_DIR
    / "data"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "checkpoints" / "franka_edge_flow"

ACTION_DIM = 7
DEFAULT_MAX_ACTIONS = 50


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(path: Path) -> dict[str, object]:
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


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    experiment_name: str
    seed: int
    device: str
    batch_size: int
    epochs: int
    lr: float
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
    max_train_batches: int | None
    max_val_batches: int | None
    resume: str | None
    debug: bool


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t[:, None] * frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class EdgeSetFlowModel(nn.Module):
    """The same conditional 32-slot Transformer used for the SOC model."""

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
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.cond_dim = int(cond_dim)
        self.set_size = int(set_size)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # A 1,639D Franka edge cannot be losslessly compressed through the
        # hidden token just to reproduce the coordinate-wise noise term.  This
        # time-conditioned residual preserves every noisy input coordinate;
        # the Transformer learns the lower-dimensional structured correction.
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
        if noisy_edges.ndim != 3:
            raise ValueError(f"Expected (B,K,D), received {tuple(noisy_edges.shape)}")
        if noisy_edges.shape[1:] != (self.set_size, self.edge_dim):
            raise ValueError(
                f"Expected (*,{self.set_size},{self.edge_dim}), received "
                f"{tuple(noisy_edges.shape)}"
            )
        h = self.edge_in(noisy_edges)
        time_token = self.time_embed(t.reshape(-1).to(noisy_edges.dtype))
        skip_scale = self.skip_scale(time_token)[:, None, :]
        global_token = time_token + self.cond_embed(cond)
        h = h + global_token[:, None, :] + self.slot_embed
        return skip_scale * noisy_edges + self.out(self.backbone(h))


class FrankaEdgeBundleStore:
    """RAM-backed decoder for the HDF5 reference representation."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        started = time.perf_counter()
        with h5py.File(self.path, "r") as source:
            if source.attrs.get("format_name") != "franka_variable_length_edge_bundle":
                raise ValueError(f"Not a Franka edge-bundle dataset: {self.path}")
            metadata_value = source["metadata_json"][()]
            if isinstance(metadata_value, bytes):
                metadata_value = metadata_value.decode("utf-8")
            self.metadata = json.loads(str(metadata_value))
            self.conds = {
                split: source[f"conds_{split}"][:].astype(np.float32, copy=False)
                for split in ("train", "val")
            }
            self.bundle_ids = {
                split: source[f"source_edge_ids_{split}"][:]
                for split in ("train", "val")
            }
            outcomes = source["bundle_outcomes"]
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
            raw = source["raw_edges"]
            self.raw_trajectory = raw["trajectory_index"][:]
            self.raw_start = raw["start_index"][:]
            self.raw_num_steps = raw["num_steps"][:]
            self.raw_length = raw["trajectory_length"][:]

            names = source["source_trajectories/trajectory_names"][:]
            decoded_names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in names
            ]
            trajectory_lengths = np.asarray(
                [
                    source["source_trajectories"][name]["accelerations"].shape[0]
                    for name in decoded_names
                ],
                dtype=np.int64,
            )
            self.trajectory_offsets = np.zeros(len(decoded_names), dtype=np.int64)
            if len(decoded_names) > 1:
                self.trajectory_offsets[1:] = np.cumsum(trajectory_lengths[:-1])
            self.accelerations = np.empty(
                (int(trajectory_lengths.sum()), ACTION_DIM), dtype=np.float32
            )
            for index, (name, length) in enumerate(
                zip(decoded_names, trajectory_lengths)
            ):
                offset = int(self.trajectory_offsets[index])
                self.accelerations[offset : offset + int(length)] = source[
                    "source_trajectories"
                ][name]["accelerations"][:]

        norm = self.metadata["normalization"]
        self.q_range = np.asarray(norm["q_upper"], dtype=np.float32) - np.asarray(
            norm["q_lower"], dtype=np.float32
        )
        self.dq_max = np.asarray(norm["dq_max_abs"], dtype=np.float32)
        self.ddq_max = np.asarray(norm["ddq_max_abs"], dtype=np.float32)
        self.dt = float(self.metadata["dt"])
        self.max_duration = float(self.metadata["max_suffix_duration"])
        self.set_size = int(self.metadata["set_size"])
        self.max_actions = int(np.max(self.raw_num_steps))
        if self.max_actions != DEFAULT_MAX_ACTIONS:
            raise ValueError(
                f"Expected {DEFAULT_MAX_ACTIONS} maximum actions, found {self.max_actions}"
            )
        if not np.array_equal(self.raw_length, self.raw_num_steps + 1):
            raise ValueError("trajectory_length must equal num_steps + 1")
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
                "cond_dim": 14,
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
                        "lexicographic normalized delta_q, then normalized num_steps"
                    ),
                },
            }
        )
        self.load_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return len(self.conds["train"]) + len(self.conds["val"])

    def encode_batch(
        self, split: str, sample_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        edge_ids = self.bundle_ids[split][sample_indices]
        batch_size = edge_ids.shape[0]
        action_counts = self.bundle_num_steps[split][sample_indices].astype(
            np.int64
        )
        starts = (
            self.trajectory_offsets[self.raw_trajectory[edge_ids]]
            + self.raw_start[edge_ids]
        )
        steps = np.arange(self.max_actions, dtype=np.int64)[None, None, :]
        action_mask = steps < action_counts[:, :, None]
        safe_steps = np.minimum(steps, action_counts[:, :, None] - 1)
        pool_indices = starts[:, :, None] + safe_steps
        actions = self.accelerations[pool_indices]
        actions = actions / self.ddq_max[None, None, None, :]
        actions *= action_mask[:, :, :, None]

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

        # The source file keeps deterministic raw-edge-ID order.  Reorder the
        # same members geometrically so learned slots have consistent meaning.
        keys = [step_count]
        keys.extend(delta_q[:, :, joint] for joint in range(6, -1, -1))
        order = np.lexsort(tuple(keys), axis=1)
        edges = np.take_along_axis(edges, order[:, :, None], axis=1)
        action_mask = np.take_along_axis(action_mask, order[:, :, None], axis=1)
        action_counts = np.take_along_axis(action_counts, order, axis=1)
        cond = self.conds[split][sample_indices]
        return cond, edges, action_mask, action_counts


class IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return int(index)


class EncodeBatch:
    def __init__(self, store: FrankaEdgeBundleStore, split: str):
        self.store = store
        self.split = split

    def __call__(self, indices: list[int]):
        cond, edges, mask, counts = self.store.encode_batch(
            self.split, np.asarray(indices, dtype=np.int64)
        )
        return (
            torch.from_numpy(cond),
            torch.from_numpy(edges),
            torch.from_numpy(mask),
            torch.from_numpy(counts),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    parser.add_argument("--checkpoint-every", type=int, default=10)
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
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flow_matching_loss(
    model: nn.Module,
    edges: torch.Tensor,
    cond: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_block_dim: int,
    max_actions: int,
    padded_action_weight: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    noisy_edges = (1.0 - t_view) * noise + t_view * edges
    target = edges - noise
    prediction = model(noisy_edges, t, cond)
    squared = (prediction - target).square()

    action_error = squared[:, :, :action_block_dim].reshape(
        edges.shape[0], edges.shape[1], max_actions, ACTION_DIM
    )
    valid_mask = action_mask[:, :, :, None].to(action_error.dtype)
    valid_count = valid_mask.sum(dim=(2, 3)) * ACTION_DIM
    valid_action_per_edge = (action_error * valid_mask).sum(dim=(2, 3)) / (
        valid_count.clamp_min(1.0)
    )
    action_loss = valid_action_per_edge.mean()

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
    total = (
        action_loss + step_count_loss + delta_q_loss + delta_dq_loss
    ) / 4.0 + padded_action_weight * padding_loss
    return total, {
        "action": action_loss.detach(),
        "padding": padding_loss.detach(),
        "step_count": step_count_loss.detach(),
        "delta_q": delta_q_loss.detach(),
        "delta_dq": delta_dq_loss.detach(),
    }


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    action_block_dim: int,
    max_actions: int,
    padded_action_weight: float,
    max_batches: int | None,
    description: str,
    validation_seed: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        key: 0.0
        for key in (
            "loss",
            "action",
            "padding",
            "step_count",
            "delta_q",
            "delta_dq",
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
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, blocks = flow_matching_loss(
                model,
                edges,
                cond,
                action_mask,
                action_block_dim=action_block_dim,
                max_actions=max_actions,
                padded_action_weight=padded_action_weight,
                generator=generator,
            )
            if optimizer is not None:
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
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
) -> torch.Tensor:
    model.eval()
    edges = torch.randn(cond.shape[0], set_size, edge_dim, device=cond.device)
    dt = 1.0 / float(steps)
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
    config: TrainConfig,
    metadata: dict,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    inference_only: bool = False,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "dataset_metadata": metadata,
        "epoch": int(epoch),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    if not inference_only:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def append_metrics(path: Path, epoch: int, train: dict, val: dict) -> None:
    fields = [
        "loss",
        "action",
        "padding",
        "step_count",
        "delta_q",
        "delta_dq",
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
    for block in ("action", "padding", "step_count", "delta_q", "delta_dq"):
        axes[1].plot(values["epoch"], values[f"val_{block}"], label=block)
    axes[1].set(title="Validation loss by feature block", xlabel="Epoch", ylabel="MSE")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_workers != 0:
        raise ValueError("Use --num-workers 0; the RAM-backed store must not be duplicated")
    if args.padded_action_weight < 0.0:
        raise ValueError("padded-action-weight must be nonnegative")
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

    set_seed(args.seed)
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
    experiment_name = args.experiment_name or (
        f"franka_edge_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config = TrainConfig(
        dataset=str(args.dataset.resolve()),
        output_dir=str(args.output_dir.resolve()),
        experiment_name=experiment_name,
        seed=args.seed,
        device=str(device),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
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
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=None if args.resume is None else str(args.resume.resolve()),
        debug=args.debug,
    )

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
    model = EdgeSetFlowModel(
        edge_dim=store.edge_dim,
        cond_dim=14,
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

    run_dir = Path(config.output_dir) / config.experiment_name
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("val_metrics", {}).get("loss", float("inf")))
        best_epoch = int(checkpoint["epoch"])
        run_dir = args.resume.resolve().parent
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump({"config": asdict(config), "dataset_metadata": store.metadata}, stream, indent=2)

    print(f"Training: {config.experiment_name}", flush=True)
    print(f"Dataset: {store.path}", flush=True)
    print(f"RAM data store load: {store.load_seconds:.3f}s", flush=True)
    print(f"Train/validation: {len(store.conds['train']):,}/{len(store.conds['val']):,}", flush=True)
    print(f"Edge tensor: (32, {store.edge_dim})", flush=True)
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    print(f"Device: {device}", flush=True)

    metrics_path = run_dir / "losses.csv"
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
            grad_clip=config.grad_clip,
            action_block_dim=store.action_block_dim,
            max_actions=store.max_actions,
            padded_action_weight=config.padded_action_weight,
            max_batches=config.max_train_batches,
            description=f"train {epoch:03d}",
            validation_seed=None,
        )
        last_val = run_loader(
            model,
            val_loader,
            device,
            optimizer=None,
            grad_clip=0.0,
            action_block_dim=store.action_block_dim,
            max_actions=store.max_actions,
            padded_action_weight=config.padded_action_weight,
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
        plot_losses(run_dir / "loss_curve.png", metrics_path)

        if last_val["loss"] < best_val:
            best_val = last_val["loss"]
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
            )
            save_checkpoint(
                run_dir / "best_inference.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
                inference_only=True,
            )

        if config.checkpoint_every and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=store.metadata,
                epoch=epoch,
                train_metrics=last_train,
                val_metrics=last_val,
            )
        if config.sample_every and epoch % config.sample_every == 0:
            cond = torch.as_tensor(store.conds["val"][:4], device=device)
            generated = sample_edge_sets(
                model,
                cond,
                steps=config.sample_steps,
                set_size=store.set_size,
                edge_dim=store.edge_dim,
            )
            np.savez_compressed(
                run_dir / f"samples_epoch_{epoch:04d}.npz",
                cond=cond.cpu().numpy(),
                edges=generated.cpu().numpy(),
            )

    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        metadata=store.metadata,
        epoch=config.epochs,
        train_metrics=last_train,
        val_metrics=last_val,
    )
    total_seconds = time.perf_counter() - training_started
    inference_checkpoint = run_dir / "best_inference.pt"
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


if __name__ == "__main__":
    main()
