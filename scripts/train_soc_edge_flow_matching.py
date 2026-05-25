#!/usr/bin/env python3
"""Train a conditional flow-matching model for SOC edge-set generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "soc_edge_set_flow_matching_k32_n100000.npz"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "checkpoints" / "soc_edge_flow"
DEFAULT_DEVICE = "cuda:1"


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
    consistency_loss_weight: float
    consistency_rollout_steps: int
    debug: bool
    max_train_batches: int | None
    max_val_batches: int | None


class SOCEdgeSetDataset(Dataset):
    def __init__(self, path: Path, split: str):
        data = np.load(path, allow_pickle=False)
        if split == "train":
            self.conds = data["conds_train"].astype(np.float32)
            self.edges = data["edges_train"].astype(np.float32)
        elif split == "val":
            self.conds = data["conds_val"].astype(np.float32)
            self.edges = data["edges_val"].astype(np.float32)
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return self.conds.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.conds[idx]), torch.from_numpy(self.edges[idx])


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class EdgeSetFlowModel(nn.Module):
    def __init__(
        self,
        *,
        edge_dim: int = 8,
        cond_dim: int = 2,
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
        self.edge_dim = edge_dim
        self.cond_dim = cond_dim
        self.set_size = set_size

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, hidden_dim),
        )
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.slot_embed = nn.Parameter(torch.zeros(1, set_size, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_dim),
        )

        nn.init.trunc_normal_(self.slot_embed, std=0.02)

    def forward(self, noisy_edges: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if noisy_edges.ndim != 3:
            raise ValueError(f"Expected noisy_edges shape (B,K,D), got {tuple(noisy_edges.shape)}")
        if noisy_edges.shape[1] != self.set_size:
            raise ValueError(f"Expected set size {self.set_size}, got {noisy_edges.shape[1]}")

        t = t.reshape(-1).to(noisy_edges.dtype)
        h = self.edge_in(noisy_edges)
        global_token = self.time_embed(t) + self.cond_embed(cond)
        h = h + global_token[:, None, :] + self.slot_embed
        h = self.backbone(h)
        return self.out(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--cond-embed-dim", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument(
        "--consistency-loss-weight",
        type=float,
        default=0.01,
        help="Weight for bounded differentiable SOC rollout consistency on the estimated clean edge.",
    )
    parser.add_argument(
        "--consistency-rollout-steps",
        type=int,
        default=20,
        help="Fixed RK4 substeps for the differentiable consistency rollout.",
    )
    parser.add_argument("--debug", action="store_true", help="Run a tiny CPU-friendly training smoke test.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested_device}, but CUDA is not available.")
        if ":" in requested_device:
            index = int(requested_device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {requested_device}, but only {torch.cuda.device_count()} CUDA devices are visible."
                )
    return requested_device


def load_metadata(dataset_path: Path) -> dict:
    data = np.load(dataset_path, allow_pickle=False)
    return json.loads(str(data["metadata_json"]))


def soc_dynamics(
    state: torch.Tensor,
    control: torch.Tensor,
    wheelbase: float = 0.7,
    max_speed: float = 1.0,
    max_phi: float = math.pi / 3.0,
) -> torch.Tensor:
    theta = state[..., 2]
    v = state[..., 3].clamp(-max_speed, max_speed)
    phi = state[..., 4].clamp(-max_phi, max_phi)
    acceleration = control[..., 0]
    steering_rate = control[..., 1]
    return torch.stack(
        [
            v * torch.cos(theta),
            v * torch.sin(theta),
            v * torch.tan(phi) / wheelbase,
            acceleration,
            steering_rate,
        ],
        dim=-1,
    )


def soc_rk4_step(
    state: torch.Tensor,
    control: torch.Tensor,
    dt: torch.Tensor,
    max_speed: float,
    max_phi: float,
) -> torch.Tensor:
    k1 = soc_dynamics(state, control, max_speed=max_speed, max_phi=max_phi)
    k2 = soc_dynamics(state + 0.5 * dt * k1, control, max_speed=max_speed, max_phi=max_phi)
    k3 = soc_dynamics(state + 0.5 * dt * k2, control, max_speed=max_speed, max_phi=max_phi)
    k4 = soc_dynamics(state + dt * k3, control, max_speed=max_speed, max_phi=max_phi)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return torch.stack(
        [
            next_state[..., 0],
            next_state[..., 1],
            next_state[..., 2],
            next_state[..., 3].clamp(-max_speed, max_speed),
            next_state[..., 4].clamp(-max_phi, max_phi),
        ],
        dim=-1,
    )


def rollout_consistency_loss(
    clean_edges_estimate: torch.Tensor,
    cond: torch.Tensor,
    metadata: dict,
    rollout_steps: int,
) -> torch.Tensor:
    norm = metadata["normalization"]
    max_acceleration = float(norm["max_acceleration"])
    max_steering_rate = float(norm["max_steering_rate"])
    max_timestep = float(norm["max_timestep"])
    max_speed = float(norm["max_speed"])
    max_phi = float(norm["max_phi"])
    dx_scale = float(norm["dx_scale"])
    dy_scale = float(norm["dy_scale"])
    dtheta_scale = float(norm["dtheta_scale"])

    bounded_edges = clean_edges_estimate.tanh()
    control = torch.stack(
        [
            bounded_edges[..., 0] * max_acceleration,
            bounded_edges[..., 1] * max_steering_rate,
        ],
        dim=-1,
    )
    timestep = bounded_edges[..., 2].add(1.0).mul(0.5).clamp(0.05, 1.0) * max_timestep

    batch_size, set_size, _ = clean_edges_estimate.shape
    state = torch.zeros(batch_size, set_size, 5, device=clean_edges_estimate.device, dtype=clean_edges_estimate.dtype)
    state[..., 3] = cond[:, None, 0] * max_speed
    state[..., 4] = cond[:, None, 1] * max_phi

    dt = (timestep / float(rollout_steps))[..., None]
    for _ in range(rollout_steps):
        state = soc_rk4_step(state, control, dt, max_speed, max_phi)

    rolled_normalized = torch.stack(
        [
            state[..., 0] / dx_scale,
            state[..., 1] / dy_scale,
            state[..., 2] / dtheta_scale,
            state[..., 3] / max_speed,
            state[..., 4] / max_phi,
        ],
        dim=-1,
    )
    predicted_final = bounded_edges[..., 3:8]
    return F.smooth_l1_loss(predicted_final, rolled_normalized, beta=0.1)


def flow_matching_loss(
    model: nn.Module,
    edges: torch.Tensor,
    cond: torch.Tensor,
    metadata: dict | None = None,
    consistency_loss_weight: float = 0.0,
    consistency_rollout_steps: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(edges)
    t = torch.rand(edges.shape[0], device=edges.device, dtype=edges.dtype)
    t_view = t[:, None, None]
    noisy_edges = (1.0 - t_view) * noise + t_view * edges
    target_velocity = edges - noise
    pred_velocity = model(noisy_edges, t, cond)
    fm_loss = F.mse_loss(pred_velocity, target_velocity)

    consistency_loss = torch.zeros((), device=edges.device, dtype=edges.dtype)
    if metadata is not None and consistency_loss_weight > 0.0:
        clean_edges_estimate = noisy_edges + (1.0 - t_view) * pred_velocity
        consistency_loss = rollout_consistency_loss(
            clean_edges_estimate,
            cond,
            metadata,
            consistency_rollout_steps,
        )

    total_loss = fm_loss + consistency_loss_weight * consistency_loss
    return total_loss, fm_loss.detach(), consistency_loss.detach()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metadata: dict,
    consistency_loss_weight: float,
    consistency_rollout_steps: int,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_fm_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0
    for batch_idx, (cond, edges) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        cond = cond.to(device)
        edges = edges.to(device)
        loss, fm_loss, consistency_loss = flow_matching_loss(
            model,
            edges,
            cond,
            metadata,
            consistency_loss_weight,
            consistency_rollout_steps,
        )
        total_loss += float(loss.item())
        total_fm_loss += float(fm_loss.item())
        total_consistency_loss += float(consistency_loss.item())
        total_batches += 1
    denom = max(total_batches, 1)
    return total_loss / denom, total_fm_loss / denom, total_consistency_loss / denom


@torch.no_grad()
def sample_edges(
    model: nn.Module,
    cond: torch.Tensor,
    *,
    steps: int,
    edge_dim: int,
    set_size: int,
) -> torch.Tensor:
    model.eval()
    edges = torch.randn(cond.shape[0], set_size, edge_dim, device=cond.device)
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((cond.shape[0],), step / steps, device=cond.device, dtype=edges.dtype)
        velocity = model(edges, t, cond)
        edges = edges + dt * velocity
    return edges


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metadata: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "dataset_metadata": metadata,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def append_loss_row(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_fm_loss: float,
    val_fm_loss: float,
    train_consistency_loss: float,
    val_consistency_loss: float,
) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_fm_loss",
                "val_fm_loss",
                "train_consistency_loss",
                "val_consistency_loss",
            ])
        writer.writerow([
            epoch,
            f"{train_loss:.10f}",
            f"{val_loss:.10f}",
            f"{train_fm_loss:.10f}",
            f"{val_fm_loss:.10f}",
            f"{train_consistency_loss:.10f}",
            f"{val_consistency_loss:.10f}",
        ])


def plot_loss_curve(path: Path, epochs: list[int], train_losses: list[float], val_losses: list[float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="train", linewidth=2.0)
    ax.plot(epochs, val_losses, label="val", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("flow matching MSE")
    ax.set_title("SOC edge-set flow matching loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    metadata: dict,
    consistency_loss_weight: float,
    consistency_rollout_steps: int,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_fm_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, (cond, edges) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        cond = cond.to(device)
        edges = edges.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, fm_loss, consistency_loss = flow_matching_loss(
            model,
            edges,
            cond,
            metadata,
            consistency_loss_weight,
            consistency_rollout_steps,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_fm_loss += float(fm_loss.item())
        total_consistency_loss += float(consistency_loss.item())
        total_batches += 1
        progress.set_postfix(
            loss=f"{loss.item():.5f}",
            fm=f"{fm_loss.item():.5f}",
            dyn=f"{consistency_loss.item():.5f}",
        )
    denom = max(total_batches, 1)
    return total_loss / denom, total_fm_loss / denom, total_consistency_loss / denom


def main() -> None:
    args = parse_args()
    device_name = resolve_device(args.device)
    if args.debug:
        args.batch_size = min(args.batch_size, 16)
        args.epochs = min(args.epochs, 1)
        args.hidden_dim = min(args.hidden_dim, 64)
        args.depth = min(args.depth, 2)
        args.num_heads = min(args.num_heads, 4)
        args.num_workers = 0
        args.max_train_batches = args.max_train_batches or 2
        args.max_val_batches = args.max_val_batches or 1
        args.sample_every = 1
        args.checkpoint_every = 1
        args.sample_steps = min(args.sample_steps, 4)

    experiment_name = args.experiment_name or f"soc_edge_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = TrainConfig(
        dataset=str(Path(args.dataset).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        experiment_name=experiment_name,
        seed=args.seed,
        device=device_name,
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
        consistency_loss_weight=args.consistency_loss_weight,
        consistency_rollout_steps=args.consistency_rollout_steps,
        debug=args.debug,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    set_seed(config.seed)
    dataset_path = Path(config.dataset)
    metadata = load_metadata(dataset_path)
    edge_dim = int(metadata["edge_dim"])
    cond_dim = int(metadata["cond_dim"])
    set_size = int(metadata["set_size"])

    train_dataset = SOCEdgeSetDataset(dataset_path, "train")
    val_dataset = SOCEdgeSetDataset(dataset_path, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device_name.startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device_name.startswith("cuda"),
    )

    device = torch.device(device_name)
    model = EdgeSetFlowModel(
        edge_dim=edge_dim,
        cond_dim=cond_dim,
        set_size=set_size,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        time_embed_dim=config.time_embed_dim,
        cond_embed_dim=config.cond_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "dataset_metadata": metadata}, f, indent=2)

    print(f"Training {config.experiment_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Train/val: {len(train_dataset)} / {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    best_val = float("inf")
    loss_csv = run_dir / "losses.csv"
    loss_plot = run_dir / "loss_curve.png"
    loss_epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_fm_loss, train_consistency_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.grad_clip,
            metadata,
            config.consistency_loss_weight,
            config.consistency_rollout_steps,
            max_batches=config.max_train_batches,
        )
        val_loss, val_fm_loss, val_consistency_loss = evaluate(
            model,
            val_loader,
            device,
            metadata,
            config.consistency_loss_weight,
            config.consistency_rollout_steps,
            max_batches=config.max_val_batches,
        )
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_fm={train_fm_loss:.6f} val_fm={val_fm_loss:.6f} "
            f"train_dyn={train_consistency_loss:.6f} val_dyn={val_consistency_loss:.6f}"
        )
        loss_epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        append_loss_row(
            loss_csv,
            epoch,
            train_loss,
            val_loss,
            train_fm_loss,
            val_fm_loss,
            train_consistency_loss,
            val_consistency_loss,
        )
        plot_loss_curve(loss_plot, loss_epochs, train_losses, val_losses)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=metadata,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=metadata,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        if config.sample_every > 0 and epoch % config.sample_every == 0:
            cond = torch.as_tensor(val_dataset.conds[:4], device=device)
            sampled = sample_edges(
                model,
                cond,
                steps=config.sample_steps,
                edge_dim=edge_dim,
                set_size=set_size,
            )
            np.savez_compressed(
                run_dir / f"samples_epoch_{epoch:04d}.npz",
                cond=cond.detach().cpu().numpy(),
                edges=sampled.detach().cpu().numpy(),
            )

    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        metadata=metadata,
        epoch=config.epochs,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    print(f"Saved checkpoints to {run_dir}")


if __name__ == "__main__":
    main()
