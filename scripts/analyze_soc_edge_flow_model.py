#!/usr/bin/env python3
"""Analyze a trained SOC edge-set flow-matching checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MRMP_SRC))

from scripts.train_soc_edge_flow_matching import EdgeSetFlowModel, sample_edges  # noqa: E402

try:
    from Agents.SecondOrderCar import SecondOrderCar  # noqa: E402
except Exception:  # pragma: no cover - analysis still works without dynamics import.
    SecondOrderCar = None


DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "soc_edge_flow" / "soc_edge_flow_k32_v1" / "best.pt"
DEFAULT_DATASET = ROOT_DIR / "data" / "soc_edge_set_flow_matching_k32_n100000.npz"
DEFAULT_BUNDLE = (
    ROOT_DIR
    / "mrmp_with_kite_extend"
    / "edge_bundles_unclamped"
    / "eb_second_order_car_kinodynamic_TI_edges_100000.npz"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analysis" / "soc_edge_flow_k32_v1"
EDGE_FIELDS = ["a", "steering_rate", "T", "dx", "dy", "dtheta", "v_final", "phi_final"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-conditions", type=int, default=512)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--speed-batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--speed-steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available.")
        if ":" in device:
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible.")
    return torch.device(device)


def load_metadata(dataset_path: Path) -> dict:
    data = np.load(dataset_path, allow_pickle=False)
    return json.loads(str(data["metadata_json"]))


def load_model(checkpoint_path: Path, dataset_metadata: dict, device: torch.device) -> EdgeSetFlowModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = EdgeSetFlowModel(
        edge_dim=int(dataset_metadata["edge_dim"]),
        cond_dim=int(dataset_metadata["cond_dim"]),
        set_size=int(dataset_metadata["set_size"]),
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        time_embed_dim=int(config["time_embed_dim"]),
        cond_embed_dim=int(config["cond_embed_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def denormalize_edges(edges: np.ndarray, norm: dict) -> np.ndarray:
    out = edges.copy()
    out[..., 0] *= norm["max_acceleration"]
    out[..., 1] *= norm["max_steering_rate"]
    out[..., 2] *= norm["max_timestep"]
    out[..., 3] *= norm["dx_scale"]
    out[..., 4] *= norm["dy_scale"]
    out[..., 5] *= norm["dtheta_scale"]
    out[..., 6] *= norm["max_speed"]
    out[..., 7] *= norm["max_phi"]
    return out


def denormalize_conditions(conds: np.ndarray, norm: dict) -> np.ndarray:
    out = conds.copy()
    out[..., 0] *= norm["max_speed"]
    out[..., 1] *= norm["max_phi"]
    return out


def summarize_array(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def distribution_metrics(generated: np.ndarray, reference: np.ndarray) -> dict:
    generated_flat = generated.reshape(-1, generated.shape[-1])
    reference_flat = reference.reshape(-1, reference.shape[-1])
    metrics = {}
    for i, name in enumerate(EDGE_FIELDS):
        gen = generated_flat[:, i]
        ref = reference_flat[:, i]
        metrics[name] = {
            "generated": summarize_array(gen),
            "reference": summarize_array(ref),
            "mean_abs_shift": float(abs(np.mean(gen) - np.mean(ref))),
            "std_abs_shift": float(abs(np.std(gen) - np.std(ref))),
        }
    metrics["normalized_out_of_soft_range_fraction"] = float(np.mean(np.abs(generated_flat) > 1.25))
    metrics["normalized_out_of_hard_range_fraction"] = float(np.mean(np.abs(generated_flat) > 2.0))
    return metrics


def diversity_metrics(edge_sets: np.ndarray) -> dict:
    min_distances = []
    mean_pair_distances = []
    for edge_set in edge_sets:
        diff = edge_set[:, None, :] - edge_set[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist += np.eye(dist.shape[0]) * 1e9
        min_distances.append(float(np.min(dist)))
        mean_pair_distances.append(float(np.mean(dist[dist < 1e8])))
    return {
        "mean_min_pairwise_distance": float(np.mean(min_distances)),
        "p05_min_pairwise_distance": float(np.percentile(min_distances, 5)),
        "mean_pairwise_distance": float(np.mean(mean_pair_distances)),
    }


def nearest_neighbor_metrics(generated: np.ndarray, reference: np.ndarray) -> dict:
    tree = cKDTree(reference.reshape(-1, reference.shape[-1]))
    distances, _ = tree.query(generated.reshape(-1, generated.shape[-1]), k=1)
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def rollout_generated_edges(conds: np.ndarray, generated_denorm: np.ndarray, norm: dict) -> np.ndarray | None:
    if SecondOrderCar is None:
        return None

    agent = SecondOrderCar(
        max_speed=norm["max_speed"],
        max_acceleration=norm["max_acceleration"],
        max_phi=norm["max_phi"],
        max_steering_rate=norm["max_steering_rate"],
    )
    conds_denorm = denormalize_conditions(conds, norm)
    rolled = np.empty(generated_denorm[..., 3:8].shape, dtype=np.float64)
    for i in range(generated_denorm.shape[0]):
        start_state = np.array([0.0, 0.0, 0.0, conds_denorm[i, 0], conds_denorm[i, 1]], dtype=np.float64)
        for j in range(generated_denorm.shape[1]):
            action = generated_denorm[i, j, :2].astype(np.float64)
            timestep = float(max(generated_denorm[i, j, 2], 0.01))
            num_steps = max(1, int(round(timestep / 0.1)))
            final_state, _ = agent.get_next_state(start_state, action, timestep, num_steps)
            rolled[i, j] = final_state[[0, 1, 2, 3, 4]]
    return rolled


def dynamics_metrics(generated_denorm: np.ndarray, rolled_denorm: np.ndarray | None) -> dict:
    if rolled_denorm is None:
        return {"available": False}
    predicted = generated_denorm[..., 3:8]
    err = predicted - rolled_denorm
    norms = np.linalg.norm(err, axis=-1)
    metrics = {"available": True, "joint_error": summarize_array(norms)}
    for i, name in enumerate(["dx", "dy", "dtheta", "v_final", "phi_final"]):
        metrics[name] = summarize_array(np.abs(err[..., i]))
    return metrics


def benchmark_sampling(model: EdgeSetFlowModel, conds: np.ndarray, device: torch.device, batch_sizes: list[int], steps_list: list[int]) -> list[dict]:
    results = []
    for batch_size in batch_sizes:
        cond = np.resize(conds, (batch_size, conds.shape[1])).astype(np.float32)
        cond_t = torch.as_tensor(cond, device=device)
        for steps in steps_list:
            for _ in range(3):
                _ = sample_edges(model, cond_t, steps=steps, edge_dim=model.edge_dim, set_size=model.set_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            runs = 20
            for _ in range(runs):
                _ = sample_edges(model, cond_t, steps=steps, edge_dim=model.edge_dim, set_size=model.set_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) / runs
            results.append(
                {
                    "batch_size": batch_size,
                    "steps": steps,
                    "ms_per_batch": float(elapsed * 1e3),
                    "ms_per_condition": float(elapsed * 1e3 / batch_size),
                }
            )
    return results


def plot_histograms(output: Path, generated: np.ndarray, reference: np.ndarray) -> None:
    gen = generated.reshape(-1, generated.shape[-1])
    ref = reference.reshape(-1, reference.shape[-1])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        ax.hist(ref[:, i], bins=60, alpha=0.55, density=True, label="val")
        ax.hist(gen[:, i], bins=60, alpha=0.55, density=True, label="generated")
        ax.set_title(EDGE_FIELDS[i])
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_condition_rollouts(output_dir: Path, conds: np.ndarray, generated_denorm: np.ndarray, reference_denorm: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(8, conds.shape[0])
    for i in range(n):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ref = reference_denorm[i]
        gen = generated_denorm[i]
        ax.scatter(ref[:, 3], ref[:, 4], s=22, alpha=0.65, label="val final xy")
        ax.scatter(gen[:, 3], gen[:, 4], s=26, alpha=0.75, label="generated final xy")
        for row in gen:
            ax.plot([0.0, row[3]], [0.0, row[4]], color="tab:orange", alpha=0.25, linewidth=0.8)
        ax.scatter([0.0], [0.0], c="black", s=24)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"cond v_norm={conds[i,0]:.2f}, phi_norm={conds[i,1]:.2f}")
        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"condition_{i:02d}_rollouts.png", dpi=160)
        plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.debug:
        args.num_conditions = min(args.num_conditions, 32)
        args.sample_steps = min(args.sample_steps, 4)
        args.speed_batch_sizes = [1, 8]
        args.speed_steps = [2, 4]

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    dataset = np.load(dataset_path, allow_pickle=False)
    metadata = load_metadata(dataset_path)
    norm = metadata["normalization"]
    val_conds = dataset["conds_val"].astype(np.float32)
    val_edges = dataset["edges_val"].astype(np.float32)

    num_conditions = min(args.num_conditions, val_conds.shape[0])
    chosen = rng.choice(val_conds.shape[0], size=num_conditions, replace=False)
    conds = val_conds[chosen]
    reference_edges = val_edges[chosen]

    model = load_model(Path(args.checkpoint), metadata, device)
    cond_t = torch.as_tensor(conds, device=device)
    generated = sample_edges(
        model,
        cond_t,
        steps=args.sample_steps,
        edge_dim=int(metadata["edge_dim"]),
        set_size=int(metadata["set_size"]),
    ).detach().cpu().numpy()

    generated_denorm = denormalize_edges(generated, norm)
    reference_denorm = denormalize_edges(reference_edges, norm)
    rolled_denorm = rollout_generated_edges(conds, generated_denorm, norm)

    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(dataset_path.resolve()),
        "num_conditions": int(num_conditions),
        "sample_steps": int(args.sample_steps),
        "distribution_normalized": distribution_metrics(generated, reference_edges),
        "distribution_denormalized": distribution_metrics(generated_denorm, reference_denorm),
        "generated_diversity_normalized": diversity_metrics(generated),
        "reference_diversity_normalized": diversity_metrics(reference_edges),
        "nearest_neighbor_to_validation_normalized": nearest_neighbor_metrics(generated, reference_edges),
        "dynamics_consistency_denormalized": dynamics_metrics(generated_denorm, rolled_denorm),
        "speed_benchmark": benchmark_sampling(model, conds[: min(128, len(conds))], device, args.speed_batch_sizes, args.speed_steps),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        output_dir / "generated_vs_reference.npz",
        conds=conds,
        generated_normalized=generated,
        reference_normalized=reference_edges,
        generated_denormalized=generated_denorm,
        reference_denormalized=reference_denorm,
        rolled_denormalized=rolled_denorm if rolled_denorm is not None else np.array([]),
    )
    plot_histograms(output_dir / "histograms_normalized.png", generated, reference_edges)
    plot_histograms(output_dir / "histograms_denormalized.png", generated_denorm, reference_denorm)
    plot_condition_rollouts(output_dir / "condition_rollouts", conds, generated_denorm, reference_denorm)

    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
