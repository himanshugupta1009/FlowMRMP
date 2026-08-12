#!/usr/bin/env python3
"""Compare generated and held-out Franka edge-bundle distributions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import random
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_SCRIPTS_DIR = ROOT_DIR / "scripts"
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
for module_path in (TRAINING_SCRIPTS_DIR, MRMP_SRC):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from train_franka_edge_flow_matching import (
    ACTION_DIM,
    EdgeSetFlowModel,
    FrankaEdgeBundleStore,
    file_sha256,
    resolve_device,
    sample_edge_sets,
)


from FrankaPanda import FrankaSelfCollisionChecker  # noqa: E402


DEFAULT_DATASET = (
    ROOT_DIR
    / "data"
    / "franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5"
)
DEFAULT_CHECKPOINT = (
    ROOT_DIR
    / "checkpoints"
    / "franka_edge_flow"
    / "franka_edge_flow_k32_200k_pool128_n350000_max50_fullvalid_mps_v1"
    / "best_inference.pt"
)
DEFAULT_URDF = ROOT_DIR / "assets" / "robots" / "panda" / "panda.urdf"


def collision_asset_hashes(urdf_path: Path) -> dict[str, str]:
    robot_dir = urdf_path.resolve().parent
    paths = [urdf_path.resolve(), *sorted(robot_dir.glob("*.srdf"))]
    paths.extend(
        sorted(
            path
            for path in (robot_dir / "meshes" / "collision").rglob("*")
            if path.is_file()
        )
    )
    return {str(path.relative_to(ROOT_DIR)): file_sha256(path) for path in paths}


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "torch", "h5py", "pybullet", "matplotlib"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-name",
        default=None,
        help="Optional label for the automatically created run/evaluation directory.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--num-conditions", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--self-collision-edges", type=int, default=256)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=991)
    return parser.parse_args()


def default_evaluation_dir(
    checkpoint: Path,
    evaluation_name: str | None,
    *,
    num_conditions: int,
    sample_steps: int,
    seed: int,
) -> Path:
    """Return a unique evaluation directory alongside a managed training run."""
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name == "checkpoints":
        run_dir = checkpoint.parent.parent
        label = evaluation_name or (
            f"{checkpoint.stem}_n{num_conditions}_s{sample_steps}_seed{seed}"
        )
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "evaluation"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return run_dir / "evaluations" / f"{timestamp}_{label}"
    return checkpoint.parent / "evaluation"


def percentile_summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("mean", "std", "p05", "p50", "p95")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def action_statistics(
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    action_block_dim: int,
    max_actions: int,
) -> dict[str, object]:
    actions = edges[:, :, :action_block_dim].reshape(
        edges.shape[0], edges.shape[1], max_actions, ACTION_DIM
    )
    mask = np.arange(max_actions)[None, None, :] < counts[:, :, None]
    valid = actions[mask]
    differences = actions[:, :, 1:] - actions[:, :, :-1]
    difference_mask = (
        np.arange(max_actions - 1)[None, None, :] < (counts[:, :, None] - 1)
    )
    valid_differences = differences[difference_mask]
    return {
        "normalized_action_abs": percentile_summary(np.abs(valid).reshape(-1)),
        "normalized_adjacent_action_delta_abs": percentile_summary(
            np.abs(valid_differences).reshape(-1)
        ),
        "fraction_at_or_above_normalized_limit": float(np.mean(np.abs(valid) >= 1.0)),
    }


def descriptor_chamfer(
    true_edges: np.ndarray, generated_edges: np.ndarray, action_block_dim: int
) -> np.ndarray:
    true_descriptor = true_edges[:, :, action_block_dim:]
    generated_descriptor = generated_edges[:, :, action_block_dim:]
    distances = np.linalg.norm(
        true_descriptor[:, :, None, :] - generated_descriptor[:, None, :, :],
        axis=-1,
    )
    return 0.5 * (
        distances.min(axis=2).mean(axis=1) + distances.min(axis=1).mean(axis=1)
    )


def relative_outcome_consistency(
    generated: np.ndarray,
    counts: np.ndarray,
    actions: np.ndarray,
    query_states: np.ndarray,
    store: FrankaEdgeBundleStore,
) -> dict[str, object]:
    """Compare learned relative descriptors with the executed acceleration rollout."""
    actions64 = np.asarray(actions, dtype=np.float64)
    cumulative = np.cumsum(actions64, axis=2)
    velocity_delta_path = store.dt * cumulative
    position_increment = (
        query_states[:, None, None, 7:] * store.dt
        + store.dt**2 * (cumulative - 0.5 * actions64)
    )
    position_delta_path = np.cumsum(position_increment, axis=2)
    final_indices = np.clip(counts - 1, 0, store.max_actions - 1)
    batch_indices = np.arange(len(generated))[:, None]
    edge_indices = np.arange(store.set_size)[None, :]
    integrated_delta_q = position_delta_path[
        batch_indices, edge_indices, final_indices
    ]
    integrated_delta_dq = velocity_delta_path[
        batch_indices, edge_indices, final_indices
    ]

    q_range = np.asarray(store.q_range, dtype=np.float64)
    dq_scale = 2.0 * np.asarray(store.dq_max, dtype=np.float64)
    predicted_delta_q = generated[:, :, store.delta_q_slice] * q_range[None, None, :]
    predicted_delta_dq = (
        generated[:, :, store.delta_dq_slice] * dq_scale[None, None, :]
    )
    q_error = np.linalg.norm(predicted_delta_q - integrated_delta_q, axis=2)
    dq_error = np.linalg.norm(predicted_delta_dq - integrated_delta_dq, axis=2)
    return {
        "delta_q_l2_error_radians": percentile_summary(q_error.reshape(-1)),
        "delta_dq_l2_error_radians_per_second": percentile_summary(
            dq_error.reshape(-1)
        ),
        "note": (
            "The planner executes the float64-integrated acceleration endpoint; "
            "learned relative outcomes are auxiliary descriptors."
        ),
    }


def rollout_validity(
    conditions: np.ndarray,
    generated: np.ndarray,
    counts: np.ndarray,
    store: FrankaEdgeBundleStore,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized_actions = generated[:, :, : store.action_block_dim].reshape(
        len(conditions), store.set_size, store.max_actions, ACTION_DIM
    )
    actions = np.clip(normalized_actions, -1.0, 1.0) * store.ddq_max[
        None, None, None, :
    ]
    q_lower = np.asarray(store.metadata["normalization"]["q_lower"], dtype=np.float32)
    q_upper = np.asarray(store.metadata["normalization"]["q_upper"], dtype=np.float32)
    q = q_lower + 0.5 * (conditions[:, :7] + 1.0) * (q_upper - q_lower)
    dq = conditions[:, 7:] * store.dq_max[None, :]
    query_states = np.concatenate((q, dq), axis=1).astype(np.float32, copy=False)
    edge_finite = np.ones((len(conditions), store.set_size), dtype=bool)
    edge_limit_valid = np.ones_like(edge_finite)
    limit_valid_prefix_steps = np.zeros_like(counts, dtype=np.int64)
    finite_waypoints = 0
    limit_valid_waypoints = 0
    total_waypoints = 0
    steps = np.arange(store.max_actions)[None, :]
    for condition_index, state in enumerate(query_states):
        acceleration = actions[condition_index].astype(np.float32, copy=False)
        cumulative = np.cumsum(acceleration, axis=1)
        velocity = state[None, None, 7:] + store.dt * cumulative
        position_increment = (
            state[None, None, 7:] * store.dt
            + store.dt**2 * (cumulative - 0.5 * acceleration)
        )
        position = state[None, None, :7] + np.cumsum(position_increment, axis=1)
        finite = np.all(np.isfinite(position), axis=2) & np.all(
            np.isfinite(velocity), axis=2
        )
        within_limits = (
            finite
            & np.all(position >= q_lower[None, None, :], axis=2)
            & np.all(position <= q_upper[None, None, :], axis=2)
            & np.all(np.abs(velocity) <= store.dq_max[None, None, :], axis=2)
        )
        mask = steps < counts[condition_index, :, None]
        edge_finite[condition_index] = np.all(~mask | finite, axis=1)
        edge_limit_valid[condition_index] = np.all(~mask | within_limits, axis=1)
        invalid = mask & ~within_limits
        has_invalid = np.any(invalid, axis=1)
        first_invalid = np.argmax(invalid, axis=1)
        limit_valid_prefix_steps[condition_index] = np.where(
            has_invalid, first_invalid, counts[condition_index]
        )
        total_waypoints += int(mask.sum())
        finite_waypoints += int(np.sum(mask & finite))
        limit_valid_waypoints += int(np.sum(mask & within_limits))
    total_edges = int(edge_limit_valid.size)
    executed_action_mask = (
        np.arange(store.max_actions)[None, None, :] < counts[:, :, None]
    )
    unclipped_executed = np.abs(normalized_actions)[executed_action_mask]
    return (
        {
            "total_generated_edges": total_edges,
            "finite_edge_fraction": float(edge_finite.mean()),
            "joint_velocity_limit_valid_edge_fraction": float(
                edge_limit_valid.mean()
            ),
            "total_executed_waypoints": total_waypoints,
            "finite_waypoint_fraction": float(finite_waypoints / total_waypoints),
            "joint_velocity_limit_valid_waypoint_fraction": float(
                limit_valid_waypoints / total_waypoints
            ),
            "limit_valid_prefix_steps": percentile_summary(
                limit_valid_prefix_steps.reshape(-1)
            ),
            "fraction_with_at_least_5_limit_valid_steps": float(
                np.mean(limit_valid_prefix_steps >= 5)
            ),
            "acceleration_values_clipped_fraction": float(
                np.mean(unclipped_executed > 1.0)
            ),
        },
        edge_limit_valid,
        limit_valid_prefix_steps,
        actions,
        query_states,
    )


def sampled_self_collision_validity(
    *,
    limit_valid_prefix_steps: np.ndarray,
    actions: np.ndarray,
    query_states: np.ndarray,
    dt: float,
    sample_count: int,
    rng: np.random.Generator,
    urdf: Path,
    minimum_prefix_steps: int = 5,
) -> dict[str, object]:
    candidates = np.argwhere(limit_valid_prefix_steps >= minimum_prefix_steps)
    sample_count = min(max(0, sample_count), len(candidates))
    if sample_count == 0:
        return {
            "minimum_limit_valid_prefix_steps": minimum_prefix_steps,
            "sampled_limit_valid_prefix_edges": 0,
            "sampled_edges_with_self_collision": 0,
            "sampled_limit_valid_prefix_self_collision_free_fraction": None,
            "sampled_initial_states_checked": 0,
            "sampled_waypoints_checked": 0,
        }
    chosen = candidates[
        rng.choice(len(candidates), size=sample_count, replace=False)
    ]
    checker = FrankaSelfCollisionChecker(urdf)
    collisions = 0
    waypoints_checked = 0
    initial_states_checked = 0
    try:
        for condition_index, edge_index in chosen:
            state = query_states[condition_index].astype(np.float64, copy=True)
            initial_states_checked += 1
            collided = checker.in_collision(state[:7])
            if not collided:
                for acceleration in actions[
                    condition_index,
                    edge_index,
                    : limit_valid_prefix_steps[condition_index, edge_index],
                ]:
                    state[:7] = (
                        state[:7]
                        + state[7:] * dt
                        + 0.5 * acceleration * dt * dt
                    )
                    state[7:] = state[7:] + acceleration * dt
                    waypoints_checked += 1
                    if checker.in_collision(state[:7]):
                        collided = True
                        break
            collisions += int(collided)
    finally:
        checker.close()
    return {
        "minimum_limit_valid_prefix_steps": minimum_prefix_steps,
        "sampled_limit_valid_prefix_edges": sample_count,
        "sampled_edges_with_self_collision": collisions,
        "sampled_limit_valid_prefix_self_collision_free_fraction": float(
            (sample_count - collisions) / sample_count
        ),
        "sampled_initial_states_checked": initial_states_checked,
        "sampled_waypoints_checked": waypoints_checked,
    }


def main() -> None:
    args = parse_args()
    if args.num_conditions <= 0:
        raise ValueError("num-conditions must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.sample_steps <= 0:
        raise ValueError("sample-steps must be positive")
    if args.self_collision_edges < 0:
        raise ValueError("self-collision-edges cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    store = FrankaEdgeBundleStore(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset_sha256 = file_sha256(args.dataset.resolve())
    expected_dataset_sha256 = checkpoint.get("dataset_metadata", {}).get(
        "dataset_sha256"
    )
    if expected_dataset_sha256 is None:
        raise ValueError("Checkpoint does not contain a dataset SHA-256 fingerprint")
    if expected_dataset_sha256 != dataset_sha256:
        raise ValueError(
            "Checkpoint/dataset mismatch: "
            f"expected {expected_dataset_sha256}, found {dataset_sha256}"
        )
    config = checkpoint["config"]
    model = EdgeSetFlowModel(
        edge_dim=store.edge_dim,
        cond_dim=14,
        set_size=store.set_size,
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

    rng = np.random.default_rng(args.seed)
    count = min(args.num_conditions, len(store.conds["val"]))
    indices = rng.choice(len(store.conds["val"]), size=count, replace=False)
    cond, true_edges, _, true_counts = store.encode_batch("val", indices)
    generated_parts = []
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            condition = torch.as_tensor(
                cond[offset : offset + args.batch_size], device=device
            )
            generated_parts.append(
                sample_edge_sets(
                    model,
                    condition,
                    steps=args.sample_steps,
                    set_size=store.set_size,
                    edge_dim=store.edge_dim,
                ).cpu().numpy()
            )
    generated = np.concatenate(generated_parts, axis=0)
    if not np.isfinite(generated).all():
        raise ValueError("Flow sampler produced non-finite values")
    generated_step_fraction = np.clip(
        generated[:, :, store.step_count_index],
        1.0 / store.max_actions,
        1.0,
    )
    generated_counts = np.clip(
        np.rint(generated_step_fraction * store.max_actions).astype(np.int64),
        1,
        store.max_actions,
    )
    true_step_fraction = true_edges[:, :, store.step_count_index]
    chamfer = descriptor_chamfer(true_edges, generated, store.action_block_dim)
    (
        rollout_stats,
        edge_limit_valid,
        limit_valid_prefix_steps,
        decoded_actions,
        query_states,
    ) = rollout_validity(cond, generated, generated_counts, store)
    collision_stats = sampled_self_collision_validity(
        limit_valid_prefix_steps=limit_valid_prefix_steps,
        actions=decoded_actions,
        query_states=query_states,
        dt=store.dt,
        sample_count=args.self_collision_edges,
        rng=rng,
        urdf=args.urdf,
    )
    outcome_consistency = relative_outcome_consistency(
        generated,
        generated_counts,
        decoded_actions,
        query_states,
        store,
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics", {}),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "requested_self_collision_edges": args.self_collision_edges,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "urdf": str(args.urdf.resolve()),
        "collision_asset_sha256": collision_asset_hashes(args.urdf),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "evaluation_runtime_source_file_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT_DIR)): file_sha256(
                Path(__file__).resolve()
            ),
            "scripts/train_franka_edge_flow_matching.py": file_sha256(
                ROOT_DIR / "scripts" / "train_franka_edge_flow_matching.py"
            ),
            "scripts/FrankaPanda.py": file_sha256(
                TRAINING_SCRIPTS_DIR / "FrankaPanda.py"
            ),
        },
        "num_conditions": count,
        "sample_steps": args.sample_steps,
        "evaluation_scope_note": (
            f"Held-out means the {len(store.conds['val']):,}-bundle validation "
            "split. Training and "
            "validation share the same copied raw-edge library and source "
            "trajectory corpus; these metrics do not measure unseen-trajectory "
            "generalization."
        ),
        "duration_seconds": {
            "held_out": percentile_summary(true_counts * store.dt),
            "generated": percentile_summary(generated_counts * store.dt),
        },
        "normalized_step_count": {
            "held_out": percentile_summary(true_step_fraction),
            "generated": percentile_summary(generated_step_fraction),
        },
        "action_count": {
            "held_out": percentile_summary(true_counts),
            "generated": percentile_summary(generated_counts),
        },
        "actions": {
            "held_out": action_statistics(
                true_edges,
                true_counts,
                action_block_dim=store.action_block_dim,
                max_actions=store.max_actions,
            ),
            "generated_unclipped": action_statistics(
                generated,
                generated_counts,
                action_block_dim=store.action_block_dim,
                max_actions=store.max_actions,
            ),
        },
        "outcome_descriptor_symmetric_chamfer": percentile_summary(chamfer),
        "relative_outcome_vs_integrated_rollout": outcome_consistency,
        "decoded_rollout_validity": {**rollout_stats, **collision_stats},
    }

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_evaluation_dir(
            args.checkpoint,
            args.evaluation_name,
            num_conditions=args.num_conditions,
            sample_steps=args.sample_steps,
            seed=args.seed,
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        output_dir / "samples.npz",
        condition=cond,
        true_edges=true_edges,
        generated_edges=generated,
        true_action_counts=true_counts,
        generated_action_counts=generated_counts,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(
        true_counts.reshape(-1) * store.dt,
        bins=30,
        alpha=0.6,
        label="held out",
    )
    axes[0].hist(
        generated_counts.reshape(-1) * store.dt,
        bins=30,
        alpha=0.6,
        label="generated",
    )
    axes[0].set(title="Edge duration", xlabel="Seconds")
    true_actions = true_edges[:, :, : store.action_block_dim].reshape(
        count, store.set_size, store.max_actions, ACTION_DIM
    )
    generated_actions = generated[:, :, : store.action_block_dim].reshape(
        count, store.set_size, store.max_actions, ACTION_DIM
    )
    axes[1].hist(true_actions.reshape(-1), bins=50, alpha=0.6, label="held out")
    axes[1].hist(generated_actions.reshape(-1), bins=50, alpha=0.6, label="generated")
    axes[1].set(title="Normalized acceleration including padding", xlabel="ddq / limit")
    axes[2].hist(chamfer, bins=30, color="#4c78a8")
    axes[2].set(title="Outcome-descriptor set distance", xlabel="Symmetric Chamfer")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis is not axes[2]:
            axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "distribution_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
