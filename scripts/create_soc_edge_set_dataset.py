#!/usr/bin/env python3
"""Create set-valued SOC edge data from an existing kinodynamic edge bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT_DIR
    / "mrmp_with_kite_extend"
    / "edge_bundles_unclamped"
    / "eb_second_order_car_kinodynamic_TI_edges_100000.npz"
)
DEFAULT_OUTPUT = ROOT_DIR / "data" / "soc_edge_set_flow_matching_k32_n100000.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--set-size", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=256)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--max-radius", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--max-phi", type=float, default=float(np.pi / 3.0))
    parser.add_argument("--max-acceleration", type=float, default=2.0)
    parser.add_argument("--max-steering-rate", type=float, default=0.5)
    parser.add_argument("--max-timestep", type=float, default=2.0)
    parser.add_argument("--dx-scale", type=float, default=2.0)
    parser.add_argument("--dy-scale", type=float, default=1.5)
    return parser.parse_args()


def edge_features(actions, timesteps, final_states, *, args):
    edges = np.empty((actions.shape[0], 8), dtype=np.float32)
    edges[:, 0] = actions[:, 0] / args.max_acceleration
    edges[:, 1] = actions[:, 1] / args.max_steering_rate
    edges[:, 2] = timesteps / args.max_timestep
    edges[:, 3] = final_states[:, 0] / args.dx_scale
    edges[:, 4] = final_states[:, 1] / args.dy_scale
    edges[:, 5] = final_states[:, 2] / np.pi
    edges[:, 6] = final_states[:, 3] / args.max_speed
    edges[:, 7] = final_states[:, 4] / args.max_phi
    return edges


def diversity_features(edges):
    scales = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.7, 0.5, 0.5], dtype=np.float32)
    return edges * scales


def farthest_point_sample(features, k, rng):
    n = features.shape[0]
    selected = np.empty(k, dtype=np.int64)
    selected[0] = rng.integers(n)
    min_dist2 = np.full(n, np.inf, dtype=np.float32)

    for i in range(1, k):
        diff = features - features[selected[i - 1]]
        dist2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def canonical_order(edge_set):
    angles = np.arctan2(edge_set[:, 4], edge_set[:, 3])
    return np.lexsort((edge_set[:, 2], angles))


def sample_query(rng, starts_norm, args):
    u = rng.random()
    if u < 0.6:
        return starts_norm[rng.integers(starts_norm.shape[0])]
    if u < 0.9:
        return rng.uniform(-1.0, 1.0, size=2)

    # Boundary-biased dynamic states.
    v = rng.choice([-1.0, 1.0]) if rng.random() < 0.5 else rng.uniform(-1.0, 1.0)
    phi = rng.choice([-1.0, 1.0]) if rng.random() < 0.5 else rng.uniform(-1.0, 1.0)
    v += rng.normal(0.0, 0.03)
    phi += rng.normal(0.0, 0.03)
    return np.clip(np.array([v, phi], dtype=np.float64), -1.0, 1.0)


def query_candidates(tree, query, args, rng):
    radius = args.radius
    candidates = np.empty(0, dtype=np.int64)
    while radius <= args.max_radius:
        hits = tree.query_ball_point(query, r=radius)
        if len(hits) >= args.set_size:
            candidates = np.asarray(hits, dtype=np.int64)
            break
        radius *= 1.5

    if candidates.size < args.set_size:
        _, hits = tree.query(query, k=max(args.set_size, args.candidate_pool))
        candidates = np.asarray(hits, dtype=np.int64).reshape(-1)

    if candidates.size > args.candidate_pool:
        candidates = rng.choice(candidates, size=args.candidate_pool, replace=False)

    return candidates


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    bundle = np.load(args.bundle, allow_pickle=True)
    starts = bundle["start_states"].astype(np.float32)
    actions = bundle["actions"].astype(np.float32)
    timesteps = bundle["timesteps"].astype(np.float32)
    lengths = bundle["trajectory_lengths"].astype(np.int64)
    trajectories = bundle["trajectories"].astype(np.float32)
    final_states = trajectories[np.arange(lengths.shape[0]), lengths - 1]

    starts_norm = np.column_stack(
        [starts[:, 3] / args.max_speed, starts[:, 4] / args.max_phi]
    ).astype(np.float32)
    tree = cKDTree(starts_norm)

    all_edges = edge_features(actions, timesteps, final_states, args=args)
    all_diversity = diversity_features(all_edges)

    num_total = args.num_samples
    conds = np.empty((num_total, 2), dtype=np.float32)
    edge_sets = np.empty((num_total, args.set_size, 8), dtype=np.float32)
    source_edge_ids = np.empty((num_total, args.set_size), dtype=np.int64)

    for sample_idx in tqdm(range(num_total), desc="Building SOC edge-set dataset"):
        query = sample_query(rng, starts_norm, args)
        candidate_ids = query_candidates(tree, query, args, rng)
        local_choice = farthest_point_sample(all_diversity[candidate_ids], args.set_size, rng)
        chosen_ids = candidate_ids[local_choice]
        chosen_edges = all_edges[chosen_ids]

        order = canonical_order(chosen_edges)
        conds[sample_idx] = query
        edge_sets[sample_idx] = chosen_edges[order]
        source_edge_ids[sample_idx] = chosen_ids[order]

    perm = rng.permutation(num_total)
    num_val = int(round(num_total * args.val_fraction))
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]

    metadata = {
        "bundle": str(Path(args.bundle).resolve()),
        "num_samples": int(args.num_samples),
        "num_train": int(train_idx.shape[0]),
        "num_val": int(val_idx.shape[0]),
        "set_size": int(args.set_size),
        "edge_dim": 8,
        "cond_dim": 2,
        "edge_fields": [
            "acceleration_norm",
            "steering_rate_norm",
            "timestep_norm",
            "dx_norm",
            "dy_norm",
            "dtheta_norm",
            "v_final_norm",
            "phi_final_norm",
        ],
        "cond_fields": ["v_norm", "phi_norm"],
        "normalization": {
            "max_speed": args.max_speed,
            "max_phi": args.max_phi,
            "max_acceleration": args.max_acceleration,
            "max_steering_rate": args.max_steering_rate,
            "max_timestep": args.max_timestep,
            "dx_scale": args.dx_scale,
            "dy_scale": args.dy_scale,
            "dtheta_scale": float(np.pi),
        },
        "query_mixture": {
            "bundle_start_state": 0.6,
            "uniform_v_phi": 0.3,
            "boundary_biased": 0.1,
        },
        "candidate_radius_initial": args.radius,
        "candidate_radius_max": args.max_radius,
        "candidate_pool": args.candidate_pool,
        "selection": "farthest_point_sample_on_normalized_edge_features_then_angle_time_sort",
        "seed": args.seed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        conds_train=conds[train_idx],
        edges_train=edge_sets[train_idx],
        source_edge_ids_train=source_edge_ids[train_idx],
        conds_val=conds[val_idx],
        edges_val=edge_sets[val_idx],
        source_edge_ids_val=source_edge_ids[val_idx],
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )
    print(f"Saved {output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
