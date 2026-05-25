#!/usr/bin/env python3
"""Train a FlowMRMP flow-matching policy from the local DiTree-style dataset."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DITREE_DIR = ROOT_DIR / "ditree"

# The imported DiTree files use local imports such as `from maze_datasets import ...`.
sys.path.insert(0, str(DITREE_DIR))
os.chdir(DITREE_DIR)

from train_diffusion_policy import train_by_steps  # noqa: E402


UNET_DIMS = {
    "small": [64, 128, 256],
    "medium": [256, 512, 1024],
    "large": [512, 1024, 2048],
    "xlarge": [1024, 2048, 4096],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DITREE_DIR / "cfgs" / "carmaze.yaml"))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--debug", action="store_true", help="Use a tiny dataset and short rollout for smoke testing.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--rollout-every", type=int, default=0, help="0 disables rollouts during training.")
    parser.add_argument("--initial-rollout", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--unet-size", choices=sorted(UNET_DIMS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    unet_size = args.unet_size or cfg.get("unet_size", cfg.get("unet_down_dims", "large"))
    experiment_name = args.experiment_name or f"flowmrmp_{cfg.get('env_id', 'carmaze')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rollout_every = args.rollout_every if args.rollout_every > 0 else 10**12

    train_by_steps(
        experiment_name=experiment_name,
        debug=args.debug or cfg.get("debug", False),
        device=args.device or cfg.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=str(ROOT_DIR / "checkpoints"),
        prediction_type=cfg.get("prediction_type", "actions"),
        obs_history=cfg.get("obs_history", 1),
        action_history=cfg.get("action_history", 1),
        goal_conditioned=cfg.get("goal_conditioned", True),
        local_map_conditioned=cfg.get("local_map_conditioned", True),
        local_map_size=cfg.get("local_map_size", 20),
        local_map_scale=cfg.get("local_map_scale", 0.2),
        local_map_embedding_dim=cfg.get("local_map_embedding_dim", 400),
        local_map_encoder=cfg.get("local_map_encoder", "resnet"),
        num_epochs=args.epochs or cfg.get("num_epochs", 15),
        batch_size=args.batch_size or cfg.get("batch_size", 128),
        num_workers=args.num_workers if args.num_workers is not None else cfg.get("num_workers", 3),
        checkpoint_every=args.checkpoint_every,
        rollout_every=rollout_every,
        rollouts=args.rollout_every > 0,
        initial_rollout=args.initial_rollout or cfg.get("initial_rollout", False),
        env_id=cfg.get("env_id", "carmaze"),
        augmentations=cfg.get("augmentations", ["mirror"]),
        policy="flow_matching",
        num_diffusion_iters=cfg.get("num_diffusion_iters", 1),
        unet_down_dims=UNET_DIMS[unet_size],
        pred_horizon=cfg.get("pred_horizon", 64),
        action_horizon=cfg.get("action_horizon", 8),
    )


if __name__ == "__main__":
    main()
