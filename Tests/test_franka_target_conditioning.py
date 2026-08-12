"""Tests for target-conditioned Franka edge data and training losses."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from create_franka_target_conditioned_edge_dataset import (  # noqa: E402
    target_dependent_order,
)
from train_franka_edge_flow_matching import (  # noqa: E402
    EdgeSetFlowModel,
    flow_matching_loss,
    target_condition_metrics,
)


class FrankaTargetConditioningTests(unittest.TestCase):
    """Protect the two new condition fields and target-aware training loss."""

    def test_goal_mode_order_ignores_velocity(self):
        """Position-only goals must rank edges without terminal velocity."""
        edges = np.zeros((1, 4, 14), dtype=np.float32)
        target = np.zeros((1, 14), dtype=np.float32)
        target[0, 0] = 1.0
        edges[0, :, 0] = [1.0, 0.9, 0.0, 0.0]
        edges[0, :, 7] = [10.0, 0.0, 0.0, 0.0]
        order = target_dependent_order(
            edges,
            target,
            np.array([True]),
            directed_count=2,
        )
        np.testing.assert_array_equal(order[0, :2], np.array([0, 1]))

    def test_target_metrics_use_position_only_for_goal_rows(self):
        """Goal-mode recall ignores velocity while full-state recall does not."""
        action_block_dim = 14
        estimated = torch.zeros((2, 4, action_block_dim + 15))
        cond = torch.zeros((2, 29))
        cond[:, 14] = 1.0
        cond[:, 21] = 1.0
        cond[1, 28] = 1.0
        # Edge zero exactly matches target q but not target dq.
        estimated[:, 0, action_block_dim + 1] = 0.5
        distance, recall = target_condition_metrics(
            estimated,
            cond,
            action_block_dim=action_block_dim,
            recall_radius=0.4,
        )
        self.assertGreater(float(distance[0]), 0.9)
        self.assertAlmostEqual(float(distance[1]), 0.0)
        self.assertEqual(float(recall[0]), 0.0)
        self.assertEqual(float(recall[1]), 1.0)

    def test_target_aware_flow_loss_backpropagates(self):
        """The combined flow and target-progress objective is differentiable."""
        max_actions = 2
        action_block_dim = max_actions * 7
        edge_dim = action_block_dim + 15
        model = EdgeSetFlowModel(
            edge_dim=edge_dim,
            cond_dim=29,
            set_size=4,
            hidden_dim=16,
            depth=1,
            num_heads=4,
            time_embed_dim=8,
            cond_embed_dim=8,
        )
        edges = torch.zeros((3, 4, edge_dim))
        cond = torch.zeros((3, 29))
        cond[:, 14] = 0.5
        cond[1, 28] = 1.0
        mask = torch.ones((3, 4, max_actions), dtype=torch.bool)
        loss, metrics = flow_matching_loss(
            model,
            edges,
            cond,
            mask,
            action_block_dim=action_block_dim,
            max_actions=max_actions,
            padded_action_weight=0.01,
            target_progress_weight=0.25,
            target_recall_radius=0.4,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("target_progress", metrics)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
