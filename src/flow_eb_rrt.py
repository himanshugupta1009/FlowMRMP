"""Flow-generated edge-bundle RRT integration for FlowMRMP.

The implementation reuses the MRMP KiTE-RRT planner machinery as a dependency,
but keeps all FlowMRMP-specific code in this repository's top-level ``src``.

For Franka, each tree node is a 14D state ``[q(7), dq(7)]``. The flow model
generates 32 candidate edges, where each edge contains up to 50 joint-
acceleration commands, a predicted step count, and a predicted relative 14D
outcome ``[delta_q, delta_dq]``. Planning uses the learned outcome only to rank
candidates against the sampled RRT target. Before an edge can enter the tree,
its controls are re-propagated through the Franka dynamics and the resulting
waypoints are checked for acceleration, jerk, position, velocity, and
self-collision validity.

High-level extension sequence:
1. Sample a 14D RRT target and choose its nearest existing tree node.
2. Generate or retrieve that node's flow edge bundle.
3. Rank unused edges by predicted terminal distance to the sampled target.
4. Traverse the ranked list, propagating and validating one edge at a time.
5. Add the first valid result; use a random-control fallback if none succeeds.
"""

from __future__ import annotations

from collections import deque
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MRMP_SRC) not in sys.path:
    sys.path.insert(0, str(MRMP_SRC))

from kinodynamic_TI_eb_rrt import KinoTIEBRRT, KinoTIEBTreeNode  # noqa: E402
from scripts.train_soc_edge_flow_matching import EdgeSetFlowModel  # noqa: E402
from scripts.train_franka_edge_flow_matching import (  # noqa: E402
    EdgeSetFlowModel as FrankaEdgeSetFlowModel,
)


# ---------------------------------------------------------------------------
# Tree-node and generated-bundle containers
# ---------------------------------------------------------------------------

class FlowEBTreeNode(KinoTIEBTreeNode):
    """RRT node extended with a lazily generated, node-local flow bundle."""

    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost):
        """Initialize inherited RRT fields and an empty flow-bundle cache."""
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,
                         path_from_parent, time_so_far, cost)
        # None means the model has not yet been queried for this tree node.
        self.flow_edge_bundle = None


class GeneratedEdgeBundle:
    """Fixed-control EdgeBundle compatibility container used by the SOC path."""

    def __init__(self, actions, timesteps, start_states, final_states):
        """Store one control, duration, start, and predicted end per edge."""
        self.actions = np.asarray(actions, dtype=np.float64)
        self.timesteps = np.asarray(timesteps, dtype=np.float64)
        self.start_states = np.asarray(start_states, dtype=np.float64)
        self.final_states = np.asarray(final_states, dtype=np.float64)
        self.num_edges = int(self.timesteps.shape[0])


class GeneratedSequenceEdgeBundle:
    """One Franka bundle containing variable-length acceleration sequences.

    ``final_states`` and ``relative_changes`` are learned estimates used only
    for inexpensive ranking. They never bypass physical propagation.
    """

    def __init__(
        self,
        action_sequences,
        action_dt,
        start_states,
        final_states,
        relative_changes,
    ):
        """Build a bundle and verify that every edge has aligned state metadata."""
        self.action_sequences = [
            np.asarray(actions, dtype=np.float32) for actions in action_sequences
        ]
        self.actions = self.action_sequences
        self.action_dt = float(action_dt)
        self.timesteps = np.asarray(
            [len(actions) * self.action_dt for actions in self.action_sequences],
            dtype=np.float64,
        )
        self.start_states = np.asarray(start_states, dtype=np.float64)
        # These are the fast, learned endpoint estimates used only for ranking.
        # The state added to the tree always comes from validated propagation.
        self.final_states = np.asarray(final_states, dtype=np.float64)
        self.relative_changes = np.asarray(relative_changes, dtype=np.float64)
        self.num_edges = len(self.action_sequences)
        if not (
            self.start_states.shape[0]
            == self.final_states.shape[0]
            == self.relative_changes.shape[0]
            == self.num_edges
        ):
            raise ValueError("inconsistent generated sequence-bundle dimensions")
        expected_state_shape = (self.num_edges, self.start_states.shape[1])
        if self.relative_changes.shape != expected_state_shape:
            raise ValueError(
                "relative_changes must contain one full-state delta per edge"
            )

    def release_edge(self, edge_index):
        """Release a sequence after its one permitted RRT trial to save memory."""
        self.action_sequences[int(edge_index)] = None


# ---------------------------------------------------------------------------
# Original Second Order Car generator retained for shared FlowEBRRT support
# ---------------------------------------------------------------------------


class SOCFlowEdgeGenerator:
    """Load a trained SOC edge-set flow model and sample denormalized edges."""

    def __init__(self, *,
                 checkpoint_path,
                 device="cuda:1",
                 sample_steps=16,
                 clamp_outputs=True,
                 seed=123):
        """Load the SOC checkpoint and reconstruct its model/normalization."""
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            root_relative = ROOT_DIR / self.checkpoint_path
            if root_relative.exists():
                self.checkpoint_path = root_relative
        self.device = self._resolve_device(device)
        self.sample_steps = int(sample_steps)
        self.clamp_outputs = bool(clamp_outputs)
        self.seed = int(seed)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,
                                weights_only=False)
        self.config = checkpoint["config"]
        self.metadata = checkpoint["dataset_metadata"]
        self.normalization = self.metadata["normalization"]
        self.edge_dim = int(self.metadata["edge_dim"])
        self.cond_dim = int(self.metadata["cond_dim"])
        self.set_size = int(self.metadata["set_size"])

        self.model = EdgeSetFlowModel(
            edge_dim=self.edge_dim,
            cond_dim=self.cond_dim,
            set_size=self.set_size,
            hidden_dim=int(self.config["hidden_dim"]),
            depth=int(self.config["depth"]),
            num_heads=int(self.config["num_heads"]),
            mlp_ratio=float(self.config["mlp_ratio"]),
            dropout=float(self.config["dropout"]),
            time_embed_dim=int(self.config["time_embed_dim"]),
            cond_embed_dim=int(self.config["cond_embed_dim"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(self.seed)
        self.enable_profile = False
        self.profile = {
            "samples": 0,
            "sample_total_s": 0.0,
            "model_s": 0.0,
            "postprocess_s": 0.0,
        }

    @staticmethod
    def _resolve_device(device):
        """Use the requested CUDA device when present, otherwise fall back to CPU."""
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return torch.device("cpu")
            if ":" in device:
                index = int(device.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    return torch.device("cpu")
        return torch.device(device)

    def condition_from_state(self, state):
        """Extract the SOC condition ``[speed, steering]`` from one state."""
        norm = self.normalization
        return np.array([
            state[3] / float(norm["max_speed"]),
            state[4] / float(norm["max_phi"]),
        ], dtype=np.float32)

    def conditions_from_states(self, states):
        """Vectorized SOC condition extraction for batched inference."""
        norm = self.normalization
        states = np.asarray(states, dtype=np.float32)
        cond = np.empty((states.shape[0], 2), dtype=np.float32)
        cond[:, 0] = states[:, 3] / float(norm["max_speed"])
        cond[:, 1] = states[:, 4] / float(norm["max_phi"])
        return np.clip(cond, -1.0, 1.0)

    def denormalize_edges(self, edges):
        """Convert eight normalized SOC output features back to physical units."""
        norm = self.normalization
        out = np.asarray(edges, dtype=np.float32).copy()
        out[..., 0] *= float(norm["max_acceleration"])
        out[..., 1] *= float(norm["max_steering_rate"])
        out[..., 2] *= float(norm["max_timestep"])
        out[..., 3] *= float(norm["dx_scale"])
        out[..., 4] *= float(norm["dy_scale"])
        out[..., 5] *= float(norm["dtheta_scale"])
        out[..., 6] *= float(norm["max_speed"])
        out[..., 7] *= float(norm["max_phi"])
        return out

    def canonical_order(self, edges):
        """Order SOC edges by endpoint direction and then duration."""
        angles = np.arctan2(edges[:, 4], edges[:, 3])
        return np.lexsort((edges[:, 2], angles))

    def set_profile_enabled(self, enabled):
        """Enable or disable synchronized SOC inference timings."""
        self.enable_profile = bool(enabled)

    def _sync_if_cuda(self):
        """Synchronize CUDA so optional wall-clock profiling is accurate."""
        if self.enable_profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _sample_edges_tensor(self, cond):
        """Integrate the SOC learned flow ODE from Gaussian noise to edge sets."""
        batch_size = cond.shape[0]
        edges = torch.randn(
            batch_size, self.set_size, self.edge_dim,
            device=self.device,
            generator=self.torch_generator,
        )
        dt = 1.0 / self.sample_steps
        self._sync_if_cuda()
        model_t0 = time.perf_counter() if self.enable_profile else 0.0
        for step in range(self.sample_steps):
            t = torch.full((batch_size,), step / self.sample_steps,
                           device=self.device, dtype=edges.dtype)
            edges = edges + dt * self.model(edges, t, cond)
        self._sync_if_cuda()
        model_s = time.perf_counter() - model_t0 if self.enable_profile else 0.0
        return edges, model_s

    def _postprocess_edges(self, edges, states, num_edges=None):
        """Clamp, sort, and wrap decoded SOC edge sets for the planner."""
        post_t0 = time.perf_counter() if self.enable_profile else 0.0
        edges = self.denormalize_edges(edges.detach().cpu().numpy())
        if self.clamp_outputs:
            norm = self.normalization
            edges[..., 0] = np.clip(edges[..., 0],
                                    -float(norm["max_acceleration"]),
                                    float(norm["max_acceleration"]))
            edges[..., 1] = np.clip(edges[..., 1],
                                    -float(norm["max_steering_rate"]),
                                    float(norm["max_steering_rate"]))
            edges[..., 2] = np.clip(edges[..., 2], 0.05, float(norm["max_timestep"]))
            edges[..., 6] = np.clip(edges[..., 6],
                                    -float(norm["max_speed"]),
                                    float(norm["max_speed"]))
            edges[..., 7] = np.clip(edges[..., 7],
                                    -float(norm["max_phi"]),
                                    float(norm["max_phi"]))

        bundles = []
        limit = None if num_edges is None else int(num_edges)
        for state, edge_set in zip(states, edges):
            edge_set = edge_set[self.canonical_order(edge_set)]
            if limit is not None:
                edge_set = edge_set[:limit]

            start_states = np.zeros((edge_set.shape[0], 5), dtype=np.float64)
            start_states[:, 3] = state[3]
            start_states[:, 4] = state[4]
            final_states = np.zeros((edge_set.shape[0], 5), dtype=np.float64)
            final_states[:, 0] = edge_set[:, 3]
            final_states[:, 1] = edge_set[:, 4]
            final_states[:, 2] = edge_set[:, 5]
            final_states[:, 3] = edge_set[:, 6]
            final_states[:, 4] = edge_set[:, 7]
            bundles.append(
                GeneratedEdgeBundle(edge_set[:, :2], edge_set[:, 2],
                                    start_states, final_states)
            )

        post_s = time.perf_counter() - post_t0 if self.enable_profile else 0.0
        return bundles, post_s

    @torch.no_grad()
    def sample_batch(self, states, num_edges=None):
        """Generate one SOC edge bundle for every supplied state."""
        sample_t0 = time.perf_counter() if self.enable_profile else 0.0
        states = np.asarray(states, dtype=np.float64)
        cond_np = self.conditions_from_states(states)
        cond = torch.as_tensor(cond_np, device=self.device)

        edge_tensor, model_s = self._sample_edges_tensor(cond)
        bundles, post_s = self._postprocess_edges(edge_tensor, states, num_edges)

        self.profile["samples"] += states.shape[0]
        if self.enable_profile:
            self.profile["model_s"] += model_s
            self.profile["postprocess_s"] += post_s
            self.profile["sample_total_s"] += time.perf_counter() - sample_t0
        return bundles

    @torch.no_grad()
    def sample(self, state, num_edges=None):
        """Single-state convenience wrapper around :meth:`sample_batch`."""
        return self.sample_batch(np.asarray(state, dtype=np.float64)[None, :],
                                 num_edges=num_edges)[0]


# ---------------------------------------------------------------------------
# Franka flow-model inference and edge decoding
# ---------------------------------------------------------------------------


class FrankaFlowEdgeGenerator:
    """Sample and decode variable-length Franka acceleration-sequence bundles.

    The checkpoint is self-describing: its saved HDF5 metadata specifies all
    normalization scales and encoded-vector slices. That prevents planner-time
    decoding from silently drifting away from the training representation.
    """

    def __init__(
        self,
        *,
        checkpoint_path,
        device="auto",
        sample_steps=16,
        clamp_outputs=False,
        seed=123,
    ):
        """Load an inference checkpoint and reconstruct its exact architecture."""
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            root_relative = ROOT_DIR / self.checkpoint_path
            if root_relative.exists():
                self.checkpoint_path = root_relative
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = self._resolve_device(device)
        self.sample_steps = int(sample_steps)
        if self.sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        self.clamp_outputs = bool(clamp_outputs)
        self.seed = int(seed)

        # The checkpoint contains both network weights and the dataset encoding
        # contract needed to turn D-dimensional predictions into real controls.
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.config = checkpoint["config"]
        self.metadata = checkpoint["dataset_metadata"]
        self.normalization = self.metadata["normalization"]
        self.encoding = self.metadata["flow_encoding"]
        self.edge_dim = int(self.metadata["edge_dim"])
        self.cond_dim = int(self.metadata["cond_dim"])
        self.set_size = int(self.metadata["set_size"])
        self.max_actions = int(self.encoding["max_actions"])
        self.action_dim = int(self.encoding["action_dim"])
        self.action_block_dim = int(self.encoding["action_block_dim"])
        # New checkpoints predict a discrete normalized count. The older
        # duration branch remains readable for backward compatibility.
        if "step_count_index" in self.encoding:
            self.step_count_index = int(self.encoding["step_count_index"])
            self.duration_index = self.step_count_index
            self.length_encoding = "normalized_step_count"
        else:
            self.duration_index = int(self.encoding["duration_index"])
            self.step_count_index = self.duration_index
            self.length_encoding = "normalized_duration"
        self.dt = float(self.metadata["dt"])
        self.max_duration = float(self.metadata["max_suffix_duration"])
        self.q_lower = np.asarray(self.normalization["q_lower"], dtype=np.float32)
        self.q_upper = np.asarray(self.normalization["q_upper"], dtype=np.float32)
        self.dq_max = np.asarray(self.normalization["dq_max_abs"], dtype=np.float32)
        self.ddq_max = np.asarray(self.normalization["ddq_max_abs"], dtype=np.float32)
        # Relative outcomes are mandatory because FlowEBRRT ranks endpoints by
        # adding these deltas to the exact node from which it is extending.
        if "delta_q_slice" not in self.encoding or "delta_dq_slice" not in self.encoding:
            raise ValueError(
                "Franka FlowEBRRT requires a checkpoint trained with relative "
                "delta_q and delta_dq outcomes"
            )
        self.delta_q_slice = slice(*self.encoding["delta_q_slice"])
        self.delta_dq_slice = slice(*self.encoding["delta_dq_slice"])
        self.q_range = self.q_upper - self.q_lower

        self.model = FrankaEdgeSetFlowModel(
            edge_dim=self.edge_dim,
            cond_dim=self.cond_dim,
            set_size=self.set_size,
            hidden_dim=int(self.config["hidden_dim"]),
            depth=int(self.config["depth"]),
            num_heads=int(self.config["num_heads"]),
            mlp_ratio=float(self.config["mlp_ratio"]),
            dropout=float(self.config["dropout"]),
            time_embed_dim=int(self.config["time_embed_dim"]),
            cond_embed_dim=int(self.config["cond_embed_dim"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(self.seed)
        self.enable_profile = False
        self.profile = {
            "samples": 0,
            "sample_total_s": 0.0,
            "model_s": 0.0,
            "postprocess_s": 0.0,
        }

    @staticmethod
    def _resolve_device(device):
        """Resolve auto/CUDA/MPS/CPU and reject unavailable explicit devices."""
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        if str(device).startswith("mps") and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable")
        return torch.device(device)

    def set_profile_enabled(self, enabled):
        """Enable synchronization-based timing for model and postprocessing work."""
        self.enable_profile = bool(enabled)

    def _sync(self):
        """Synchronize asynchronous accelerators only when profiling is enabled."""
        if not self.enable_profile:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def conditions_from_states(self, states):
        """Normalize batched 14D states to the model condition range ``[-1,1]``.

        Joint positions use their asymmetric lower/upper ranges; velocities use
        symmetric per-joint maximum magnitudes.
        """
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 14:
            raise ValueError(f"Expected states shape (B,14), received {states.shape}")
        q_norm = 2.0 * (states[:, :7] - self.q_lower) / (
            self.q_upper - self.q_lower
        ) - 1.0
        dq_norm = states[:, 7:] / self.dq_max
        return np.clip(np.concatenate((q_norm, dq_norm), axis=1), -1.0, 1.0)

    def _sample_encoded(self, cond):
        """Euler-integrate the learned flow from noise to ``(B,K,D)`` predictions."""
        batch_size = cond.shape[0]
        # Every planner query starts with fresh Gaussian edge-set noise.
        edges = torch.randn(
            batch_size,
            self.set_size,
            self.edge_dim,
            device=self.device,
            generator=self.torch_generator,
        )
        step_size = 1.0 / self.sample_steps
        self._sync()
        started = time.perf_counter() if self.enable_profile else 0.0
        # Match training diagnostics: explicit Euler over flow time [0, 1].
        for step in range(self.sample_steps):
            t = torch.full(
                (batch_size,),
                step / self.sample_steps,
                device=self.device,
                dtype=edges.dtype,
            )
            edges = edges + step_size * self.model(edges, t, cond)
        self._sync()
        model_seconds = time.perf_counter() - started if self.enable_profile else 0.0
        return edges, model_seconds

    @staticmethod
    def _integrate(state, actions, dt):
        """Reference double-integrator rollout for one acceleration sequence."""
        current = np.asarray(state, dtype=np.float64).copy()
        path = np.empty((len(actions), 14), dtype=np.float64)
        for index, acceleration in enumerate(actions):
            next_state = np.empty(14, dtype=np.float64)
            next_state[:7] = (
                current[:7]
                + current[7:] * dt
                + 0.5 * acceleration * dt * dt
            )
            next_state[7:] = current[7:] + acceleration * dt
            path[index] = next_state
            current = next_state
        return path

    @staticmethod
    def _integrate_batch(state, actions, dt):
        """Vectorized exact rollout for diagnostics on rectangular sequences."""
        state = np.asarray(state, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        cumulative_acceleration = np.cumsum(actions, axis=1)
        velocity_path = state[None, None, 7:] + dt * cumulative_acceleration
        position_increment = (
            state[None, None, 7:] * dt
            + dt * dt * (cumulative_acceleration - 0.5 * actions)
        )
        position_path = state[None, None, :7] + np.cumsum(
            position_increment, axis=1
        )
        return np.concatenate((position_path, velocity_path), axis=2)

    def _decode(self, encoded, states, num_edges):
        """Decode model vectors into controls, counts, and predicted outcomes.

        Input ``encoded`` has shape ``(B,K,D)``. The returned bundle members
        contain only the first predicted N controls, so zero padding is never
        executed. Relative outcomes are denormalized and added to each exact
        start state for ranking; control propagation later determines truth.
        """
        started = time.perf_counter() if self.enable_profile else 0.0
        encoded = encoded.detach().cpu().numpy()
        # Recover the rectangular padded action block (B,K,L,7).
        normalized_actions = encoded[:, :, : self.action_block_dim].reshape(
            len(states), self.set_size, self.max_actions, self.action_dim
        )
        if self.clamp_outputs:
            normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        actions = normalized_actions * self.ddq_max[None, None, None, :]
        # Round the learned continuous output to an executable integer N in
        # [1, max_actions]. This N is the sole control-sequence cutoff.
        length_fraction = encoded[:, :, self.step_count_index]
        if self.length_encoding == "normalized_step_count":
            length_fraction = np.clip(
                length_fraction, 1.0 / self.max_actions, 1.0
            )
            action_counts = np.rint(
                length_fraction * self.max_actions
            ).astype(np.int64)
        else:
            length_fraction = np.clip(
                length_fraction, self.dt / self.max_duration, 1.0
            )
            action_counts = np.rint(
                length_fraction * self.max_duration / self.dt
            ).astype(np.int64)
        action_counts = np.clip(action_counts, 1, self.max_actions)
        # Decode the learned relative final state. No integration is required
        # for this ranking estimate, which keeps candidate sorting inexpensive.
        predicted_delta_q = (
            encoded[:, :, self.delta_q_slice]
            * self.q_range[None, None, :]
        )
        predicted_delta_dq = (
            encoded[:, :, self.delta_dq_slice]
            * (2.0 * self.dq_max[None, None, :])
        )
        predicted_relative_changes = np.concatenate(
            (predicted_delta_q, predicted_delta_dq), axis=2
        )

        bundles = []
        edge_limit = self.set_size if num_edges is None else min(int(num_edges), self.set_size)
        for batch_index, state in enumerate(states):
            sequences = []
            for edge_index in range(edge_limit):
                count = int(action_counts[batch_index, edge_index])
                # Discard padded controls immediately; downstream code sees a
                # genuinely ragged sequence of exactly N acceleration rows.
                sequence = np.array(
                    actions[batch_index, edge_index, :count],
                    dtype=np.float32,
                    copy=True,
                )
                sequences.append(sequence)
            start_states = np.repeat(
                np.asarray(state, dtype=np.float64)[None, :], edge_limit, axis=0
            )
            relative_changes = np.asarray(
                predicted_relative_changes[batch_index, :edge_limit],
                dtype=np.float64,
            )
            predicted_final_states = start_states + relative_changes
            bundles.append(
                GeneratedSequenceEdgeBundle(
                    sequences,
                    self.dt,
                    start_states,
                    predicted_final_states,
                    relative_changes,
                )
            )
        postprocess_seconds = (
            time.perf_counter() - started if self.enable_profile else 0.0
        )
        return bundles, postprocess_seconds

    @torch.no_grad()
    def sample_batch(self, states, num_edges=None):
        """Generate decoded bundles for a batch of exact 14D tree-node states."""
        started = time.perf_counter() if self.enable_profile else 0.0
        states = np.asarray(states, dtype=np.float64)
        cond = torch.as_tensor(
            self.conditions_from_states(states), device=self.device, dtype=torch.float32
        )
        encoded, model_seconds = self._sample_encoded(cond)
        bundles, postprocess_seconds = self._decode(encoded, states, num_edges)
        self.profile["samples"] += len(states)
        if self.enable_profile:
            self.profile["model_s"] += model_seconds
            self.profile["postprocess_s"] += postprocess_seconds
            self.profile["sample_total_s"] += time.perf_counter() - started
        return bundles

    @torch.no_grad()
    def sample(self, state, num_edges=None):
        """Generate one decoded bundle for one exact 14D tree-node state."""
        return self.sample_batch(
            np.asarray(state, dtype=np.float64)[None, :], num_edges=num_edges
        )[0]


# ---------------------------------------------------------------------------
# Flow-guided kinodynamic RRT
# ---------------------------------------------------------------------------


class FlowEBRRT(KinoTIEBRRT):
    """KinoTIEBRRT variant with lazy, node-local flow edge bundles.

    The inherited class supplies tree storage, nearest-neighbor search, random
    target sampling, collision interfaces, and random-control fallback. This
    subclass replaces static edge lookup with FM generation, predicted-outcome
    sorting, and sequence-aware validation.
    """

    def __init__(self, * ,
                 start, goal, goal_radius, env, agent,
                 flow_edge_generator,
                 use_fixed_sampling_time=True,
                 sampling_time_step=1.0,
                 minimum_time_step=0.1,
                 max_iter=1000,
                 planning_time=10.0,
                 isvalid_function,
                 cost_function,
                 reached_goal_function,
                 random_point_function,
                 translate_function,
                 sort_edges_function,
                 max_num_edges_per_node=32,
                 flow_prefetch_batch_size=1,
                 minimum_sequence_prefix_steps=5,
                 truncate_sequence_to_target=False,
                 num_skip_edges=10,
                 num_random_edges=1,
                 epsilon_random=0.01,
                 udf_seed=77,
                 goal_sampling_probability=0.1,
                 dynamic_agent_clearance=0.0,
                 debug_flag=False,
                 print_logs=False,
                 dynamic_obstacles=None):
        """Configure planning limits, model batching, and candidate traversal."""
        if dynamic_obstacles is None:
            from numba.typed import List
            from numba import types
            dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))

        # The inherited constructor requires an edge-bundle object even though
        # this subclass generates a different bundle for every node.
        dummy_edge_bundle = GeneratedEdgeBundle(
            actions=np.zeros((1, agent.action_length), dtype=np.float64),
            timesteps=np.ones(1, dtype=np.float64),
            start_states=np.zeros((1, agent.state_length), dtype=np.float64),
            final_states=np.zeros((1, agent.state_length), dtype=np.float64),
        )

        super().__init__(
            start=start, goal=goal, goal_radius=goal_radius, env=env, agent=agent,
            edge_bundle=dummy_edge_bundle,
            use_fixed_sampling_time=use_fixed_sampling_time,
            sampling_time_step=sampling_time_step,
            minimum_time_step=minimum_time_step,
            max_iter=max_iter,
            planning_time=planning_time,
            isvalid_function=isvalid_function,
            cost_function=cost_function,
            reached_goal_function=reached_goal_function,
            random_point_function=random_point_function,
            translate_function=translate_function,
            sort_edges_function=sort_edges_function,
            max_num_edges_per_node=max_num_edges_per_node,
            num_skip_edges=num_skip_edges,
            num_random_edges=num_random_edges,
            epsilon_random=epsilon_random,
            eb_kd_tree=None,
            get_eb_kd_tree_query=None,
            kd_tree_delta_radius=0.0,
            udf_seed=udf_seed,
            goal_sampling_probability=goal_sampling_probability,
            dynamic_agent_clearance=dynamic_agent_clearance,
            debug_flag=debug_flag,
            print_logs=print_logs,
            dynamic_obstacles=dynamic_obstacles,
        )
        self.flow_edge_generator = flow_edge_generator
        self.node_class = FlowEBTreeNode
        self.flow_prefetch_batch_size = max(1, int(flow_prefetch_batch_size))
        # Retained in the public configuration for checkpoint/benchmark
        # compatibility. Full-duration execution means target-based truncation
        # no longer consumes this value; only a verified goal-reaching prefix
        # may terminate an edge early.
        self.minimum_sequence_prefix_steps = max(
            1, int(minimum_sequence_prefix_steps)
        )
        if int(num_skip_edges) <= 0:
            raise ValueError("num_skip_edges must be positive")
        if truncate_sequence_to_target:
            raise ValueError(
                "Flow edges now execute their predicted full duration; "
                "target-based prefix truncation is incompatible with fast "
                "relative-outcome ranking"
            )
        self.truncate_sequence_to_target = False
        self._plan_deadline = None
        # Newly added nodes enter this queue so multiple node conditions can be
        # inferred together in the next model call.
        self.uncached_flow_node_ids = deque()
        self.profile = {
            "flow_generation_s": 0.0,
            "sort_edges_s": 0.0,
            "try_edge_s": 0.0,
            "random_control_s": 0.0,
            "extend_calls": 0,
            "flow_generation_calls": 0,
            "flow_generated_bundles": 0,
            "flow_cache_hits": 0,
            "flow_cache_misses": 0,
            "flow_prefetch_select_s": 0.0,
            "try_edge_calls": 0,
            "random_control_calls": 0,
            "sequence_edges_executed": 0,
            "sequence_executed_steps": 0,
            "sequence_available_steps": 0,
            "sequence_acceleration_rejections": 0,
            "sequence_jerk_rejections": 0,
        }

    def set_profile_enabled(self, enabled):
        """Enable generator timing; planner counters are always accumulated."""
        self.flow_edge_generator.set_profile_enabled(enabled)

    def get_random_time(self):
        """Sample a duration that is an integer multiple of integration dt."""
        max_steps = max(1, int(np.floor(self.max_sample_T / self.minimum_time_step)))
        steps = int(self.rng.integers(1, max_steps + 1))
        return float(steps * self.minimum_time_step)

    def _deadline_reached(self):
        """Return whether the current planning call exhausted wall-clock budget."""
        return self._plan_deadline is not None and time.time() >= self._plan_deadline

    def plan_path(self):
        """Run inherited RRT planning while exposing its deadline to inner loops."""
        # The inherited planner checks its budget between extensions.  Retain
        # that behavior and also expose the deadline inside expensive Flow
        # extension loops so one extension cannot substantially overrun it.
        self._plan_deadline = time.time() + self.planning_time
        try:
            return super().plan_path()
        finally:
            self._plan_deadline = None

    def get_path_to_node_id(self, goal_node_id):
        """Return a path while preserving ragged acceleration sequences."""
        reverse_nodes = []
        node_id = int(goal_node_id)
        while node_id != -1:
            node = self.tree.nodes[node_id]["value"]
            reverse_nodes.append(node)
            node_id = int(node.parent_id)
        nodes = reverse_nodes[::-1]
        ids = np.asarray([node.id for node in nodes], dtype=np.int32)
        states = np.stack([node.state for node in nodes]).astype(np.float64)
        control_list = [
            np.asarray(node.parent_action).copy() for node in nodes[1:]
        ]
        controls = (
            np.stack(control_list)
            if control_list
            and all(
                control.shape == (self.agent.action_length,)
                for control in control_list
            )
            else control_list
        )
        timesteps = np.asarray(
            [node.parent_action_duration for node in nodes[1:]],
            dtype=np.float64,
        )
        return ids, states, controls, timesteps

    def reset_tree(self, some_existing_tree=None):
        """Reset inherited tree state and discard the inference-prefetch queue."""
        super().reset_tree(some_existing_tree)
        self.uncached_flow_node_ids = deque()

    def add_rrt_node(self, *args, **kwargs):
        """Add a node normally, then mark it as eligible for batched FM prefetch."""
        node_id = super().add_rrt_node(*args, **kwargs)
        self.uncached_flow_node_ids.append(node_id)
        return node_id

    def _attach_flow_edge_bundle(self, node, edge_bundle):
        """Cache one generated bundle and initialize its untried-edge mask."""
        node.flow_edge_bundle = edge_bundle
        node.edge_bundle_indices = np.arange(edge_bundle.num_edges, dtype=np.int64)
        node.edge_bundle_mask = np.full((edge_bundle.num_edges,), False, dtype=bool)

    def _select_prefetch_nodes(self, parent_node):
        """Choose uncached nodes to share one batched model inference call."""
        t0 = time.perf_counter()
        nodes = [parent_node]
        seen_node_ids = {parent_node.id}
        while (len(nodes) < self.flow_prefetch_batch_size and
               self.uncached_flow_node_ids):
            node_id = self.uncached_flow_node_ids.popleft()
            if node_id in seen_node_ids or node_id not in self.tree.nodes:
                continue
            node = self.tree.nodes[node_id]["value"]
            if node.edge_bundle_indices is not None:
                continue
            nodes.append(node)
            seen_node_ids.add(node_id)
        self.profile["flow_prefetch_select_s"] += time.perf_counter() - t0
        return nodes

    def _ensure_flow_edges_for_node(self, parent_node):
        """Generate and cache the parent bundle, optionally prefetching peers."""
        if parent_node.edge_bundle_indices is not None:
            self.profile["flow_cache_hits"] += 1
            return

        self.profile["flow_cache_misses"] += 1
        nodes = self._select_prefetch_nodes(parent_node)
        states = np.stack([node.state for node in nodes])
        t0 = time.perf_counter()
        edge_bundles = self.flow_edge_generator.sample_batch(
            states,
            num_edges=self.max_num_edges_per_node,
        )
        self.profile["flow_generation_s"] += time.perf_counter() - t0
        self.profile["flow_generation_calls"] += 1
        self.profile["flow_generated_bundles"] += len(edge_bundles)
        for node, edge_bundle in zip(nodes, edge_bundles):
            if node.edge_bundle_indices is None:
                self._attach_flow_edge_bundle(node, edge_bundle)

    def _try_edge_from_bundle(self, edge_bundle_index, parent_node,
        parent_node_id, mask_index, curr_edge_mask, debug_prefix=""):
        """Propagate, validate, and possibly add one ranked candidate edge.

        Returns True when the edge either adds a normal node or reaches the
        goal. Returns False after any control/dynamics/collision rejection.
        Every sequence is tried at most once and released afterward.
        """
        t0 = time.perf_counter()
        self.profile["try_edge_calls"] += 1
        edge_bundle = parent_node.flow_edge_bundle
        if edge_bundle is None:
            raise RuntimeError("Flow edge bundle was not generated for this node.")

        # Franka bundles contain variable-length sequences; SOC bundles retain
        # the older single-constant-control representation.
        sequence_edge = hasattr(edge_bundle, "action_sequences")
        action = edge_bundle.actions[edge_bundle_index]
        if sequence_edge:
            available_steps = len(action)
            action = np.array(action, dtype=np.float32, copy=True)
            # Reject acceleration or inter-step jerk violations before the more
            # expensive state rollout and self-collision checks.
            if hasattr(self.agent, "action_sequence_violation"):
                violation = self.agent.action_sequence_violation(
                    action, edge_bundle.action_dt
                )
                if violation is not None:
                    if violation == "jerk":
                        self.profile["sequence_jerk_rejections"] += 1
                        if hasattr(self.agent, "jerk_rejections"):
                            self.agent.jerk_rejections += 1
                    else:
                        self.profile["sequence_acceleration_rejections"] += 1
                        if hasattr(self.agent, "acceleration_rejections"):
                            self.agent.acceleration_rejections += 1
                    edge_bundle.release_edge(edge_bundle_index)
                    curr_edge_mask[mask_index] = True
                    self.profile["try_edge_s"] += time.perf_counter() - t0
                    return False
            self.profile["sequence_edges_executed"] += 1
            self.profile["sequence_executed_steps"] += len(action)
            self.profile["sequence_available_steps"] += available_steps
            timestep = float(len(action) * edge_bundle.action_dt)
            # This propagation, not the FM-predicted endpoint, determines the
            # state and intermediate waypoints considered for tree insertion.
            new_state, path_to_new_state = self.agent.get_next_state_sequence(
                parent_node.state, action, edge_bundle.action_dt
            )
            edge_bundle.release_edge(edge_bundle_index)
        else:
            timestep = float(edge_bundle.timesteps[edge_bundle_index])
            num_record_steps = max(1, round(timestep / self.minimum_time_step))
            new_state, path_to_new_state = self.agent.get_next_state(
                parent_node.state, action, timestep, num_steps=num_record_steps
            )

        # A sequence can safely reach the goal before a later generated
        # waypoint becomes invalid.  Detect and validate that prefix first.
        if sequence_edge:
            goal_index = next(
                (
                    index
                    for index, state in enumerate(path_to_new_state)
                    if self.reached_goal(
                        state, self.goal, self.goal_radius, self.agent
                    )[0]
                ),
                None,
            )
            if goal_index is not None:
                prefix = path_to_new_state[: goal_index + 1]
                prefix_action = action[: goal_index + 1]
                prefix_time = float((goal_index + 1) * edge_bundle.action_dt)
                prefix_valid = self.isvalid(
                    prefix, self.agent.radius, self.env.size,
                    self.static_circular_obstacles, self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                    self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                    self.dynamic_agent_clearance, self.env.boundary_buffer,
                    parent_node.time_elapsed, prefix_time, self.minimum_time_step,
                )
                total_elapsed_time = parent_node.time_elapsed + prefix_time
                if prefix_valid and not self.dynamic_col_checker_to_end(
                    prefix[-1], self.agent.radius, self.dynamic_agent_obstacles,
                    self.dynamic_agent_clearance, total_elapsed_time,
                    self.minimum_time_step,
                ):
                    edge_cost = self.cost(
                        self.env, self.agent, parent_node.state, prefix_action,
                        prefix_time, prefix,
                    )
                    total_cost = parent_node.cost_so_far + edge_cost
                    new_node_id = self.add_rrt_node(
                        prefix[-1], parent_node_id, prefix_action, prefix_time,
                        prefix, total_elapsed_time, total_cost,
                    )
                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_time = total_elapsed_time
                    self.path_cost = total_cost
                    curr_edge_mask[mask_index] = True
                    self.profile["try_edge_s"] += time.perf_counter() - t0
                    return True
                if not prefix_valid:
                    curr_edge_mask[mask_index] = True
                    self.profile["try_edge_s"] += time.perf_counter() - t0
                    return False

        # Validate every propagated waypoint against the agent's joint/velocity
        # limits, self-collision checker, and any environment constraints.
        accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.agent.dynamic_limit_indices,
                        self.agent.dynamic_limit_values, self.env.obstacle_buffer,
                        self.dynamic_agent_clearance,
                        self.env.boundary_buffer, parent_node.time_elapsed,
                        timestep, self.minimum_time_step)

        if not accept_new_node:
            curr_edge_mask[mask_index] = True
            if self.debug_flag:
                print(f"{debug_prefix}Flow EB edge invalid. Trying another candidate.")
                print("Invalid State :", new_state)
            self.profile["try_edge_s"] += time.perf_counter() - t0
            return False

        reached_goal_flag, goal_distance = self.reached_goal(
            new_state, self.goal, self.goal_radius, self.agent)

        if reached_goal_flag:
            total_elapsed_time = parent_node.time_elapsed + timestep
            if not self.dynamic_col_checker_to_end(
                new_state, self.agent.radius, self.dynamic_agent_obstacles,
                self.dynamic_agent_clearance, total_elapsed_time,
                self.minimum_time_step,
            ):
                edge_cost = self.cost(self.env, self.agent, parent_node.state,
                                      action, timestep, path_to_new_state)
                total_cost = parent_node.cost_so_far + edge_cost
                new_node_id = self.add_rrt_node(
                    new_state, parent_node_id, action, timestep, path_to_new_state,
                    total_elapsed_time, total_cost)
                self.path_found = True
                self.goal_node_id = new_node_id
                self.path_time = total_elapsed_time
                self.path_cost = total_cost
                curr_edge_mask[mask_index] = True
                self.profile["try_edge_s"] += time.perf_counter() - t0
                return True

        # Preserve the inherited near-goal safeguard: if the endpoint is close,
        # scan intermediate waypoints in case the path crossed the goal region.
        if not reached_goal_flag and goal_distance < self.threshold:
            total_elapsed_time = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path_to_new_state):
                total_elapsed_time += self.minimum_time_step
                goal_flag, _ = self.reached_goal(
                    intermediate_state, self.goal, self.goal_radius, self.agent)
                if goal_flag:
                    if self.dynamic_col_checker_to_end(
                        intermediate_state, self.agent.radius,
                        self.dynamic_agent_obstacles, self.dynamic_agent_clearance,
                        total_elapsed_time, self.minimum_time_step,
                    ):
                        continue
                    modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                    new_path_to_new_state = path_to_new_state[:index + 1]
                    edge_cost = self.cost(self.env, self.agent, parent_node.state,
                                          action, modified_edge_time,
                                          new_path_to_new_state)
                    total_cost = parent_node.cost_so_far + edge_cost
                    stored_action = action[: index + 1] if sequence_edge else action
                    new_node_id = self.add_rrt_node(
                        intermediate_state, parent_node_id, stored_action,
                        modified_edge_time, new_path_to_new_state,
                        total_elapsed_time, total_cost)
                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time
                    curr_edge_mask[mask_index] = True
                    self.profile["try_edge_s"] += time.perf_counter() - t0
                    return True

        # A valid non-goal endpoint becomes one ordinary RRT tree node.
        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                              action, timestep, path_to_new_state)
        total_cost = parent_node.cost_so_far + edge_cost
        total_elapsed_time = parent_node.time_elapsed + timestep
        self.add_rrt_node(new_state, parent_node_id, action, timestep,
                          path_to_new_state, total_elapsed_time, total_cost)
        curr_edge_mask[mask_index] = True
        self.profile["try_edge_s"] += time.perf_counter() - t0
        return True

    def _sort_sequence_edges(self, edge_bundle, random_point, curr_edge_indices,
                             curr_edge_mask):
        """Rank untried edges using the FM-predicted relative terminal state.

        This deliberately performs no control propagation. Propagation and all
        validity checks happen later, in sorted order, inside
        ``_try_edge_from_bundle``.
        """
        target = self._distance_metric_state(random_point)
        count = len(curr_edge_indices)
        num_valid = 0
        for local_index, edge_index in enumerate(curr_edge_indices):
            if curr_edge_mask[local_index]:
                self.distance_array[local_index] = np.inf
                continue
            edge_index = int(edge_index)
            predicted_final_state = edge_bundle.final_states[edge_index]
            if not np.isfinite(predicted_final_state).all():
                self.distance_array[local_index] = np.inf
                curr_edge_mask[local_index] = True
                edge_bundle.release_edge(edge_index)
                continue
            predicted_endpoint = self._distance_metric_state(
                predicted_final_state
            )
            self.distance_array[local_index] = float(
                np.linalg.norm(predicted_endpoint - target)
            )
            num_valid += 1
        order = np.argsort(self.distance_array[:count], kind="stable")
        return order[:num_valid], num_valid

    def _try_random_control_profiled(self, parent_node, parent_node_id, random_point):
        """Run the inherited random-control fallback with timing and deadline checks."""
        if self._deadline_reached():
            return False
        t0 = time.perf_counter()
        self.profile["random_control_calls"] += 1
        result = self._try_random_control(parent_node, parent_node_id, random_point)
        self.profile["random_control_s"] += time.perf_counter() - t0
        return result

    def extend_tree(self, parent_node_id, parent_node, random_point):
        """Perform one flow-guided extension toward a sampled 14D target.

        Epsilon exploration can bypass flow entirely. Otherwise, candidates are
        generated/cached, ranked by predicted endpoint, and tried in order. If
        all selected flow edges fail, random controls preserve exploration.
        """
        self.profile["extend_calls"] += 1
        if self._deadline_reached():
            return
        # Occasional pure-random extensions prevent total dependence on model
        # support and retain probabilistic exploration behavior.
        if self.epsilon_random > 0.0 and self.rng.random() < self.epsilon_random:
            for _ in range(self.num_random_edges):
                if self._try_random_control_profiled(parent_node, parent_node_id, random_point):
                    return
            return

        self._ensure_flow_edges_for_node(parent_node)
        if self._deadline_reached():
            return
        eb = parent_node.flow_edge_bundle
        curr_edge_indices = parent_node.edge_bundle_indices
        curr_edge_mask = parent_node.edge_bundle_mask

        t0 = time.perf_counter()
        if hasattr(eb, "action_sequences"):
            sorted_indices, num_valid_edges = self._sort_sequence_edges(
                eb, random_point, curr_edge_indices, curr_edge_mask
            )
        else:
            sorted_indices, num_valid_edges = self.sort_edges(
                parent_node.state, random_point, eb.start_states, eb.final_states,
                curr_edge_indices, curr_edge_mask, self.distance_array)
        self.profile["sort_edges_s"] += time.perf_counter() - t0

        # num_skip_edges is the maximum number of ranked FM candidates that one
        # extension may validate before falling back to random control.
        trial_count = min(num_valid_edges, self.num_skip_edges)
        # Traverse the nearest predicted outcomes in strict sorted order. The
        # first edge whose full propagated path is valid becomes the extension.
        for x in sorted_indices[:trial_count]:
            if self._deadline_reached():
                return
            edge_bundle_index = curr_edge_indices[x]
            if self._try_edge_from_bundle(
                edge_bundle_index, parent_node, parent_node_id, x, curr_edge_mask,
                debug_prefix="[flow-sorted] ",
            ):
                return

        for _ in range(self.num_random_edges):
            if self._deadline_reached():
                return
            if self._try_random_control_profiled(parent_node, parent_node_id, random_point):
                return

    def print_profile(self):
        """Print accumulated generation, sorting, validation, and fallback timings."""
        generator_profile = self.flow_edge_generator.profile
        print("FlowEBRRT profile:")
        for key in (
            "extend_calls",
            "flow_generation_calls",
            "flow_generated_bundles",
            "flow_cache_hits",
            "flow_cache_misses",
            "try_edge_calls",
            "random_control_calls",
            "sequence_edges_executed",
            "sequence_executed_steps",
            "sequence_available_steps",
            "sequence_acceleration_rejections",
            "sequence_jerk_rejections",
        ):
            print(f"  {key}: {self.profile[key]}")
        for key in (
            "flow_generation_s",
            "flow_prefetch_select_s",
            "sort_edges_s",
            "try_edge_s",
            "random_control_s",
        ):
            print(f"  {key}: {self.profile[key]:.6f}")
        print("  generator_samples:", generator_profile["samples"])
        for key in ("sample_total_s", "model_s", "postprocess_s"):
            print(f"  generator_{key}: {generator_profile[key]:.6f}")


# ---------------------------------------------------------------------------
# Planner factories
# ---------------------------------------------------------------------------


def get_flow_eb_rrt_planner_soc(start, goal, goal_radius, agent, env, *,
                                checkpoint_path="checkpoints/soc_edge_flow/soc_edge_flow_k32_v1/best.pt",
                                device="cuda:1",
                                sample_steps=16,
                                flow_prefetch_batch_size=1):
    """Factory mirroring the SOC KiTE-RRT setup, but using generated Flow edges."""
    flow_generator = SOCFlowEdgeGenerator(
        checkpoint_path=checkpoint_path,
        device=device,
        sample_steps=sample_steps,
        clamp_outputs=True,
        seed=42 + agent.id,
    )
    return FlowEBRRT(
        start=start, goal=goal, goal_radius=goal_radius,
        env=env, agent=agent, flow_edge_generator=flow_generator,
        use_fixed_sampling_time=False,
        sampling_time_step=2.0,
        minimum_time_step=0.1,
        max_iter=10000,
        planning_time=600.0,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=32,
        flow_prefetch_batch_size=flow_prefetch_batch_size,
        num_skip_edges=10,
        num_random_edges=1,
        epsilon_random=0.01,
        udf_seed=0,
        debug_flag=False,
        print_logs=False,
    )


def get_flow_eb_rrt_planner_franka(
    start,
    goal,
    goal_radius,
    agent,
    env,
    *,
    checkpoint_path,
    device="auto",
    sample_steps=16,
    flow_prefetch_batch_size=16,
    minimum_sequence_prefix_steps=5,
    truncate_sequence_to_target=False,
    planning_time=30.0,
    max_iter=10_000_000,
    max_random_edge_time=0.30,
    num_skip_edges=32,
    num_random_edges=1,
    epsilon_random=0.05,
    goal_sampling_probability=0.30,
    seed=0,
):
    """Build a fully wired Franka FlowEBRRT planner.

    The agent supplies 14D sampling, normalized nearest-neighbor distance,
    double-integrator propagation, goal checking in 7D joint-position space,
    and all validity/cost functions. The generator supplies 32 learned edges
    per queried node. Random fallback duration is controlled separately by
    ``max_random_edge_time``; FM duration comes from each predicted step count.
    """
    flow_generator = FrankaFlowEdgeGenerator(
        checkpoint_path=checkpoint_path,
        device=device,
        sample_steps=sample_steps,
        clamp_outputs=False,
        seed=42 + int(seed),
    )
    return FlowEBRRT(
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        env=env,
        agent=agent,
        flow_edge_generator=flow_generator,
        use_fixed_sampling_time=False,
        sampling_time_step=max_random_edge_time,
        minimum_time_step=flow_generator.dt,
        max_iter=max_iter,
        planning_time=planning_time,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=flow_generator.set_size,
        flow_prefetch_batch_size=flow_prefetch_batch_size,
        minimum_sequence_prefix_steps=minimum_sequence_prefix_steps,
        truncate_sequence_to_target=truncate_sequence_to_target,
        num_skip_edges=num_skip_edges,
        num_random_edges=num_random_edges,
        epsilon_random=epsilon_random,
        udf_seed=seed,
        goal_sampling_probability=goal_sampling_probability,
        debug_flag=False,
        print_logs=False,
    )
