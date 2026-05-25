"""Flow-generated edge-bundle RRT integration for FlowMRMP.

The implementation reuses the MRMP KiTE-RRT planner machinery as a dependency,
but keeps all FlowMRMP-specific code in this repository's top-level ``src``.
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


class FlowEBTreeNode(KinoTIEBTreeNode):
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost):
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,
                         path_from_parent, time_so_far, cost)
        self.flow_edge_bundle = None


class GeneratedEdgeBundle:
    """Small EdgeBundle-like container for one generated node-local bundle."""

    def __init__(self, actions, timesteps, start_states, final_states):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.timesteps = np.asarray(timesteps, dtype=np.float64)
        self.start_states = np.asarray(start_states, dtype=np.float64)
        self.final_states = np.asarray(final_states, dtype=np.float64)
        self.num_edges = int(self.timesteps.shape[0])


class SOCFlowEdgeGenerator:
    """Load a trained SOC edge-set flow model and sample denormalized edges."""

    def __init__(self, *,
                 checkpoint_path,
                 device="cuda:1",
                 sample_steps=16,
                 clamp_outputs=True,
                 seed=123):
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
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return torch.device("cpu")
            if ":" in device:
                index = int(device.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    return torch.device("cpu")
        return torch.device(device)

    def condition_from_state(self, state):
        norm = self.normalization
        return np.array([
            state[3] / float(norm["max_speed"]),
            state[4] / float(norm["max_phi"]),
        ], dtype=np.float32)

    def conditions_from_states(self, states):
        norm = self.normalization
        states = np.asarray(states, dtype=np.float32)
        cond = np.empty((states.shape[0], 2), dtype=np.float32)
        cond[:, 0] = states[:, 3] / float(norm["max_speed"])
        cond[:, 1] = states[:, 4] / float(norm["max_phi"])
        return np.clip(cond, -1.0, 1.0)

    def denormalize_edges(self, edges):
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
        angles = np.arctan2(edges[:, 4], edges[:, 3])
        return np.lexsort((edges[:, 2], angles))

    def set_profile_enabled(self, enabled):
        self.enable_profile = bool(enabled)

    def _sync_if_cuda(self):
        if self.enable_profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _sample_edges_tensor(self, cond):
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
        return self.sample_batch(np.asarray(state, dtype=np.float64)[None, :],
                                 num_edges=num_edges)[0]


class FlowEBRRT(KinoTIEBRRT):
    """KinoTIEBRRT variant that generates node-local edge bundles with flow."""

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
                 num_skip_edges=10,
                 num_random_edges=1,
                 epsilon_random=0.01,
                 udf_seed=77,
                 goal_sampling_probability=0.1,
                 dynamic_agent_clearance=0.0,
                 debug_flag=False,
                 print_logs=False,
                 dynamic_obstacles=None):

        if dynamic_obstacles is None:
            from numba.typed import List
            from numba import types
            dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))

        dummy_edge_bundle = GeneratedEdgeBundle(
            actions=np.zeros((1, 2), dtype=np.float64),
            timesteps=np.ones(1, dtype=np.float64),
            start_states=np.zeros((1, 5), dtype=np.float64),
            final_states=np.zeros((1, 5), dtype=np.float64),
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
        }

    def set_profile_enabled(self, enabled):
        self.flow_edge_generator.set_profile_enabled(enabled)

    def reset_tree(self, some_existing_tree=None):
        super().reset_tree(some_existing_tree)
        self.uncached_flow_node_ids = deque()

    def add_rrt_node(self, *args, **kwargs):
        node_id = super().add_rrt_node(*args, **kwargs)
        self.uncached_flow_node_ids.append(node_id)
        return node_id

    def _attach_flow_edge_bundle(self, node, edge_bundle):
        node.flow_edge_bundle = edge_bundle
        node.edge_bundle_indices = np.arange(edge_bundle.num_edges, dtype=np.int64)
        node.edge_bundle_mask = np.full((edge_bundle.num_edges,), False, dtype=bool)

    def _select_prefetch_nodes(self, parent_node):
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

        t0 = time.perf_counter()
        self.profile["try_edge_calls"] += 1
        edge_bundle = parent_node.flow_edge_bundle
        if edge_bundle is None:
            raise RuntimeError("Flow edge bundle was not generated for this node.")

        action = edge_bundle.actions[edge_bundle_index]
        timestep = float(edge_bundle.timesteps[edge_bundle_index])
        num_record_steps = max(1, round(timestep / self.minimum_time_step))

        new_state, path_to_new_state = self.agent.get_next_state(
            parent_node.state, action, timestep, num_steps=num_record_steps)

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
                    new_node_id = self.add_rrt_node(
                        intermediate_state, parent_node_id, action,
                        modified_edge_time, new_path_to_new_state,
                        total_elapsed_time, total_cost)
                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time
                    curr_edge_mask[mask_index] = True
                    self.profile["try_edge_s"] += time.perf_counter() - t0
                    return True

        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                              action, timestep, path_to_new_state)
        total_cost = parent_node.cost_so_far + edge_cost
        total_elapsed_time = parent_node.time_elapsed + timestep
        self.add_rrt_node(new_state, parent_node_id, action, timestep,
                          path_to_new_state, total_elapsed_time, total_cost)
        curr_edge_mask[mask_index] = True
        self.profile["try_edge_s"] += time.perf_counter() - t0
        return True

    def _try_random_control_profiled(self, parent_node, parent_node_id, random_point):
        t0 = time.perf_counter()
        self.profile["random_control_calls"] += 1
        result = self._try_random_control(parent_node, parent_node_id, random_point)
        self.profile["random_control_s"] += time.perf_counter() - t0
        return result

    def extend_tree(self, parent_node_id, parent_node, random_point):
        self.profile["extend_calls"] += 1
        if self.epsilon_random > 0.0 and self.rng.random() < self.epsilon_random:
            for _ in range(self.num_random_edges):
                if self._try_random_control_profiled(parent_node, parent_node_id, random_point):
                    return
            return

        self._ensure_flow_edges_for_node(parent_node)
        eb = parent_node.flow_edge_bundle
        curr_edge_indices = parent_node.edge_bundle_indices
        curr_edge_mask = parent_node.edge_bundle_mask

        t0 = time.perf_counter()
        sorted_indices, num_valid_edges = self.sort_edges(
            parent_node.state, random_point, eb.start_states, eb.final_states,
            curr_edge_indices, curr_edge_mask, self.distance_array)
        self.profile["sort_edges_s"] += time.perf_counter() - t0

        p = max(1, num_valid_edges // self.num_skip_edges)
        for idx in range(0, num_valid_edges, p):
            x = sorted_indices[idx]
            edge_bundle_index = curr_edge_indices[x]
            if self._try_edge_from_bundle(
                edge_bundle_index, parent_node, parent_node_id, x, curr_edge_mask,
                debug_prefix="[flow-sorted] ",
            ):
                return

        for _ in range(self.num_random_edges):
            if self._try_random_control_profiled(parent_node, parent_node_id, random_point):
                return

    def print_profile(self):
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
