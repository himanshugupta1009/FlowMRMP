# isst.py
import heapq
import math
import numpy as np

from typing import Callable, List, Optional, Tuple
from sst import SST  # import your SST class (the file you shared)

class ISST(SST):
    """
    Informed-SST (iSST) on top of your SST matrices & API.
    Key additions:
      - Open set (min-heap) ordered by f = g + h
      - Quality gate on selection (A*-like)
      - Blossom expansion: try multiple controls; keep best-h child
      - Branch-and-bound with current best goal cost
    """

    def __init__(
        self,
        *,
        start,
        goal,
        goal_radius,
        env,
        agent,
        isvalid_function,
        cost_function,
        reached_goal_function,
        random_point_function,
        # ---- iSST extras ----
        heuristic_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        pos_dims: int = 2,                 # if you use a positional heuristic by default
        blossom_M0: int = 8,               # initial blossom budget M
        blossom_decay: int = 2,            # subtract per revisit (floored at 1)
        quality_temp: float = 1.0,         # raise quality to this power
        maneuvers: Optional[List[Tuple[np.ndarray, float]]] = None,  # [(action, duration), ...]
        **kwargs
    ):
        super().__init__(
            start=start, goal=goal, goal_radius=goal_radius,
            env=env, agent=agent,
            isvalid_function=isvalid_function,
            cost_function=cost_function,
            reached_goal_function=reached_goal_function,
            random_point_function=random_point_function,
            **kwargs
        )

        # Heuristic: user-supplied or fallback to Euclidean on first pos_dims
        self.heuristic_fn = heuristic_function or (lambda x, g: float(np.linalg.norm(x[:pos_dims] - g[:pos_dims])))
        self.pos_dims = pos_dims

        # Blossom / selection params
        self.M0 = max(1, int(blossom_M0))
        self.blossom_decay = max(1, int(blossom_decay))
        self.quality_temp = float(quality_temp)
        self.maneuvers = maneuvers or []

        # Open sets
        self._open = []        # heap of (f, tie, index)
        self._open_aux = []    # list of indices moved to next cycle
        self._tie = 0

        # Per-node dynamic fields (parallel arrays sized with node matrix capacity)
        cap = self._node_matrix.state.shape[0]
        self._h = np.full(cap, np.inf, dtype=np.float64)
        self._f = np.full(cap, np.inf, dtype=np.float64)
        self._priority_uses = np.zeros(cap, dtype=np.int32)
        self._blossom_budget = np.full(cap, self.M0, dtype=np.int32)

        # Best goal (branch-and-bound)
        self._best_goal_index: Optional[int] = None
        self._best_goal_cost: float = math.inf

        # Cache start heuristic
        self._h_start = self._heuristic(self.start)

    # ------------- helpers -------------
    def _heuristic(self, x: np.ndarray) -> float:
        return float(self.heuristic_fn(x, self.goal))

    def _push_open(self, idx: int):
        self._h[idx] = self._heuristic(self._node_matrix.state[idx])
        self._f[idx] = self._node_matrix.cost[idx] + self._h[idx]
        self._tie += 1
        heapq.heappush(self._open, (self._f[idx], self._tie, idx))

    def _recycle_open(self):
        for idx in self._open_aux:
            self._tie += 1
            heapq.heappush(self._open, (self._f[idx], self._tie, idx))
        self._open_aux.clear()

    def _quality_accept(self, idx: int) -> bool:
        """
        Probabilistic gate like in the paper:
          if no solution: q = h(x0)/h(x)
          else: q = (1 / (g/g* + h/h0))^priority
        """
        g = float(self._node_matrix.cost[idx])
        h = float(self._h[idx])
        h0 = max(self._h_start, 1e-12)

        if self._best_goal_index is None:
            q = h0 / max(h, 1e-12)
        else:
            g_star = max(self._best_goal_cost, 1e-12)
            base = 1.0 / ((g / g_star) + (h / h0))
            q = base ** max(1, int(self._priority_uses[idx]))

        if self.quality_temp != 1.0:
            q = q ** self.quality_temp

        q = min(1.0, max(0.0, q))
        return self.rng.random() <= q

    def _select_node(self) -> Optional[int]:
        if not self._open:
            self._recycle_open()
            if not self._open:
                return None

        while self._open:
            _, _, idx = heapq.heappop(self._open)
            if self._quality_accept(idx):
                return idx
            else:
                self._open_aux.append(idx)

        return None

    def _branch_and_bound_ok(self, parent_idx: int, new_state: np.ndarray, edge_cost: float) -> bool:
        g_new = float(self._node_matrix.cost[parent_idx] + edge_cost)
        h_new = self._heuristic(new_state)
        if self._best_goal_index is not None:
            return (g_new + h_new) < (self._best_goal_cost - 1e-12)
        return True

    def _update_goal_if_better(self, idx: int):
        g = float(self._node_matrix.cost[idx])
        if g < self._best_goal_cost:
            self._best_goal_cost = g
            self._best_goal_index = idx

    def _update_witness_and_activation(self, new_idx: int):
        """
        Reuse your existing SST witness logic (unchanged),
        but we’ve already computed/added the node.
        """
        sst_nodes = self._node_matrix
        sst_witnesses = self._witness_matrix
        new_state = sst_nodes.state[new_idx]

        # Find nearest witness
        if sst_witnesses.count == 0:
            # no witnesses yet
            new_witness_index = self.add_sst_witness(new_state, new_idx)
            sst_nodes.active[new_idx] = 1
            return

        # compute nearest witness
        # (mirror your njit function in python for simplicity here)
        best = 0
        best_d = np.linalg.norm(
            sst_witnesses.state[0, :self.distance_metric_state_size] - new_state[:self.distance_metric_state_size]
        )
        for i in range(1, sst_witnesses.count):
            d = np.linalg.norm(
                sst_witnesses.state[i, :self.distance_metric_state_size] - new_state[:self.distance_metric_state_size]
            )
            if d < best_d:
                best = i
                best_d = d

        if best_d > self.prune_radius:
            self.add_sst_witness(new_state, new_idx)
            sst_nodes.active[new_idx] = 1
        else:
            rep = sst_witnesses.rep_index[best]
            if sst_nodes.cost[new_idx] < sst_nodes.cost[rep]:
                sst_nodes.active[rep] = 0
                sst_witnesses.rep_index[best] = new_idx
                sst_nodes.active[new_idx] = 1

    # ------------- blossom expansion -------------
    def _blossom_once(self, parent_idx: int) -> Optional[int]:
        """
        Try multiple controls; filter by collision and BnB; keep child with best h.
        Returns new node index or None if nothing valid.
        """
        sst_nodes = self._node_matrix
        parent_state = sst_nodes.state[parent_idx]
        parent_time_elapsed = sst_nodes.time_elapsed[parent_idx]
        parent_cost = sst_nodes.cost[parent_idx]

        M = max(1, int(self._blossom_budget[parent_idx]))
        # decay for next time we revisit this parent
        if self._blossom_budget[parent_idx] > 1:
            self._blossom_budget[parent_idx] = max(1, self._blossom_budget[parent_idx] - self.blossom_decay)

        candidates = []

        def try_edge(action: np.ndarray, duration: float):
            num_record_steps = max(1, int(round(duration / self.minimum_time_step)))
            new_state, path_to_new_state = self.agent.get_next_state(
                parent_state, action, duration, num_steps=num_record_steps
            )

            # collision check (your fast preprocessed isvalid)
            ok = self.isvalid(
                path_to_new_state, self.agent.radius, self.env.size,
                self.static_circular_obstacles, self.static_rectangular_obstacles,
                self.dynamic_agent_obstacles, self.env.obstacle_buffer,
                self.env.boundary_buffer, parent_time_elapsed, duration, self.minimum_time_step
            )
            if not ok:
                return

            # edge cost + BnB
            edge_cost = self.cost(self.env, self.agent, parent_state, action, duration, path_to_new_state)
            if not self._branch_and_bound_ok(parent_idx, new_state, edge_cost):
                return

            # goal snap along the discretized path (preserve your behavior)
            reached, goal_dist = self.reached_goal(new_state, self.goal, self.goal_radius, self.agent)
            if not reached and goal_dist < self.threshold:
                # check intermediate hits
                total_elapsed_time = parent_time_elapsed
                for ii, s in enumerate(path_to_new_state):
                    total_elapsed_time += self.minimum_time_step
                    flag, _ = self.reached_goal(s, self.goal, self.goal_radius, self.agent)
                    if flag:
                        mod_time = total_elapsed_time - parent_time_elapsed
                        new_path = path_to_new_state[:ii + 1]
                        edge_cost2 = self.cost(self.env, self.agent, parent_state, action, mod_time, new_path)
                        if not self._branch_and_bound_ok(parent_idx, s, edge_cost2):
                            return
                        candidates.append(("goal", s, action, mod_time, new_path, edge_cost2))
                        return

            candidates.append(("normal", new_state, action, duration, path_to_new_state, edge_cost))

        # 1) maneuvers first (up to M)
        used = 0
        for (a_m, t_m) in self.maneuvers:
            if used >= M:
                break
            try_edge(a_m, t_m)
            used += 1

        # 2) random controls to fill remaining budget
        while used < M:
            a = self.agent.get_random_action(self.rng)
            t = self.get_time()
            try_edge(a, t)
            used += 1

        if not candidates:
            return None

        # choose by minimal heuristic h
        best_idx = None
        best_h = math.inf
        best_pack = None

        for (mode, child_state, action, duration, path, edge_cost) in candidates:
            h = self._heuristic(child_state)
            if h < best_h:
                best_h = h
                best_idx = -1
                best_pack = (mode, child_state, action, duration, path, edge_cost)

        # commit the best child into matrices
        mode, child_state, action, duration, path, edge_cost = best_pack
        total_elapsed_time = parent_time_elapsed + duration
        total_cost = parent_cost + edge_cost

        new_idx = self.add_sst_node(child_state, parent_idx, action, duration, path, total_elapsed_time, total_cost)
        # set dynamic fields for selection
        self._h[new_idx] = best_h
        self._f[new_idx] = total_cost + best_h
        self._priority_uses[new_idx] = 0
        self._blossom_budget[new_idx] = self.M0

        # witness activation & replacement (your logic)
        self._update_witness_and_activation(new_idx)

        # open-list candidate
        self._push_open(new_idx)

        # goal update
        if mode == "goal":
            self.path_found = True
            self.goal_node_id = new_idx
            self.path_cost = total_cost
            self.path_time = total_elapsed_time
            self._update_goal_if_better(new_idx)
        else:
            # check endpoint goal
            reached_goal_flag, _ = self.reached_goal(child_state, self.goal, self.goal_radius, self.agent)
            if reached_goal_flag:
                self.path_found = True
                self.goal_node_id = new_idx
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                self._update_goal_if_better(new_idx)

        return new_idx

    # ------------- main loop -------------
    def plan_path(self):
        """
        iSST planning loop. Does not sample random target points; instead
        it grows from the most promising frontier nodes in an A*-like manner
        with blossom expansions and BnB pruning.
        """
        # reset SST bookkeeping
        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.last_added_witness_id = -1
        self.reset_tree()

        # clear open lists and dynamic arrays
        self._open.clear()
        self._open_aux.clear()
        cap = self._node_matrix.state.shape[0]
        self._h[:], self._f[:], self._priority_uses[:], self._blossom_budget[:] = (
            np.inf, np.inf, 0, self.M0
        )
        self._h_start = self._heuristic(self.start)
        self._best_goal_index, self._best_goal_cost = None, math.inf

        # add root node
        state_len = len(self.start)
        empty_path = np.empty((0, state_len), dtype=np.float64)
        root_idx = self.add_sst_node(self.start, -1, self.dummy_root_action, 0.0, empty_path, 0.0, 0.0)
        self._node_matrix.active[root_idx] = 1
        self.add_sst_witness(self.start, root_idx)

        # setup root fields
        self._h[root_idx] = self._h_start
        self._f[root_idx] = self._h_start
        self._priority_uses[root_idx] = 0
        self._blossom_budget[root_idx] = self.M0
        self._push_open(root_idx)

        # run
        import time
        start_time = time.time()
        iters = 0

        while iters <= self.max_iter:
            # time budget?
            if (time.time() - start_time) >= self.planning_time:
                break
            # selection
            sel_idx = self._select_node()
            if sel_idx is None:
                # nothing to expand in this cycle
                iters += 1
                continue

            # chained blossom expansions starting from sel_idx
            curr = sel_idx
            progressed = False
            while curr is not None:
                new_idx = self._blossom_once(curr)
                self._priority_uses[sel_idx] += 1
                if new_idx is not None:
                    curr = new_idx
                    progressed = True
                else:
                    curr = None

            # if no child accepted, move selected node to next cycle
            self._open_aux.append(sel_idx)
            # recycle at the end of each top-level iteration
            self._recycle_open()

            if self.path_found:
                # keep going until time/iters if you want further refinement
                # (classic iSST would continue; your SST stops on first goal)
                break

            iters += 1

        # finalize timings like your SST
        self.path_time = round(self.path_time, self.roundoff_digits)



"""
planner = ISST(
    start=..., goal=..., goal_radius=...,
    env=env, agent=agent,
    isvalid_function=isvalid_fn,
    cost_function=cost_fn,
    reached_goal_function=reached_goal_fn,
    random_point_function=random_point_fn,   # not used by iSST but kept for API symmetry
    # iSST extras
    heuristic_function=lambda x, g: np.linalg.norm(x[:2]-g[:2]),  # make sure units match your edge cost
    blossom_M0=8,
    blossom_decay=2,
    maneuvers=[(np.array([0.5, 0.0]), 0.2), (np.array([-0.5, 0.0]), 0.2)]  # optional
)

planner.plan_path()
hires_path = planner.get_high_resolution_path_numpy_array()

"""