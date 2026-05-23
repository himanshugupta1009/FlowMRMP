"""
Generic iterative discontinuity-bounded RRT (iDb-RRT) wrapper.

This outer loop is system-agnostic:
    - build a Db-RRT planner once
    - reset/reconfigure it for the current (delta, primitive budget)
    - run search
    - if a raw path is found, attempt trajectory repair/optimization
    - if optimization is feasible, stop
    - otherwise reduce delta and increase the primitive budget

The search planner and optimizer are injected, which lets the same wrapper work
for multiple systems such as unicycle and quadcopter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class IdbRRTResult:
    solved: bool = False
    solved_raw: bool = False
    outer_iters: int = 0
    wall_time: float = 0.0
    best_raw_cost: float = math.inf
    best_opt_cost: float = math.inf
    delta_history: list[float] = field(default_factory=list)
    primitive_history: list[int] = field(default_factory=list)
    raw_planner: Any = None
    optimized_result: Any = None
    optimized_path_view: Any = None


class IdbRRTPlanner:
    """
    Generic outer-loop scheduler for iDb-RRT.

    Parameters
    ----------
    planner_factory:
        Callable taking keyword arguments `delta` and `max_motions`, returning a
        raw Db-RRT-like planner instance. The returned planner may support
        `reset_for_iteration(...)` for reuse across outer iterations.
    optimizer_function:
        Callable taking a planner and returning an object with at least a
        boolean `feasible` attribute. If present, `path_view` is recorded as the
        final optimized path.
    """

    def __init__(
        self,
        *,
        planner_factory: Callable[..., Any],
        optimizer_function: Callable[[Any], Any],
        delta_0: float,
        delta_rate: float = 0.95,
        num_primitives_0: int,
        num_primitives_rate: float = 1.5,
        max_outer_iters: int = 10,
        planning_time: float = 120.0,
        print_logs: bool = False,
    ):
        if delta_0 <= 0.0:
            raise ValueError("delta_0 must be positive")
        if not (0.0 < delta_rate < 1.0):
            raise ValueError("delta_rate must be in (0, 1)")
        if num_primitives_0 <= 0:
            raise ValueError("num_primitives_0 must be positive")
        if num_primitives_rate < 1.0:
            raise ValueError("num_primitives_rate must be at least 1")
        if max_outer_iters <= 0:
            raise ValueError("max_outer_iters must be positive")

        self.planner_factory = planner_factory
        self.optimizer_function = optimizer_function
        self.delta_0 = float(delta_0)
        self.delta_rate = float(delta_rate)
        self.num_primitives_0 = int(num_primitives_0)
        self.num_primitives_rate = float(num_primitives_rate)
        self.max_outer_iters = int(max_outer_iters)
        self.planning_time = float(planning_time)
        self.print_logs = print_logs

        self.result = IdbRRTResult()

    def plan_path(self) -> IdbRRTResult:
        result = IdbRRTResult()
        start_time = time.time()

        delta = self.delta_0
        max_motions = self.num_primitives_0
        planner = None

        for outer_iter in range(self.max_outer_iters):
            if time.time() - start_time >= self.planning_time:
                break

            result.delta_history.append(delta)
            result.primitive_history.append(max_motions)

            if planner is None:
                planner = self.planner_factory(delta=delta, max_motions=max_motions)
            elif hasattr(planner, "reset_for_iteration"):
                planner.reset_for_iteration(delta=delta, max_motions=max_motions)
            else:
                planner = self.planner_factory(delta=delta, max_motions=max_motions)
            planner.plan_path()

            result.outer_iters = outer_iter + 1

            if planner.path_found:
                result.solved_raw = True
                if planner.path_cost < result.best_raw_cost:
                    result.best_raw_cost = float(planner.path_cost)
                    result.raw_planner = planner

                opt_result = self.optimizer_function(planner)
                if getattr(opt_result, "feasible", False):
                    result.solved = True
                    result.optimized_result = opt_result
                    result.optimized_path_view = getattr(opt_result, "path_view", None)
                    result.best_opt_cost = float(getattr(opt_result, "cost", math.inf))
                    result.raw_planner = planner
                    break

            delta *= self.delta_rate
            next_max = int(math.ceil(max_motions * self.num_primitives_rate))
            max_motions = max(max_motions + 1, next_max)

        result.wall_time = time.time() - start_time
        self.result = result

        if self.print_logs:
            print(
                f"iDb-RRT: solved={result.solved}, "
                f"solved_raw={result.solved_raw}, "
                f"outer_iters={result.outer_iters}, "
                f"best_raw_cost={result.best_raw_cost}, "
                f"best_opt_cost={result.best_opt_cost}, "
                f"wall_time={result.wall_time:.3f}s"
            )

        return result
