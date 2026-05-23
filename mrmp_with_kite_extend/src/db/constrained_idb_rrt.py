"""
Generic constrained iterative discontinuity-bounded RRT (iDb-RRT) wrapper.

This outer loop reuses constrained or unconstrained Db-RRT planners and keeps
optimization optional/guarded:
    - build planner for current (delta, primitive budget)
    - optionally set dynamic constraints on the planner
    - run raw search
    - optionally optimize
    - if optimization succeeds and passes optional dynamic-constraint validation,
      return optimized result; otherwise continue outer loop

The same class works when no dynamic constraints are provided.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ConstrainedIdbRRTResult:
    solved: bool = False
    solved_raw: bool = False
    solved_optimized: bool = False
    outer_iters: int = 0
    wall_time: float = 0.0
    best_raw_cost: float = math.inf
    best_opt_cost: float = math.inf
    delta_history: list[float] = field(default_factory=list)
    primitive_history: list[int] = field(default_factory=list)
    raw_planner: Any = None
    optimized_result: Any = None
    optimized_path_view: Any = None
    last_opt_failed_constraint_check: bool = False


class ConstrainedIdbRRTPlanner:
    """
    Generic outer-loop scheduler for constrained iDb-RRT.

    Parameters
    ----------
    planner_factory:
        Callable taking keyword args `delta`, `max_motions` and returning a
        Db-RRT-like planner.
    optimizer_function:
        Callable taking a planner and returning an object with at least boolean
        attribute `feasible`.
    optimizer_constraint_checker:
        Optional callable `(planner, opt_result, constraints) -> bool` used to
        validate optimized trajectories against dynamic-agent constraints.
        If None, no post-optimization dynamic-constraint check is applied.
    """

    def __init__(
        self,
        *,
        planner_factory: Callable[..., Any],
        optimizer_function: Optional[Callable[[Any], Any]] = None,
        optimizer_constraint_checker: Optional[Callable[[Any, Any, Any], bool]] = None,
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
        self.optimizer_constraint_checker = optimizer_constraint_checker
        self.delta_0 = float(delta_0)
        self.delta_rate = float(delta_rate)
        self.num_primitives_0 = int(num_primitives_0)
        self.num_primitives_rate = float(num_primitives_rate)
        self.max_outer_iters = int(max_outer_iters)
        self.planning_time = float(planning_time)
        self.print_logs = print_logs
        self.result = ConstrainedIdbRRTResult()

    @staticmethod
    def _maybe_set_constraints(planner: Any, constraints: Any) -> None:
        if constraints is None:
            return
        setter = getattr(planner, "set_constraints", None)
        if callable(setter):
            setter(constraints)

    def _opt_result_is_feasible(self, opt_result: Any) -> bool:
        return bool(getattr(opt_result, "feasible", False))

    def _check_optimized_constraints(self, planner: Any, opt_result: Any, constraints: Any) -> bool:
        if constraints is None:
            return True
        if self.optimizer_constraint_checker is None:
            return True
        return bool(self.optimizer_constraint_checker(planner, opt_result, constraints))

    def plan_path(self, constraints: Any = None) -> ConstrainedIdbRRTResult:
        result = ConstrainedIdbRRTResult()
        start_time = time.time()

        delta = self.delta_0
        max_motions = self.num_primitives_0

        for outer_iter in range(self.max_outer_iters):
            if time.time() - start_time >= self.planning_time:
                break

            result.delta_history.append(delta)
            result.primitive_history.append(max_motions)

            planner = self.planner_factory(delta=delta, max_motions=max_motions)
            self._maybe_set_constraints(planner, constraints)
            planner.plan_path()
            result.outer_iters = outer_iter + 1

            if planner.path_found:
                result.solved_raw = True
                if planner.path_cost < result.best_raw_cost:
                    result.best_raw_cost = float(planner.path_cost)
                    result.raw_planner = planner

                # If no optimizer is supplied, raw constrained Db-RRT is the output.
                if self.optimizer_function is None:
                    result.solved = True
                    break

                opt_result = self.optimizer_function(planner)
                if self._opt_result_is_feasible(opt_result):
                    if self._check_optimized_constraints(planner, opt_result, constraints):
                        result.solved = True
                        result.solved_optimized = True
                        result.optimized_result = opt_result
                        result.optimized_path_view = getattr(opt_result, "path_view", None)
                        result.best_opt_cost = float(getattr(opt_result, "cost", math.inf))
                        result.raw_planner = planner
                        break
                    result.last_opt_failed_constraint_check = True

            delta *= self.delta_rate
            next_max = int(math.ceil(max_motions * self.num_primitives_rate))
            max_motions = max(max_motions + 1, next_max)

        result.wall_time = time.time() - start_time
        self.result = result

        if self.print_logs:
            print(
                f"Constrained iDb-RRT: solved={result.solved}, "
                f"solved_raw={result.solved_raw}, solved_opt={result.solved_optimized}, "
                f"outer_iters={result.outer_iters}, best_raw_cost={result.best_raw_cost}, "
                f"best_opt_cost={result.best_opt_cost}, wall_time={result.wall_time:.3f}s"
            )
            if result.last_opt_failed_constraint_check:
                print("Constrained iDb-RRT: last optimization rejected by dynamic-constraint check")

        return result
