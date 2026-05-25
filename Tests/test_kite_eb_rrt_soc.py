#!/usr/bin/env python3
"""Same SOC planning case as FlowEBRRT, using the original KiTE edge bundle."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MRMP_SRC = ROOT_DIR / "mrmp_with_kite_extend" / "src"
for path in (ROOT_DIR, MRMP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mapf_env_square_agent_second_order_car import (  # noqa: E402
    get_kino_TI_eb_rrt_planner_SOC,
)
from Tests.flow_rrt_test_helpers import (  # noqa: E402
    GOAL_RADIUS,
    GOAL_SOC,
    SEED,
    START_SOC,
    apply_common_planner_args,
    get_second_order_car_agent,
    make_test_environment,
    print_planner_result,
)


DEFAULT_EDGE_BUNDLE = (
    "mrmp_with_kite_extend/edge_bundles_unclamped/"
    "eb_second_order_car_kinodynamic_TI_edges_100000.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-bundle", default=DEFAULT_EDGE_BUNDLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--planning-time", type=float, default=60.0)
    parser.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
    parser.add_argument("--max-edges-per-node", type=int, default=32)
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    return parser.parse_args()


def build_planner(args):
    env = make_test_environment()
    agent = get_second_order_car_agent(1)
    planner = get_kino_TI_eb_rrt_planner_SOC(
        START_SOC,
        GOAL_SOC,
        args.goal_radius,
        agent,
        env,
        edge_bundle_file_location=args.edge_bundle,
    )
    planner.max_num_edges_per_node = args.max_edges_per_node
    planner.distance_array = np.zeros((args.max_edges_per_node,), dtype=np.float64)
    return apply_common_planner_args(planner, args)


def main() -> None:
    args = parse_args()
    print("RNG Seed:", args.seed)
    print("Edge bundle:", args.edge_bundle)

    for run_id in range(args.warmup_runs):
        warmup_planner = build_planner(args)
        warmup_planner.plan_path()
        print(f"Warmup run {run_id + 1}/{args.warmup_runs} done")

    planner = build_planner(args)
    start_time = time.perf_counter()
    planner.plan_path()
    wall_time = time.perf_counter() - start_time
    print("Measured wall time:", wall_time)
    print_planner_result(planner)


if __name__ == "__main__":
    main()
