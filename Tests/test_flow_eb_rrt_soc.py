#!/usr/bin/env python3
"""Interactive SOC FlowEBRRT test script.

Run with IPython ``%run`` to keep ``env``, ``agent``, ``planner``, and
related objects available in the shell.
"""

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

from Tests.flow_rrt_test_helpers import (  # noqa: E402
    GOAL_RADIUS,
    SEED,
    apply_common_planner_args,
    get_second_order_car_agent,
    make_flow_eb_rrt_planner_soc,
    make_test_environment,
    print_planner_result,
)


DEFAULT_CHECKPOINT = "checkpoints/soc_edge_flow/soc_edge_flow_k32_v1/best.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--sample-steps", type=int, default=8)
parser.add_argument("--flow-prefetch-batch-size", type=int, default=1)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--max-iter", type=int, default=500)
parser.add_argument("--planning-time", type=float, default=60.0)
parser.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
parser.add_argument("--print-logs", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--warmup-runs", type=int, default=0)
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--sample-only",
    action="store_true",
    help="Only instantiate the generator and sample one edge bundle.",
)
args = parser.parse_args()

print("RNG Seed:", args.seed)
print("Checkpoint:", args.checkpoint)
print("Device:", args.device)

start = np.array((1.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
goal = np.array((36.0, 33.0), dtype=np.float64)
goal_radius = args.goal_radius
env = make_test_environment()
agent = get_second_order_car_agent(1)

planner = make_flow_eb_rrt_planner_soc(
    start,
    goal,
    goal_radius,
    agent,
    env,
    checkpoint_path=args.checkpoint,
    device=args.device,
    sample_steps=args.sample_steps,
    flow_prefetch_batch_size=args.flow_prefetch_batch_size,
)
apply_common_planner_args(planner, args)
planner.set_profile_enabled(args.profile)

for run_id in range(args.warmup_runs):
    warmup_env = make_test_environment()
    warmup_agent = get_second_order_car_agent(1)
    warmup_planner = make_flow_eb_rrt_planner_soc(
        start,
        goal,
        goal_radius,
        warmup_agent,
        warmup_env,
        checkpoint_path=args.checkpoint,
        device=args.device,
        sample_steps=args.sample_steps,
        flow_prefetch_batch_size=args.flow_prefetch_batch_size,
    )
    apply_common_planner_args(warmup_planner, args)
    warmup_planner.set_profile_enabled(False)
    warmup_planner.plan_path()
    print(f"Warmup run {run_id + 1}/{args.warmup_runs} done")

if args.sample_only:
    edge_bundle = planner.flow_edge_generator.sample(planner.start, num_edges=8)
    print("Sampled Flow edge bundle")
    print("  num_edges:", edge_bundle.num_edges)
    print("  actions shape:", edge_bundle.actions.shape)
    print("  timesteps:", edge_bundle.timesteps)
    print("  first actions:")
    print(edge_bundle.actions[:3])
    print("  first final states:")
    print(edge_bundle.final_states[:3])
    if args.profile:
        planner.print_profile()
else:
    start_time = time.perf_counter()
    planner.plan_path()
    wall_time = time.perf_counter() - start_time
    print("Measured wall time:", wall_time)
    print_planner_result(planner)
    if args.profile:
        planner.print_profile()
