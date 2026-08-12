#!/usr/bin/env python3
"""Audit saved four-Franka paths using cuRobo and GPU spheres only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for module_path in (REPOSITORY_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from franka_prioritized_adapter import (  # noqa: E402
    FrankaPrioritizedCollisionAdapter,
    ROBOT_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def audit_result(result_dir: Path) -> dict[str, object]:
    summary_path = result_dir / "summary.json"
    trajectory_path = result_dir / "trajectories.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as data:
        paths = [data[f"robot_{index}_states"].copy() for index in range(4)]
        dt = float(data["integration_dt"])
    state_bank = np.concatenate(paths, axis=0)
    adapter = FrankaPrioritizedCollisionAdapter(state_bank=state_bank)
    started = time.perf_counter()
    try:
        static_records = []
        world_paths = []
        for robot_index, path in enumerate(paths):
            checker = adapter.checkers[robot_index]
            valid_parts = []
            for start in range(0, len(path), 256):
                valid_parts.append(
                    checker.collision_free_mask(path[start : start + 256, :7])
                )
            valid = np.concatenate(valid_parts)
            invalid = np.flatnonzero(~valid)
            static_records.append(
                {
                    "robot_index": robot_index,
                    "robot_name": ROBOT_NAMES[robot_index],
                    "samples": int(len(path)),
                    "invalid_self_or_static_samples": int(len(invalid)),
                    "first_invalid_sample": int(invalid[0]) if invalid.size else None,
                }
            )
            world_paths.append(
                adapter.configuration_world_spheres(robot_index, path[:, :7])
            )

        max_steps = max(len(path) for path in paths)
        pair_records = []
        total_pair_tests = 0
        for first in range(4):
            for second in range(first + 1, 4):
                first_indices = adapter.torch.arange(
                    max_steps, device=world_paths[first].device
                ).clamp(max=len(paths[first]) - 1)
                second_indices = adapter.torch.arange(
                    max_steps, device=world_paths[second].device
                ).clamp(max=len(paths[second]) - 1)
                first_spheres = world_paths[first].index_select(0, first_indices)
                second_spheres = world_paths[second].index_select(0, second_indices)
                collision = adapter._sphere_overlap_mask(
                    first_spheres, second_spheres, pair_counter="audit"
                )
                collision_indices = (
                    adapter.torch.nonzero(collision, as_tuple=False)
                    .view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                tests = int(
                    max_steps * first_spheres.shape[1] * second_spheres.shape[1]
                )
                total_pair_tests += tests
                pair_records.append(
                    {
                        "robots": [ROBOT_NAMES[first], ROBOT_NAMES[second]],
                        "colliding_samples": int(len(collision_indices)),
                        "first_collision_sample": (
                            int(collision_indices[0]) if len(collision_indices) else None
                        ),
                        "first_collision_time_seconds": (
                            float(collision_indices[0] * dt)
                            if len(collision_indices)
                            else None
                        ),
                        "sphere_pair_tests": tests,
                    }
                )
        adapter.checkers[0].synchronize()
        elapsed = time.perf_counter() - started
        collision_free = bool(
            all(record["invalid_self_or_static_samples"] == 0 for record in static_records)
            and all(record["colliding_samples"] == 0 for record in pair_records)
        )
        audit = {
            "collision_free": collision_free,
            "backend": "cuRobo API plus batched PyTorch CUDA sphere-pair tests",
            "pybullet_used": False,
            "integration_dt": dt,
            "synchronized_samples": int(max_steps),
            "total_sphere_pair_tests": total_pair_tests,
            "audit_time_seconds": elapsed,
            "robots": static_records,
            "pairs": pair_records,
        }
        (result_dir / "curobo_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary["post_run_curobo_audit"] = audit
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "problem_id": summary["problem_id"],
            "algorithm": summary["algorithm"],
            "solved": summary["solved"],
            "saved_paths_collision_free": collision_free,
            "audit": str((result_dir / "curobo_audit.json").resolve()),
        }
    finally:
        adapter.close()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_paths = sorted(run_dir.glob("problem_*/*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No result summaries below {run_dir}")
    records = []
    for summary_path in summary_paths:
        print(f"auditing {summary_path.parent}", flush=True)
        records.append(audit_result(summary_path.parent))
    (run_dir / "curobo_audits.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    run_summary_path = run_dir / "run_summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.is_file()
        else {"created_by": "audit_four_franka_prioritized.py"}
    )
    refreshed = []
    for summary_path in summary_paths:
        refreshed.append(json.loads(summary_path.read_text(encoding="utf-8")))
    run_summary["results"] = refreshed
    run_summary["curobo_audits"] = records
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved audits: {run_dir / 'curobo_audits.json'}")


if __name__ == "__main__":
    main()
