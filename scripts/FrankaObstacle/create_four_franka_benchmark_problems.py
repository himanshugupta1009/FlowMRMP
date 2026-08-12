#!/usr/bin/env python3
"""Generate fresh easy/medium/hard four-Franka shared-table problems.

Endpoints come from dataset200k, use zero terminal velocity, and are validated
with the project 295-sphere MorphIt/cuRobo model.  Difficulty is assigned from
the exact static/self/inter-arm collision structure of a 41-sample synchronized
joint interpolation.  This generator never invokes PyBullet.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
OBSTACLE_SCRIPTS = Path(__file__).resolve().parent
FRANKA_TESTING = MAIN_SCRIPTS / "FrankaTesting"
for module_path in (MAIN_SCRIPTS, OBSTACLE_SCRIPTS, FRANKA_TESTING):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import FrankaPanda, Q_LOWER, Q_UPPER  # noqa: E402
from benchmark_franka_vanilla_rrt import sample_dataset_state_bank  # noqa: E402
from four_franka_table_scene import ROBOT_LAYOUT, TABLE_DIMS, TABLE_POSE  # noqa: E402
from franka_prioritized_adapter import FrankaPrioritizedCollisionAdapter  # noqa: E402

DEFAULT_DATASET = Path("C:/Users/sodan/Downloads/dataset200k.h5")
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" /
    "four_franka_multi_planner_problems_final_v3_20260812"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-bank-size", type=int, default=80_000)
    parser.add_argument("--candidate-pool", type=int, default=4_000)
    parser.add_argument("--endpoint-sets", type=int, default=120)
    parser.add_argument("--pair-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026081207)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_endpoint_sets(adapter, candidates, count, rng):
    per_robot = []
    for robot_index in range(4):
        static, _ = adapter.validate_path(
            robot_index, candidates[:, :7], start_time=0.0, step_time=0.02
        )
        valid = np.flatnonzero(static)
        if len(valid) < 100:
            raise RuntimeError(
                f"Only {len(valid)} valid candidate states for robot {robot_index}")
        per_robot.append(valid)

    endpoint_sets = []
    attempts = 0
    while len(endpoint_sets) < count and attempts < count * 500:
        attempts += 1
        states = np.stack(
            [candidates[int(rng.choice(per_robot[index]))] for index in range(4)]
        )
        states[:, 7:] = 0.0
        valid, _ = adapter.configuration_set_valid(states)
        if valid:
            endpoint_sets.append(states)
    if len(endpoint_sets) < count:
        raise RuntimeError(
            f"Found only {len(endpoint_sets)} valid endpoint sets after {attempts} tries")
    return endpoint_sets, attempts, [len(values) for values in per_robot]


def score_pair(adapter, start, goal, samples=41):
    alpha = np.linspace(0.0, 1.0, samples)[:, None, None]
    synchronized = (1.0 - alpha) * start[None, :, :] + alpha * goal[None, :, :]
    per_robot_invalid = np.zeros((samples, 4), dtype=bool)
    world = []
    for index in range(4):
        static, _ = adapter.validate_path(
            index, synchronized[:, index, :7], start_time=0.0, step_time=0.02
        )
        limits = np.asarray([
            adapter.agents[index].is_state_within_limits(state)
            for state in synchronized[:, index]
        ])
        per_robot_invalid[:, index] = ~(static & limits)
        world.append(adapter.configuration_world_spheres(
            index, synchronized[:, index, :7]))
    pair_masks = {}
    for first in range(4):
        for second in range(first + 1, 4):
            mask = adapter._sphere_overlap_mask(
                world[first], world[second], pair_counter="scenario_generation"
            ).detach().cpu().numpy().astype(bool)
            pair_masks[f"{first}-{second}"] = mask
    interarm = np.any(np.stack(list(pair_masks.values())), axis=0)
    invalid = np.any(per_robot_invalid, axis=1) | interarm
    interior = slice(1, samples - 1)
    pair_names = [name for name, mask in pair_masks.items()
                  if bool(np.any(mask[interior]))]
    normalized_delta = np.stack([
        FrankaPanda.normalize_configuration(goal[index])
        - FrankaPanda.normalize_configuration(start[index])
        for index in range(4)
    ])
    return {
        "interpolation_samples": samples,
        "invalid_interior_samples": int(np.count_nonzero(invalid[interior])),
        "interarm_collision_samples": int(np.count_nonzero(interarm[interior])),
        "static_or_self_invalid_samples": int(np.count_nonzero(
            np.any(per_robot_invalid[interior], axis=1))),
        "colliding_pairs": pair_names,
        "colliding_pair_count": len(pair_names),
        "mean_normalized_q_distance": float(np.mean(
            np.linalg.norm(normalized_delta, axis=1))),
        "minimum_normalized_q_distance": float(np.min(
            np.linalg.norm(normalized_delta, axis=1))),
        "maximum_normalized_q_distance": float(np.max(
            np.linalg.norm(normalized_delta, axis=1))),
    }


def category(metric):
    conflicts = metric["invalid_interior_samples"]
    pairs = metric["colliding_pair_count"]
    motion = metric["mean_normalized_q_distance"]
    if conflicts == 0 and 0.44 <= motion <= 0.50 \
            and metric["minimum_normalized_q_distance"] >= 0.42 \
            and metric["maximum_normalized_q_distance"] <= 0.52:
        return "easy"
    if 1 <= conflicts <= 12 and 1 <= pairs <= 2 and motion >= 1.5:
        return "medium"
    if conflicts >= 13 and pairs >= 2 and motion >= 1.7:
        return "hard"
    return None


def find_local_easy(adapter, endpoints, rng, trials=2_000):
    """Find a short, nontrivial, collision-free pair for pilot calibration."""
    joint_half_ranges = 0.5 * (Q_UPPER - Q_LOWER)
    for _ in range(trials):
        start_index = int(rng.integers(len(endpoints)))
        start = endpoints[start_index]
        goal = start.copy()
        directions = rng.normal(0.0, 1.0, size=(4, 7))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        normalized_delta = directions * rng.uniform(0.44, 0.50, size=(4, 1))
        goal[:, :7] = np.clip(
            start[:, :7] + normalized_delta * joint_half_ranges,
            Q_LOWER + 1e-4,
            Q_UPPER - 1e-4,
        )
        goal[:, 7:] = 0.0
        valid, _ = adapter.configuration_set_valid(goal)
        if not valid:
            continue
        metric = score_pair(adapter, start, goal)
        if category(metric) == "easy":
            return start, goal, metric, (start_index, -1)
    raise RuntimeError("Could not find a locally perturbed easy scenario")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    state_bank, _, _ = sample_dataset_state_bank(
        dataset, args.state_bank_size, rng)
    candidate_indices = rng.choice(
        len(state_bank), size=min(args.candidate_pool, len(state_bank)), replace=False)
    candidates = np.asarray(state_bank[candidate_indices], dtype=np.float64).copy()
    candidates[:, 7:] = 0.0

    adapter = FrankaPrioritizedCollisionAdapter(
        state_bank=state_bank, device=args.device, acceleration_scale=0.5)
    try:
        endpoints, attempts, per_robot_counts = exact_endpoint_sets(
            adapter, candidates, args.endpoint_sets, rng)
        selected = {"easy": find_local_easy(adapter, endpoints, rng)}
        easy_metric = selected["easy"][2]
        print(
            "selected easy: local perturbation invalid="
            f"{easy_metric['invalid_interior_samples']} pairs="
            f"{easy_metric['colliding_pair_count']} motion="
            f"{easy_metric['mean_normalized_q_distance']:.3f}", flush=True)
        scored = []
        used_pairs = set()
        for _ in range(args.pair_trials):
            first, second = rng.choice(len(endpoints), size=2, replace=False)
            pair = (int(first), int(second))
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            metric = score_pair(adapter, endpoints[first], endpoints[second])
            label = category(metric)
            scored.append({"pair": pair, "category": label, **metric})
            if label in {"medium", "hard"} and label not in selected:
                selected[label] = (endpoints[first], endpoints[second], metric, pair)
                print(
                    f"selected {label}: pair={pair} invalid="
                    f"{metric['invalid_interior_samples']} pairs="
                    f"{metric['colliding_pair_count']} motion="
                    f"{metric['mean_normalized_q_distance']:.3f}", flush=True)
            if len(selected) == 3:
                break
        if len(selected) != 3:
            by_label = {label: sum(item["category"] == label for item in scored)
                        for label in ("easy", "medium", "hard")}
            raise RuntimeError(f"Could not identify all tiers; counts={by_label}")

        labels = ("easy", "medium", "hard")
        starts = np.stack([selected[label][0] for label in labels])
        goals = np.stack([selected[label][1] for label in labels])
        problem_file = output_dir / "four_franka_easy_medium_hard.npz"
        np.savez_compressed(
            problem_file,
            starts=starts,
            goals=goals,
            problem_ids=np.arange(3, dtype=np.int64),
            names=np.asarray(labels),
            robot_names=np.asarray([robot["name"] for robot in ROBOT_LAYOUT]),
            robot_base_positions=np.asarray([robot["position"] for robot in ROBOT_LAYOUT]),
            robot_base_yaw_degrees=np.asarray([robot["yaw_deg"] for robot in ROBOT_LAYOUT]),
            table_dimensions=np.asarray(TABLE_DIMS),
            table_position=np.asarray(TABLE_POSE),
        )
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "format": "four_franka_easy_medium_hard_exact_spheres_v1",
            "generator_seed": args.seed,
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "problem_file": str(problem_file),
            "problem_file_sha256": sha256(problem_file),
            "sphere_count_per_robot": 295,
            "pybullet_used": False,
            "endpoint_candidate_attempts": attempts,
            "valid_candidate_counts_by_robot": per_robot_counts,
            "difficulty_rule": {
                "easy": "0/39 invalid samples; every arm is 0.42..0.52 normalized q units from a goal radius of 0.40",
                "medium": "1..12/39 invalid samples; <=2 colliding arm pairs; mean motion >=1.5",
                "hard": ">=13/39 invalid samples; >=2 colliding arm pairs; mean motion >=1.7",
            },
            "problems": [
                {"problem_id": index, "label": label,
                 "endpoint_pair_indices": list(selected[label][3]),
                 **selected[label][2]}
                for index, label in enumerate(labels)
            ],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(problem_file)
        print(manifest_path)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
