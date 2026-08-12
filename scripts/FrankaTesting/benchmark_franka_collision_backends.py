"""Validate and benchmark MorphIt/cuRobo against the PyBullet mesh checker."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
FLOWMRMP_ROOT = ROOT_DIR / "mrmp_with_kite_extend"
SRC_DIR = FLOWMRMP_ROOT / "src"
MAIN_SCRIPTS = ROOT_DIR / "scripts"
for module_path in (MAIN_SCRIPTS, SRC_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import (  # noqa: E402
    FrankaCuroboCollisionChecker,
    JOINT_NAMES,
    Q_LOWER,
    Q_UPPER,
)

DEFAULT_ROBOT_DIR = ROOT_DIR / "assets" / "robots" / "panda"
DEFAULT_URDF = DEFAULT_ROBOT_DIR / "panda.urdf"
DEFAULT_SRDF = DEFAULT_ROBOT_DIR / "panda.srdf"
DEFAULT_PROBLEMS = ROOT_DIR / "data" / "benchmarks" / "franka_reachable_ab_100.npz"
DEFAULT_CONFIGS = [
    DEFAULT_ROBOT_DIR / "curobo" / "panda_morphit.yml",
    DEFAULT_ROBOT_DIR / "curobo" / "panda_morphit_density2.yml",
    DEFAULT_ROBOT_DIR / "curobo" / "panda_morphit_density4.yml",
]
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "results"
    / "franka_collision_backends"
    / "morphit_curobo_vs_pybullet_20260805"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uniform-samples", type=int, default=50_000)
    parser.add_argument("--kinematic-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PyBulletReferenceChecker:
    """Exact-mesh reference matching the original Franka runtime checker."""

    def __init__(self, urdf_path: Path, srdf_path: Path):
        import pybullet as pb
        from pybullet_utils.bullet_client import BulletClient

        self.pb = pb
        self.client = BulletClient(connection_mode=pb.DIRECT)
        flags = (
            pb.URDF_USE_SELF_COLLISION
            | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            | pb.URDF_IGNORE_VISUAL_SHAPES
        )
        self.body_id = self.client.loadURDF(
            str(urdf_path.resolve()), useFixedBase=True, flags=flags
        )
        joint_info = [
            self.client.getJointInfo(self.body_id, index)
            for index in range(self.client.getNumJoints(self.body_id))
        ]
        decode = lambda value: value.decode() if isinstance(value, bytes) else value
        self.joint_indices = {decode(info[1]): int(info[0]) for info in joint_info}
        self.link_indices = {decode(info[12]): int(info[0]) for info in joint_info}
        root_link = ET.parse(urdf_path).getroot().find("link")
        if root_link is None:
            raise RuntimeError("URDF has no root link")
        self.link_indices[root_link.attrib["name"]] = -1
        self.arm_joint_indices = [self.joint_indices[name] for name in JOINT_NAMES]
        for disabled in ET.parse(srdf_path).getroot().findall("disable_collisions"):
            first = disabled.attrib["link1"]
            second = disabled.attrib["link2"]
            if first in self.link_indices and second in self.link_indices:
                self.client.setCollisionFilterPair(
                    self.body_id,
                    self.body_id,
                    self.link_indices[first],
                    self.link_indices[second],
                    0,
                )

    def set_configuration(self, q: np.ndarray) -> None:
        values = [[float(value)] for value in np.asarray(q, dtype=np.float64)]
        self.client.resetJointStatesMultiDof(
            self.body_id,
            self.arm_joint_indices,
            values,
            [[0.0]] * 7,
        )

    def in_collision(self, q: np.ndarray) -> bool:
        self.set_configuration(q)
        self.client.performCollisionDetection()
        contacts = self.client.getContactPoints(self.body_id, self.body_id)
        return any(point[8] < 0.0 for point in contacts)

    def collision_mask(self, q: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (self.in_collision(configuration) for configuration in q),
            dtype=bool,
            count=len(q),
        )

    def link_poses(
        self, q: np.ndarray, link_names: list[str]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.set_configuration(q)
        poses = {}
        for name in link_names:
            index = self.link_indices[name]
            if index == -1:
                position, quaternion_xyzw = self.client.getBasePositionAndOrientation(
                    self.body_id
                )
            else:
                state = self.client.getLinkState(
                    self.body_id, index, computeForwardKinematics=True
                )
                position, quaternion_xyzw = state[4], state[5]
            poses[name] = (
                np.asarray(position, dtype=np.float64),
                np.asarray(quaternion_xyzw, dtype=np.float64),
            )
        return poses

    def close(self) -> None:
        self.client.disconnect()


def load_relevant_states(problem_path: Path) -> np.ndarray:
    with np.load(problem_path) as data:
        parts = [data["starts"][:, :7], data["goals"][:, :7]]
        if "witness_states" in data:
            parts.append(data["witness_states"][:, :7])
    return np.unique(np.concatenate(parts).astype(np.float32), axis=0)


def curobo_collision_mask(
    checker: FrankaCuroboCollisionChecker,
    q: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    parts = [
        ~checker.collision_free_mask(q[start : start + batch_size])
        for start in range(0, len(q), batch_size)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=bool)


def agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    true_positive = int(np.sum(reference & candidate))
    true_negative = int(np.sum(~reference & ~candidate))
    false_positive = int(np.sum(~reference & candidate))
    false_negative = int(np.sum(reference & ~candidate))
    total = int(len(reference))
    return {
        "samples": total,
        "pybullet_collisions": int(reference.sum()),
        "curobo_collisions": int(candidate.sum()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "agreement_fraction": (true_positive + true_negative) / total,
        "false_positive_fraction": false_positive / total,
        "false_negative_fraction": false_negative / total,
        "collision_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
    }


def benchmark_pybullet(
    checker: PyBulletReferenceChecker, q: np.ndarray
) -> dict[str, float]:
    sample = q[: min(len(q), 10_000)]
    for configuration in sample[:100]:
        checker.in_collision(configuration)
    start = time.perf_counter()
    checker.collision_mask(sample)
    elapsed = time.perf_counter() - start
    return {
        "samples": len(sample),
        "seconds": elapsed,
        "microseconds_per_configuration": elapsed * 1e6 / len(sample),
        "configurations_per_second": len(sample) / elapsed,
    }


def benchmark_curobo(
    checker: FrankaCuroboCollisionChecker, q: np.ndarray
) -> dict[str, dict[str, float]]:
    results = {}
    for batch_size in (1, 16, 64, 256, 1024, 4096):
        sample = q[:batch_size]
        checker.collision_free_mask(sample)
        checker.synchronize()
        repeats = max(5, min(1000, 20_000 // batch_size))
        start = time.perf_counter()
        for _ in range(repeats):
            checker.collision_free_mask(sample)
        checker.synchronize()
        elapsed = time.perf_counter() - start
        count = repeats * batch_size
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "repeats": repeats,
            "seconds": elapsed,
            "microseconds_per_configuration": elapsed * 1e6 / count,
            "configurations_per_second": count / elapsed,
            "batch_latency_microseconds": elapsed * 1e6 / repeats,
        }
    return results


def compare_kinematics(
    config_path: Path,
    reference: PyBulletReferenceChecker,
    q: np.ndarray,
) -> dict[str, object]:
    import torch
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kinematics_data = config_data.get("robot_cfg", config_data)["kinematics"]
    link_names = list(kinematics_data["collision_link_names"])
    for name in ("panda_link8", "panda_grasptarget"):
        if name not in link_names:
            link_names.append(name)
    non_base_links = [name for name in link_names if name != "panda_link0"]
    kinematics_data = dict(kinematics_data)
    kinematics_data["tool_frames"] = non_base_links
    model = Kinematics(KinematicsCfg.from_data_dict(kinematics_data))
    q_tensor = torch.as_tensor(q, device="cuda", dtype=torch.float32)
    pose = model.get_link_poses(q_tensor, non_base_links)
    positions = pose.position.detach().cpu().numpy()
    quaternions_wxyz = pose.quaternion.detach().cpu().numpy()

    translation_errors = []
    rotation_errors = []
    worst_translation = {"error_m": -1.0}
    worst_rotation = {"error_rad": -1.0}
    for sample_index, configuration in enumerate(q):
        pybullet_poses = reference.link_poses(configuration, link_names)
        for link_index, link_name in enumerate(non_base_links):
            pb_position, pb_xyzw = pybullet_poses[link_name]
            cu_position = positions[sample_index, link_index]
            cu_wxyz = quaternions_wxyz[sample_index, link_index]
            cu_xyzw = np.roll(cu_wxyz, -1)
            translation_error = float(np.linalg.norm(pb_position - cu_position))
            rotation_error = float(
                (
                    Rotation.from_quat(pb_xyzw).inv() * Rotation.from_quat(cu_xyzw)
                ).magnitude()
            )
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
            if translation_error > worst_translation["error_m"]:
                worst_translation = {
                    "error_m": translation_error,
                    "sample_index": sample_index,
                    "link": link_name,
                }
            if rotation_error > worst_rotation["error_rad"]:
                worst_rotation = {
                    "error_rad": rotation_error,
                    "sample_index": sample_index,
                    "link": link_name,
                }

    return {
        "samples": len(q),
        "links": link_names,
        "comparisons": len(translation_errors),
        "translation_error_m": {
            "mean": float(np.mean(translation_errors)),
            "p95": float(np.percentile(translation_errors, 95)),
            "max": float(np.max(translation_errors)),
        },
        "rotation_error_rad": {
            "mean": float(np.mean(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
            "max": float(np.max(rotation_errors)),
        },
        "worst_translation": worst_translation,
        "worst_rotation": worst_rotation,
    }


def obstacle_api_check(
    config_path: Path, q_candidates: np.ndarray
) -> dict[str, object]:
    import torch

    empty_checker = FrankaCuroboCollisionChecker(config_path)
    valid_mask = empty_checker.collision_free_mask(q_candidates)
    if not np.any(valid_mask):
        raise RuntimeError(
            "No self-collision-free configuration available for obstacle test"
        )
    q = q_candidates[int(np.flatnonzero(valid_mask)[0])]
    q_tensor = torch.as_tensor(q[None, :], device="cuda", dtype=torch.float32)
    pose = empty_checker.checker.kinematics.get_link_poses(
        q_tensor, ["panda_grasptarget"]
    )
    tool_position = pose.position[0, 0].detach().cpu().tolist()
    near_scene = {
        "cuboid": {
            "blocking_box": {
                "dims": [0.08, 0.08, 0.08],
                "pose": [*tool_position, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    far_scene = {
        "cuboid": {
            "far_box": {
                "dims": [0.08, 0.08, 0.08],
                "pose": [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    near_checker = FrankaCuroboCollisionChecker(config_path, scene_model=near_scene)
    far_checker = FrankaCuroboCollisionChecker(config_path, scene_model=far_scene)
    return {
        "configuration": q.tolist(),
        "tool_position_m": tool_position,
        "empty_world_valid": bool(empty_checker.collision_free_mask(q)[0]),
        "far_cuboid_valid": bool(far_checker.collision_free_mask(q)[0]),
        "tool_centered_cuboid_valid": bool(near_checker.collision_free_mask(q)[0]),
        "expected": {
            "empty_world_valid": True,
            "far_cuboid_valid": True,
            "tool_centered_cuboid_valid": False,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    uniform_q = rng.uniform(Q_LOWER, Q_UPPER, size=(args.uniform_samples, 7)).astype(
        np.float32
    )
    relevant_q = load_relevant_states(args.problems)

    reference = PyBulletReferenceChecker(args.urdf, args.srdf)
    try:
        start = time.perf_counter()
        uniform_reference = reference.collision_mask(uniform_q)
        uniform_reference_seconds = time.perf_counter() - start
        relevant_reference = reference.collision_mask(relevant_q)
        pybullet_timing = benchmark_pybullet(reference, uniform_q)

        config_results = {}
        checker_by_path = {}
        for config_path in args.configs:
            checker = FrankaCuroboCollisionChecker(config_path)
            checker_by_path[config_path] = checker
            uniform_candidate = curobo_collision_mask(checker, uniform_q)
            relevant_candidate = curobo_collision_mask(checker, relevant_q)
            config_results[config_path.name] = {
                "path": str(config_path.resolve()),
                "sha256": sha256(config_path),
                "uniform_agreement": agreement(uniform_reference, uniform_candidate),
                "benchmark_relevant_agreement": agreement(
                    relevant_reference, relevant_candidate
                ),
                "timing_by_batch_size": benchmark_curobo(checker, uniform_q),
                "uniform_false_positive_examples": uniform_q[
                    ~uniform_reference & uniform_candidate
                ][:10].tolist(),
                "uniform_false_negative_examples": uniform_q[
                    uniform_reference & ~uniform_candidate
                ][:10].tolist(),
            }

        # Prefer zero false negatives, then the fewest false positives, then
        # the faster 256-waypoint throughput.
        def selection_key(item):
            result = item[1]
            stats = result["uniform_agreement"]
            timing = result["timing_by_batch_size"]["256"]
            return (
                stats["false_negative"],
                stats["false_positive"],
                timing["microseconds_per_configuration"],
            )

        selected_name, selected_result = min(config_results.items(), key=selection_key)
        selected_path = Path(selected_result["path"])
        kinematic_q = relevant_q[: args.kinematic_samples]
        kinematics = compare_kinematics(selected_path, reference, kinematic_q)
        obstacle = obstacle_api_check(selected_path, relevant_q)

        report = {
            "seed": args.seed,
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": sha256(args.urdf),
            "srdf": str(args.srdf.resolve()),
            "srdf_sha256": sha256(args.srdf),
            "uniform_samples": len(uniform_q),
            "benchmark_relevant_samples": len(relevant_q),
            "pybullet_uniform_label_seconds": uniform_reference_seconds,
            "pybullet_timing": pybullet_timing,
            "configs": config_results,
            "selection_policy": (
                "fewest uniform false negatives, then fewest false positives, "
                "then fastest 256-waypoint throughput"
            ),
            "selected_config": selected_name,
            "kinematics": kinematics,
            "obstacle_api_check": obstacle,
        }
        report_path = args.output_dir / "collision_backend_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        with contextlib.suppress(Exception):
            reference.close()


if __name__ == "__main__":
    main()
