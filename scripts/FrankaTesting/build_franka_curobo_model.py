"""Build a cuRobo Franka model from this repository's exact Panda assets.

MorphIt is used only to fit link-local spheres.  The saved runtime model keeps
the URDF kinematics and uses the same SRDF collision exclusions as PyBullet.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_DIR = ROOT_DIR / "assets" / "robots" / "panda"
DEFAULT_URDF = DEFAULT_ROBOT_DIR / "panda.urdf"
DEFAULT_SRDF = DEFAULT_ROBOT_DIR / "panda.srdf"
DEFAULT_OUTPUT = DEFAULT_ROBOT_DIR / "curobo" / "panda_morphit.yml"
DEFAULT_REPORT = DEFAULT_ROBOT_DIR / "curobo" / "panda_morphit_build.json"
EXPECTED_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    parser.add_argument("--asset-path", type=Path, default=DEFAULT_ROBOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sphere-density", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--coverage-weight", type=float, default=None)
    parser.add_argument("--protrusion-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srdf_collision_ignores(srdf_path: Path) -> dict[str, list[str]]:
    """Return one canonical entry for every SRDF-disabled collision pair."""
    root = ET.parse(srdf_path).getroot()
    pairs = {
        tuple(sorted((item.attrib["link1"], item.attrib["link2"])))
        for item in root.findall("disable_collisions")
    }
    ignores: dict[str, list[str]] = {}
    for first, second in sorted(pairs):
        ignores.setdefault(first, []).append(second)
    return ignores


def movable_joint_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]


def main() -> None:
    args = parse_args()
    for path in (args.urdf, args.srdf):
        if not path.is_file():
            raise FileNotFoundError(path)

    joint_names = movable_joint_names(args.urdf)
    if joint_names != EXPECTED_ARM_JOINTS:
        raise RuntimeError(
            f"Expected movable joints {EXPECTED_ARM_JOINTS}, found {joint_names}"
        )

    import torch

    from curobo._src.geom.sphere_fit.types import SphereFitType
    from curobo.robot_builder import RobotBuilder

    if not torch.cuda.is_available():
        raise RuntimeError("MorphIt requires a CUDA-capable PyTorch environment")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    builder = RobotBuilder(
        urdf_path=str(args.urdf.resolve()),
        asset_path=str(args.asset_path.resolve()),
        tool_frames=["panda_grasptarget"],
    )
    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        fit_type=SphereFitType.MORPHIT,
        use_collision_mesh=True,
        iterations=args.iterations,
        coverage_weight=args.coverage_weight,
        protrusion_weight=args.protrusion_weight,
        compute_metrics=True,
    )

    # Match FrankaSelfCollisionChecker exactly: PyBullet excludes parent-child
    # pairs at URDF load time, and the adjacent pairs also appear in this SRDF.
    # Do not add cuRobo's sampled/default-pose pruning because that would change
    # which link pairs the two backends consider.
    ignores = srdf_collision_ignores(args.srdf)
    builder._self_collision_ignore = ignores
    builder._self_collision_buffer = {
        link_name: 0.0 for link_name in builder.collision_link_names
    }

    config = builder.build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    builder.save(config, str(args.output.resolve()))

    report = {
        "generator": "cuRobo RobotBuilder + SphereFitType.MORPHIT",
        "curobo_version": __import__("curobo").__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "seed": args.seed,
        "sphere_density": args.sphere_density,
        "iterations": args.iterations,
        "coverage_weight": args.coverage_weight,
        "protrusion_weight": args.protrusion_weight,
        "used_collision_mesh": True,
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": sha256(args.urdf),
        "srdf": str(args.srdf.resolve()),
        "srdf_sha256": sha256(args.srdf),
        "movable_joint_names": joint_names,
        "tool_frames": ["panda_grasptarget"],
        "collision_link_names": builder.collision_link_names,
        "sphere_count": builder.num_spheres,
        "sphere_count_by_link": {
            name: len(spheres)
            for name, spheres in (builder.collision_spheres or {}).items()
        },
        "self_collision_ignore": ignores,
        "self_collision_ignore_pair_count": sum(map(len, ignores.values())),
        "self_collision_buffer_m": 0.0,
        "metrics_by_link": {
            name: asdict(metrics) for name, metrics in builder.link_metrics.items()
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
