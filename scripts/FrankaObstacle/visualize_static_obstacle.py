#!/usr/bin/env python3
"""Visualize one static-cuboid benchmark path in PyBullet.

Example:
    python scripts/FrankaObstacle/visualize_static_obstacle.py \
        --problem-id 7 --planner flow
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
FRANKA_TESTING = REPOSITORY_ROOT / "scripts" / "FrankaTesting"
CORE_RRT_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
for module_path in (MAIN_SCRIPTS, FRANKA_TESTING, CORE_RRT_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import FrankaPyBulletCollisionChecker  # noqa: E402
from franka_paths import DEFAULT_URDF  # noqa: E402
from static_obstacle_test import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    OBSTACLE_DIMS,
    OBSTACLE_NAME,
    OBSTACLE_POSE,
)
from visualize_franka_rrt import (  # noqa: E402
    draw_path,
    end_effector_positions,
    find_link_index,
    save_camera_image,
    save_state_traces,
)


def parse_args() -> argparse.Namespace:
    """Parse planner/problem selection and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--planner", choices=("vanilla", "flow"), required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--frame-dt", type=float, default=0.04)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-image", type=Path, default=None)
    parser.add_argument("--save-traces", type=Path, default=None)
    return parser.parse_args()


def add_static_cuboid(checker: FrankaPyBulletCollisionChecker) -> int:
    """Add the exact cuRobo cuboid dimensions and pose to PyBullet."""
    half_extents = (0.5 * np.asarray(OBSTACLE_DIMS, dtype=np.float64)).tolist()
    position = list(OBSTACLE_POSE[:3])
    qw, qx, qy, qz = OBSTACLE_POSE[3:]
    orientation_xyzw = [qx, qy, qz, qw]
    collision = checker.client.createCollisionShape(
        checker.client.GEOM_BOX,
        halfExtents=half_extents,
    )
    visual = checker.client.createVisualShape(
        checker.client.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.95, 0.38, 0.05, 0.72],
    )
    body = checker.client.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=orientation_xyzw,
    )
    checker.client.addUserDebugText(
        OBSTACLE_NAME,
        [position[0], position[1], position[2] + half_extents[2] + 0.05],
        textColorRGB=[0.95, 0.35, 0.05],
        textSize=1.25,
    )
    return int(body)


def main() -> None:
    """Load the requested result, draw its trail and cuboid, and animate it."""
    args = parse_args()
    if args.problem_id < 0 or args.problem_id >= 25:
        raise ValueError("problem-id must be between 0 and 24")
    if args.frame_dt <= 0.0 or args.loops <= 0:
        raise ValueError("frame-dt and loops must be positive")
    results_root = args.results_root.resolve()
    results_dir = results_root / args.planner
    path_file = results_dir / "paths" / f"problem_{args.problem_id:03d}.npz"
    if not path_file.is_file():
        raise FileNotFoundError(path_file)
    with np.load(path_file) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        success = bool(data["success"])
        integration_dt = float(data["integration_dt"])
        saved_planner = str(data["planner"]) if "planner" in data else None
    if states.ndim != 2 or states.shape[1] != 14 or not len(states):
        raise ValueError(f"Malformed saved path: {states.shape}")
    expected_planner = "VanillaRRT" if args.planner == "vanilla" else "FlowEBRRT"
    if saved_planner is not None and saved_planner.lower() != expected_planner.lower():
        raise ValueError(
            f"Requested {expected_planner}, but the path stores {saved_planner}"
        )

    checker = FrankaPyBulletCollisionChecker(
        args.urdf,
        visualize=not args.headless,
        load_visuals=True,
        suppress_output=False,
    )
    try:
        add_static_cuboid(checker)
        end_effector_link = find_link_index(checker, "panda_hand")
        points = end_effector_positions(checker, states, end_effector_link)
        draw_path(checker, points)
        checker.set_configuration(states[-1, :7])

        stem = f"{args.planner}_problem_{args.problem_id:03d}"
        traces_path = args.save_traces or (results_root / f"{stem}_state_traces.png")
        save_state_traces(
            states,
            integration_dt,
            traces_path,
            f"Static cuboid: {expected_planner} problem {args.problem_id:03d} "
            f"({'success' if success else 'best partial'})",
        )
        if args.headless or args.save_image is not None:
            image_path = args.save_image or (results_root / f"{stem}_pybullet.png")
            save_camera_image(checker, image_path)
            print(f"saved PyBullet rendering: {image_path}")
        if not args.headless:
            checker.client.resetDebugVisualizerCamera(
                cameraDistance=1.35,
                cameraYaw=43.0,
                cameraPitch=-24.0,
                cameraTargetPosition=[0.38, 0.0, 0.48],
            )
            for _ in range(args.loops):
                for state in states:
                    checker.set_configuration(state[:7])
                    time.sleep(args.frame_dt)
    finally:
        checker.close()
    print(
        f"visualized {expected_planner} problem {args.problem_id:03d}: "
        f"{'successful path' if success else 'best partial path'} "
        f"({len(states)} waypoints)"
    )
    print(f"saved state traces: {traces_path}")


if __name__ == "__main__":
    main()
