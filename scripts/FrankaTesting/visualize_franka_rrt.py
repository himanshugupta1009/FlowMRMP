#!/usr/bin/env python3
"""Visualize a saved Franka VanillaRRT or FlowEBRRT path in PyBullet.

Select the planner with ``--planner``. Select a saved path with either
``--problem-id`` or ``--path-file``. If neither is supplied, the command uses
the first successful result, or the best partial path when no run succeeded.
Use ``--headless`` to render files without opening a GUI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from franka_paths import (
    CORE_RRT_SRC as FLOWMRMP_SRC,
    DEFAULT_FLOW_RESULTS_ROOT,
    DEFAULT_URDF,
    DEFAULT_VANILLA_RESULTS_ROOT,
)


DEFAULT_RUN_NAME = "franka_reachable_100_goal_r025_max50"
PLANNER_RESULTS_ROOT = {
    "flow": DEFAULT_FLOW_RESULTS_ROOT,
    "vanilla": DEFAULT_VANILLA_RESULTS_ROOT,
}
PLANNER_LABEL = {"flow": "FlowEBRRT", "vanilla": "VanillaRRT"}

if str(FLOWMRMP_SRC) not in sys.path:
    sys.path.insert(0, str(FLOWMRMP_SRC))

from Agents.FrankaPanda import FrankaSelfCollisionChecker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=tuple(PLANNER_RESULTS_ROOT),
        required=True,
        help="Planner result family to visualize.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the selected planner's default saved-run directory.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Run directory below the selected planner's results root.",
    )
    path_selection = parser.add_mutually_exclusive_group()
    path_selection.add_argument("--problem-id", type=int, default=None)
    path_selection.add_argument(
        "--path-file",
        type=Path,
        default=None,
        help="Visualize an explicit saved problem_XXX.npz file.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--frame-dt", type=float, default=0.04)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-image", type=Path, default=None)
    parser.add_argument("--save-traces", type=Path, default=None)
    return parser.parse_args()


def select_problem(results_dir: Path, requested_id: int | None) -> int:
    if requested_id is not None:
        return requested_id
    with (results_dir / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    successes = [row for row in rows if row["success"].lower() == "true"]
    if successes:
        return int(successes[0]["problem_id"])
    if not rows:
        raise ValueError(f"No rows found in {results_dir / 'trials.csv'}")
    return int(min(rows, key=lambda row: float(row["final_normalized_distance"]))["problem_id"])


def find_link_index(checker: FrankaSelfCollisionChecker, name: str) -> int:
    if name not in checker.link_indices:
        raise KeyError(f"Link {name!r} not found in Panda URDF")
    return checker.link_indices[name]


def end_effector_positions(
    checker: FrankaSelfCollisionChecker, states: np.ndarray, link_index: int
) -> list[tuple[float, float, float]]:
    points = []
    for state in states:
        checker.set_configuration(state[:7])
        link_state = checker.client.getLinkState(checker.body_id, link_index)
        points.append(tuple(link_state[0]))
    return points


def draw_path(
    checker: FrankaSelfCollisionChecker, points: list[tuple[float, float, float]]
) -> None:
    # Cap the number of visible bodies for long paths while keeping endpoints.
    if len(points) > 251:
        keep = np.linspace(0, len(points) - 1, 251).astype(int)
        points = [points[index] for index in keep]
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        fraction = index / max(1, len(points) - 2)
        color = [fraction, 0.25, 1.0 - fraction, 0.9]
        checker.client.addUserDebugLine(
            start,
            end,
            lineColorRGB=color[:3],
            lineWidth=4.0,
        )
        # Debug lines are not rendered by every DIRECT-mode renderer, so add
        # thin visual-only cylinders as a portable EE trail.
        start_array = np.asarray(start)
        end_array = np.asarray(end)
        delta = end_array - start_array
        length = float(np.linalg.norm(delta))
        if length > 1e-8:
            direction = delta / length
            z_axis = np.array([0.0, 0.0, 1.0])
            cross = np.cross(z_axis, direction)
            dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            if np.linalg.norm(cross) < 1e-10:
                quaternion = [0.0, 0.0, 0.0, 1.0] if dot > 0 else [1.0, 0.0, 0.0, 0.0]
            else:
                axis = cross / np.linalg.norm(cross)
                half_angle = 0.5 * np.arccos(dot)
                quaternion = [
                    *(axis * np.sin(half_angle)).tolist(),
                    float(np.cos(half_angle)),
                ]
            visual = checker.client.createVisualShape(
                checker.client.GEOM_CYLINDER,
                radius=0.004,
                length=length,
                rgbaColor=color,
            )
            checker.client.createMultiBody(
                baseVisualShapeIndex=visual,
                baseCollisionShapeIndex=-1,
                basePosition=((start_array + end_array) * 0.5).tolist(),
                baseOrientation=quaternion,
            )
    for point, color in ((points[0], [0.1, 0.9, 0.1, 1.0]), (points[-1], [0.9, 0.1, 0.1, 1.0])):
        visual = checker.client.createVisualShape(
            checker.client.GEOM_SPHERE, radius=0.025, rgbaColor=color
        )
        checker.client.createMultiBody(
            baseVisualShapeIndex=visual,
            baseCollisionShapeIndex=-1,
            basePosition=point,
        )


def save_camera_image(checker: FrankaSelfCollisionChecker, path: Path) -> None:
    width, height = 1280, 900
    view = checker.client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.35, 0.0, 0.45],
        distance=1.45,
        yaw=45.0,
        pitch=-25.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = checker.client.computeProjectionMatrixFOV(
        fov=55.0, aspect=width / height, nearVal=0.05, farVal=4.0
    )
    _, _, rgba, _, _ = checker.client.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=checker.client.ER_TINY_RENDERER,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)[..., :3]).save(path)


def save_state_traces(states: np.ndarray, dt: float, path: Path, title: str) -> None:
    times = np.arange(states.shape[0]) * dt
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for joint in range(7):
        axes[0].plot(times, states[:, joint], label=f"q{joint + 1}")
        axes[1].plot(times, states[:, 7 + joint], label=f"dq{joint + 1}")
    axes[0].set(ylabel="Position (rad)", title=title)
    axes[1].set(xlabel="Path time (s)", ylabel="Velocity (rad/s)")
    axes[0].legend(ncol=7, fontsize=8, loc="upper center")
    axes[1].legend(ncol=7, fontsize=8, loc="upper center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else PLANNER_RESULTS_ROOT[args.planner] / args.run_name
    ).resolve()
    if args.path_file is not None:
        path_file = args.path_file.resolve()
        match = re.fullmatch(r"problem_(\d+)", path_file.stem)
        path_name = f"problem_{int(match.group(1)):03d}" if match else path_file.stem
        output_dir = (
            path_file.parent.parent
            if path_file.parent.name == "paths"
            else path_file.parent
        )
    else:
        problem_id = select_problem(results_dir, args.problem_id)
        path_name = f"problem_{problem_id:03d}"
        path_file = results_dir / "paths" / f"{path_name}.npz"
        output_dir = results_dir
    if not path_file.is_file():
        raise FileNotFoundError(path_file)
    with np.load(path_file) as data:
        states = data["states"]
        success = bool(data["success"])
        dt = float(data["integration_dt"])
        saved_planner = str(data["planner"]) if "planner" in data else None
    planner = PLANNER_LABEL[args.planner]
    if saved_planner is not None and saved_planner.lower() != planner.lower():
        raise ValueError(
            f"Selected planner {planner!r}, but {path_file} stores {saved_planner!r}"
        )
    if states.ndim != 2 or states.shape[1] != 14 or states.shape[0] == 0:
        raise ValueError(f"Malformed saved path: {states.shape}")

    trace_path = args.save_traces or (output_dir / f"{path_name}_state_traces.png")
    save_state_traces(
        states,
        dt,
        trace_path,
        f"Franka {planner} {path_name} "
        f"({'success' if success else 'best partial'})",
    )

    checker = FrankaSelfCollisionChecker(
        args.urdf, visualize=not args.headless, load_visuals=True
    )
    ee_link = find_link_index(checker, "panda_hand")
    points = end_effector_positions(checker, states, ee_link)
    draw_path(checker, points)
    checker.set_configuration(states[-1, :7])

    if args.headless or args.save_image is not None:
        image_path = args.save_image or (output_dir / f"{path_name}_pybullet.png")
        save_camera_image(checker, image_path)
        print(f"saved PyBullet rendering: {image_path}")
    if not args.headless:
        checker.client.resetDebugVisualizerCamera(
            cameraDistance=1.45,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[0.35, 0.0, 0.45],
        )
        for _ in range(max(1, args.loops)):
            for state in states:
                checker.set_configuration(state[:7])
                time.sleep(args.frame_dt)
    checker.close()
    print(
        f"visualized {planner} {path_name}: "
        f"{'successful path' if success else 'best partial path'} ({states.shape[0]} waypoints)"
    )
    print(f"saved state traces: {trace_path}")


if __name__ == "__main__":
    main()
