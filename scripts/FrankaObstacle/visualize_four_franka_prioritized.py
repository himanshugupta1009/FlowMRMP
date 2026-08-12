#!/usr/bin/env python3
"""Render prioritized four-Franka trajectory files to PyBullet MP4 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OBSTACLE_SCRIPTS = Path(__file__).resolve().parent
DEFAULT_PROBLEMS = (
    REPOSITORY_ROOT
    / "results"
    / "franka_obstacle"
    / "four_franka_difficult_problems"
    / "four_franka_difficult_problems.npz"
)
if str(OBSTACLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(OBSTACLE_SCRIPTS))

from four_franka_table_scene import (  # noqa: E402
    DEFAULT_URDF,
    ROBOT_LAYOUT,
    add_axis_triad,
    add_table_and_mounts,
    create_box,
    load_robots,
)


ROBOT_COLORS = (
    (0.20, 0.52, 0.95, 1.0),
    (0.95, 0.55, 0.14, 1.0),
    (0.20, 0.72, 0.38, 1.0),
    (0.66, 0.36, 0.88, 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-motion-frames", type=int, default=300)
    parser.add_argument("--hold-frames", type=int, default=30)
    return parser.parse_args()


def ui_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def tint_robots(client, robot_ids: list[int]) -> None:
    for robot_id, color in zip(robot_ids, ROBOT_COLORS):
        client.changeVisualShape(robot_id, -1, rgbaColor=color)
        for link_index in range(client.getNumJoints(robot_id)):
            client.changeVisualShape(robot_id, link_index, rgbaColor=color)


def end_effector_link(client, robot_id: int) -> int:
    """Return the panda_hand link index from the loaded visualization URDF."""
    for joint_index in range(client.getNumJoints(robot_id)):
        link_name = client.getJointInfo(robot_id, joint_index)[12].decode("utf-8")
        if link_name == "panda_hand":
            return joint_index
    raise RuntimeError("Loaded Franka URDF has no panda_hand link")


def add_goal_spheres(
    client,
    robot_ids: list[int],
    movable: list[list[int]],
    goals: np.ndarray,
) -> list[int]:
    """Add one translucent visual marker at each goal end-effector position."""
    marker_ids = []
    for robot_index, (robot_id, joints, goal, color) in enumerate(
        zip(robot_ids, movable, goals, ROBOT_COLORS)
    ):
        for joint, position in zip(joints, goal[:7]):
            client.resetJointState(robot_id, joint, float(position))
        link_state = client.getLinkState(
            robot_id,
            end_effector_link(client, robot_id),
            computeForwardKinematics=True,
        )
        goal_position = link_state[4]
        visual_id = client.createVisualShape(
            client.GEOM_SPHERE,
            radius=0.085,
            rgbaColor=[color[0], color[1], color[2], 0.38],
        )
        marker_ids.append(
            int(
                client.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_id,
                    basePosition=goal_position,
                )
            )
        )
    return marker_ids


def render_frame(client, *, width: int, height: int, summary: dict, frame_text: str):
    view = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.72],
        distance=3.25,
        yaw=42.0,
        pitch=-31.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = client.computeProjectionMatrixFOV(
        fov=52.0,
        aspect=width / height,
        nearVal=0.05,
        farVal=7.0,
    )
    image_width, image_height, rgba, _, _ = client.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=client.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[-2.0, -3.0, 5.0],
    )
    pixels = np.asarray(rgba, dtype=np.uint8).reshape(image_height, image_width, 4)
    frame = Image.fromarray(pixels[..., :3]).convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((18, 16, width - 18, 118), radius=14, fill=(22, 28, 36, 220))
    status = "SOLVED" if summary["solved"] else "FAILED / BEST PARTIAL PATHS"
    status_color = (117, 235, 153, 255) if summary["solved"] else (255, 174, 91, 255)
    draw.text(
        (36, 27),
        f"Problem {summary['problem_id']}: {summary['scenario_name']} | {summary['algorithm']}",
        font=ui_font(24, bold=True),
        fill=(255, 255, 255, 255),
    )
    draw.text((36, 68), status, font=ui_font(20, bold=True), fill=status_color)
    abbreviations = {
        "northwest": "NW",
        "northeast": "NE",
        "southwest": "SW",
        "southeast": "SE",
    }
    order = ">".join(abbreviations[name] for name in summary["priority_names"])
    draw.text(
        (330, 72),
        f"order {order} | {frame_text}",
        font=ui_font(15),
        fill=(224, 231, 240, 255),
    )
    draw.text(
        (330, 94),
        "translucent markers = end-effector goals",
        font=ui_font(13),
        fill=(188, 202, 216, 255),
    )
    return np.asarray(Image.alpha_composite(frame, overlay).convert("RGB"))


def load_result(result_dir: Path):
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "trajectories.npz", allow_pickle=False) as data:
        paths = [data[f"robot_{index}_states"].copy() for index in range(4)]
        dt = float(data["integration_dt"])
    return summary, paths, dt


def render_result(
    result_dir: Path,
    video_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    max_motion_frames: int,
    hold_frames: int,
    goals: np.ndarray,
) -> dict[str, object]:
    import pybullet as pb
    from pybullet_utils.bullet_client import BulletClient

    summary, paths, dt = load_result(result_dir)
    client = BulletClient(connection_mode=pb.DIRECT)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.configureDebugVisualizer(client.COV_ENABLE_GUI, 0)
        client.resetSimulation()
        # A single floor body keeps mesh-video rendering fast. The denser grid
        # is retained in the still-scene generator but adds dozens of bodies
        # that do not help trajectory playback.
        create_box(
            client,
            (3.4, 3.0, 0.025),
            (0.0, 0.0, -0.0175),
            [0.87, 0.89, 0.91, 1.0],
        )
        add_table_and_mounts(client)
        robot_ids = load_robots(client, DEFAULT_URDF)
        add_axis_triad(client)
        tint_robots(client, robot_ids)
        movable = [
            [
                joint
                for joint in range(client.getNumJoints(robot_id))
                if client.getJointInfo(robot_id, joint)[2] == client.JOINT_REVOLUTE
            ][:7]
            for robot_id in robot_ids
        ]
        add_goal_spheres(client, robot_ids, movable, goals)

        max_steps = max(len(path) for path in paths)
        motion_count = min(max_motion_frames, max_steps)
        indices = np.unique(
            np.rint(np.linspace(0, max_steps - 1, motion_count)).astype(np.int64)
        )

        def set_frame(global_index: int) -> None:
            for robot_index, (robot_id, joints, path) in enumerate(
                zip(robot_ids, movable, paths)
            ):
                state = path[min(global_index, len(path) - 1)]
                for joint, position in zip(joints, state[:7]):
                    client.resetJointState(robot_id, joint, float(position))

        with imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="warning",
        ) as writer:
            set_frame(0)
            first = render_frame(
                client,
                width=width,
                height=height,
                summary=summary,
                frame_text="start",
            )
            for _ in range(hold_frames):
                writer.append_data(first)
            for global_index in indices:
                set_frame(int(global_index))
                elapsed = global_index * dt
                writer.append_data(
                    render_frame(
                        client,
                        width=width,
                        height=height,
                        summary=summary,
                        frame_text=f"trajectory t={elapsed:.2f}s",
                    )
                )
            last = render_frame(
                client,
                width=width,
                height=height,
                summary=summary,
                frame_text="final saved state",
            )
            for _ in range(hold_frames):
                writer.append_data(last)
        return {
            "problem_id": summary["problem_id"],
            "algorithm": summary["algorithm"],
            "solved": summary["solved"],
            "video": str(video_path.resolve()),
            "frames": int(2 * hold_frames + len(indices)),
            "fps": fps,
            "source": str(result_dir.resolve()),
            "pybullet_role": "visualization only",
            "goal_markers": "four translucent end-effector spheres",
        }
    finally:
        client.disconnect()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "videos")).resolve()
    problems_path = args.problems.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not problems_path.is_file():
        raise FileNotFoundError(problems_path)
    with np.load(problems_path, allow_pickle=False) as problem_data:
        problem_goals = {
            int(problem_id): problem_data["goals"][index].copy()
            for index, problem_id in enumerate(problem_data["problem_ids"])
        }
    summaries = sorted(run_dir.glob("problem_*/*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No prioritized result summaries below {run_dir}")
    records = []
    for summary_path in summaries:
        result_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        algorithm = str(summary["algorithm"]).lower().replace("rrt", "")
        video_path = output_dir / (
            f"problem_{int(summary['problem_id']):02d}_{summary['scenario_name']}_{algorithm}.mp4"
        )
        print(f"rendering {video_path.name}", flush=True)
        records.append(
            render_result(
                result_dir,
                video_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
                max_motion_frames=args.max_motion_frames,
                hold_frames=args.hold_frames,
                goals=problem_goals[int(summary["problem_id"])],
            )
        )
    (output_dir / "videos.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved videos: {output_dir}")


if __name__ == "__main__":
    main()
