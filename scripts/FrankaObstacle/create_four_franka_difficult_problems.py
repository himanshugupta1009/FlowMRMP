#!/usr/bin/env python3
"""Create and render three difficult four-Franka shared-table problems.

Each start and goal is valid under the active 295-sphere cuRobo self-collision
model and the PyBullet table/inter-robot model. Difficulty is demonstrated by
collisions along the synchronized straight-line joint interpolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPTS = REPOSITORY_ROOT / "scripts"
CORE_RRT_SOURCE = REPOSITORY_ROOT / "mrmp_with_kite_extend" / "src"
for module_path in (MAIN_SCRIPTS, Path(__file__).resolve().parent, CORE_RRT_SOURCE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from FrankaPanda import FrankaSelfCollisionChecker  # noqa: E402
from four_franka_table_scene import (  # noqa: E402
    DEFAULT_URDF,
    ROBOT_LAYOUT,
    TABLE_DIMS,
    TABLE_POSE,
    add_axis_triad,
    add_ground_and_grid,
    add_table_and_mounts,
    font,
    load_robots,
)

DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" / "four_franka_difficult_problems"
)
INTERPOLATION_SAMPLES = 41
ROBOT_COLORS = (
    (0.20, 0.52, 0.95, 1.0),
    (0.95, 0.55, 0.14, 1.0),
    (0.20, 0.72, 0.38, 1.0),
    (0.66, 0.36, 0.88, 1.0),
)
ROBOT_SHORT_NAMES = ("NW", "NE", "SW", "SE")


def q(values: list[list[float]]) -> np.ndarray:
    """Make a checked four-by-seven joint-position array."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4, 7):
        raise ValueError(f"Expected a (4, 7) configuration, got {array.shape}")
    return array


PROBLEMS = (
    {
        "problem_id": 0,
        "name": "dense_opposing_weave",
        "title": "Dense opposing weave",
        "description": (
            "All four arms make large simultaneous moves through overlapping "
            "workspace; the direct synchronized motion has a long conflict interval."
        ),
        "start": q(
            [
                [
                    -0.487139,
                    0.621705,
                    0.115652,
                    -2.112688,
                    0.952239,
                    2.880927,
                    1.381358,
                ],
                [
                    -0.042833,
                    0.847317,
                    0.488506,
                    -1.728743,
                    -0.153139,
                    2.538216,
                    0.453475,
                ],
                [
                    0.480981,
                    0.587680,
                    -0.497636,
                    -1.458266,
                    -0.657805,
                    2.878009,
                    0.914266,
                ],
                [
                    -0.051851,
                    0.109523,
                    0.771775,
                    -1.692796,
                    -0.802096,
                    1.322451,
                    1.798848,
                ],
            ]
        ),
        "goal": q(
            [
                [
                    0.825866,
                    0.624038,
                    -0.900050,
                    -0.827035,
                    0.441621,
                    0.897493,
                    1.477024,
                ],
                [
                    -0.379701,
                    0.684740,
                    0.360239,
                    -2.302450,
                    -1.475672,
                    2.750755,
                    0.107762,
                ],
                [0.024753, 0.467833, 0.003123, -1.278079, 0.715826, 2.635722, 0.017008],
                [
                    -1.529691,
                    0.014212,
                    0.993983,
                    -1.849452,
                    -0.256981,
                    1.463508,
                    -0.362729,
                ],
            ]
        ),
    },
    {
        "problem_id": 1,
        "name": "center_lane_exchange",
        "title": "Center-lane exchange",
        "description": (
            "The northwest and southwest workspaces compress toward the center "
            "while the east-side arms reposition around the same narrow corridor."
        ),
        "start": q(
            [
                [
                    -0.437294,
                    0.378308,
                    -0.104204,
                    -2.179448,
                    1.320952,
                    0.524216,
                    -0.062572,
                ],
                [
                    -0.288113,
                    0.420470,
                    0.587729,
                    -1.527136,
                    -0.074434,
                    2.201607,
                    1.735053,
                ],
                [
                    0.298773,
                    0.103825,
                    -0.029921,
                    -2.184207,
                    -0.242564,
                    2.170405,
                    1.051965,
                ],
                [
                    -0.761146,
                    0.186522,
                    0.257313,
                    -1.593902,
                    0.971868,
                    1.824132,
                    1.943998,
                ],
            ]
        ),
        "goal": q(
            [
                [
                    0.582234,
                    0.269235,
                    -0.039344,
                    -1.355161,
                    -0.609843,
                    1.366155,
                    -0.434302,
                ],
                [
                    -0.552380,
                    -0.224503,
                    0.731521,
                    -2.039709,
                    -0.146737,
                    1.662329,
                    2.219933,
                ],
                [
                    0.097768,
                    0.895104,
                    -0.628246,
                    -1.693311,
                    -1.052420,
                    1.847251,
                    0.934387,
                ],
                [
                    0.234466,
                    1.042059,
                    -0.143653,
                    -1.259100,
                    0.640991,
                    2.220411,
                    1.191438,
                ],
            ]
        ),
    },
    {
        "problem_id": 2,
        "name": "diagonal_crossing",
        "title": "Diagonal four-arm crossing",
        "description": (
            "Large diagonal reaches force the four swept volumes through a shared "
            "central region even though both endpoint arrangements are safe."
        ),
        "start": q(
            [
                [
                    -0.647681,
                    0.216185,
                    0.600011,
                    -1.995211,
                    0.483243,
                    0.992246,
                    0.907733,
                ],
                [
                    0.801227,
                    0.319479,
                    -0.617231,
                    -1.818234,
                    1.127605,
                    1.314035,
                    -1.341353,
                ],
                [
                    0.098273,
                    0.678929,
                    -0.889204,
                    -1.161756,
                    0.394890,
                    2.429553,
                    0.002091,
                ],
                [
                    0.621796,
                    -0.351860,
                    -0.246793,
                    -2.217655,
                    0.678763,
                    1.607480,
                    1.912561,
                ],
            ]
        ),
        "goal": q(
            [
                [
                    0.913016,
                    1.030129,
                    -0.024943,
                    -0.724565,
                    -1.963288,
                    2.180926,
                    1.530097,
                ],
                [
                    -1.028628,
                    0.737920,
                    0.962501,
                    -2.066689,
                    -0.750680,
                    1.932332,
                    2.113383,
                ],
                [0.024753, 0.467833, 0.003123, -1.278079, 0.715826, 2.635722, 0.017008],
                [
                    -0.810187,
                    0.862803,
                    0.904850,
                    -1.596956,
                    1.913231,
                    1.853430,
                    0.841546,
                ],
            ]
        ),
    },
)


def parse_args() -> argparse.Namespace:
    """Parse paths and rendering dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-width", type=int, default=850)
    parser.add_argument("--panel-height", type=int, default=620)
    return parser.parse_args()


def joint_indices(client, robot_id: int) -> list[int]:
    """Return the seven Panda revolute-joint indices."""
    return [
        index
        for index in range(client.getNumJoints(robot_id))
        if client.getJointInfo(robot_id, index)[2] == client.JOINT_REVOLUTE
    ][:7]


def hand_link_index(client, robot_id: int) -> int:
    """Resolve panda_hand in the loaded project URDF."""
    links = {
        client.getJointInfo(robot_id, index)[12].decode(): index
        for index in range(client.getNumJoints(robot_id))
    }
    return int(links["panda_hand"])


def set_configuration(client, robot_ids, robot_joints, configuration) -> None:
    """Apply a four-arm joint configuration and refresh contacts."""
    for body_id, joints, positions in zip(robot_ids, robot_joints, configuration):
        for joint_index, position in zip(joints, positions):
            client.resetJointState(body_id, joint_index, float(position))
    client.performCollisionDetection()


def tint_robots(client, robot_ids) -> None:
    """Give each arm a persistent identity color."""
    for robot_id, color in zip(robot_ids, ROBOT_COLORS):
        for link_index in range(-1, client.getNumJoints(robot_id)):
            client.changeVisualShape(robot_id, link_index, rgbaColor=color)


def world_contacts(client, robot_ids, table_id) -> dict[str, object]:
    """Collect table and inter-robot contacts for the current state."""
    table_contacts = []
    pair_contacts = []
    contact_positions = []
    for robot_index, robot_id in enumerate(robot_ids):
        contacts = client.getContactPoints(bodyA=robot_id, bodyB=table_id)
        if contacts:
            table_contacts.append(ROBOT_SHORT_NAMES[robot_index])
            contact_positions.extend(
                np.mean([contact[5], contact[6]], axis=0).tolist()
                for contact in contacts
            )
    for first in range(len(robot_ids)):
        for second in range(first + 1, len(robot_ids)):
            contacts = client.getContactPoints(
                bodyA=robot_ids[first], bodyB=robot_ids[second]
            )
            if contacts:
                pair_contacts.append(
                    f"{ROBOT_SHORT_NAMES[first]}-{ROBOT_SHORT_NAMES[second]}"
                )
                contact_positions.extend(
                    np.mean([contact[5], contact[6]], axis=0).tolist()
                    for contact in contacts
                )
    return {
        "table": sorted(set(table_contacts)),
        "pairs": sorted(set(pair_contacts)),
        "positions": contact_positions,
    }


def end_effector_positions(client, robot_ids, hand_links) -> list[list[float]]:
    """Return the four hand origins in world coordinates."""
    return [
        [float(value) for value in client.getLinkState(body_id, link_index)[0]]
        for body_id, link_index in zip(robot_ids, hand_links)
    ]


def add_markers(client, positions, colors, radius: float) -> list[int]:
    """Add visual-only spheres and return their body IDs."""
    bodies = []
    for position, color in zip(positions, colors):
        visual = client.createVisualShape(
            client.GEOM_SPHERE, radius=radius, rgbaColor=color
        )
        bodies.append(
            int(
                client.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual,
                    basePosition=position,
                )
            )
        )
    return bodies


def capture(client, width: int, height: int) -> Image.Image:
    """Capture the common metric-grid camera without annotations."""
    view = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.76],
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
    return Image.fromarray(pixels[..., :3])


def label_panel(image: Image.Image, label: str, color) -> Image.Image:
    """Place a compact label within one rendered panel."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    box = (24, 22, 360, 80)
    draw.rounded_rectangle(box, radius=12, fill=(20, 27, 35), outline=color, width=3)
    draw.text((44, 33), label, font=font(25, bold=True), fill=color)
    return result


def render_problem_image(
    client,
    robot_ids,
    robot_joints,
    hand_links,
    problem,
    audit,
    output_path: Path,
    panel_width: int,
    panel_height: int,
) -> None:
    """Render valid start, colliding direct-path witness, and valid goal."""
    witness_index = int(audit["collision_witness_sample"])
    alpha = witness_index / (INTERPOLATION_SAMPLES - 1)
    witness = (1.0 - alpha) * problem["start"] + alpha * problem["goal"]
    panels = []
    panel_specs = (
        (problem["start"], "START - VALID", (109, 232, 150)),
        (witness, f"DIRECT PATH t={alpha:.2f} - COLLISION", (255, 92, 92)),
        (problem["goal"], "GOAL - VALID", (109, 232, 150)),
    )
    for panel_index, (configuration, label, label_color) in enumerate(panel_specs):
        set_configuration(client, robot_ids, robot_joints, configuration)
        ee_markers = add_markers(
            client,
            end_effector_positions(client, robot_ids, hand_links),
            ROBOT_COLORS,
            0.027,
        )
        collision_markers = []
        if panel_index == 1:
            contacts = world_contacts(client, robot_ids, audit["table_id"])
            collision_markers = add_markers(
                client,
                contacts["positions"],
                [(1.0, 0.02, 0.02, 1.0)] * len(contacts["positions"]),
                0.045,
            )
        panel = label_panel(
            capture(client, panel_width, panel_height), label, label_color
        )
        panels.append(panel)
        for body_id in ee_markers + collision_markers:
            client.removeBody(body_id)

    header_height = 225
    canvas = Image.new(
        "RGB", (panel_width * 3, panel_height + header_height), (27, 34, 43)
    )
    for panel_index, panel in enumerate(panels):
        canvas.paste(panel, (panel_index * panel_width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (42, 26),
        f"Problem {problem['problem_id'] + 1}: {problem['title']}",
        font=font(36, bold=True),
        fill=(255, 255, 255),
    )
    draw.text(
        (42, 82),
        problem["description"],
        font=font(24),
        fill=(215, 225, 237),
    )
    difficulty = (
        f"Naive synchronized interpolation: {audit['invalid_interior_samples']}/39 "
        f"invalid samples; {audit['inter_robot_collision_samples']} inter-robot, "
        f"{audit['table_collision_samples']} table, "
        f"{audit['self_collision_samples']} self-collision."
    )
    draw.text((42, 128), difficulty, font=font(24), fill=(255, 174, 119))
    legend_x = 42
    for short_name, color in zip(ROBOT_SHORT_NAMES, ROBOT_COLORS):
        rgb = tuple(int(255 * component) for component in color[:3])
        draw.ellipse((legend_x, 178, legend_x + 24, 202), fill=rgb)
        draw.text((legend_x + 34, 174), short_name, font=font(22), fill=(235, 240, 246))
        legend_x += 105
    draw.text(
        (520, 174),
        "Red spheres mark direct-path contact points   |   Grid: 0.25 m   |   XYZ: red/green/blue",
        font=font(22),
        fill=(205, 214, 226),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def audit_problem(client, robot_ids, robot_joints, table_id, self_checker, problem):
    """Validate endpoints and characterize the direct synchronized motion."""
    path = np.linspace(problem["start"], problem["goal"], INTERPOLATION_SAMPLES)
    self_free = self_checker.collision_free_mask(path.reshape(-1, 7)).reshape(
        INTERPOLATION_SAMPLES, 4
    )
    table_samples = []
    pair_samples = []
    pair_names = set()
    world_collision_by_sample = {}
    for sample_index, configuration in enumerate(path):
        set_configuration(client, robot_ids, robot_joints, configuration)
        contacts = world_contacts(client, robot_ids, table_id)
        if contacts["table"]:
            table_samples.append(sample_index)
        if contacts["pairs"]:
            pair_samples.append(sample_index)
            pair_names.update(contacts["pairs"])
        if contacts["table"] or contacts["pairs"]:
            world_collision_by_sample[sample_index] = contacts
    self_samples = np.flatnonzero(~np.all(self_free, axis=1)).tolist()
    invalid = sorted(set(table_samples) | set(pair_samples) | set(self_samples))
    if 0 in invalid or INTERPOLATION_SAMPLES - 1 in invalid:
        raise RuntimeError(f"Problem {problem['problem_id']} has an invalid endpoint")
    interior_invalid = [
        index for index in invalid if 0 < index < INTERPOLATION_SAMPLES - 1
    ]
    if not pair_samples:
        raise RuntimeError(
            f"Problem {problem['problem_id']} lacks an inter-robot conflict"
        )
    witness_index = pair_samples[len(pair_samples) // 2]
    start_ee = None
    goal_ee = None
    for label, configuration in (
        ("start", problem["start"]),
        ("goal", problem["goal"]),
    ):
        set_configuration(client, robot_ids, robot_joints, configuration)
        positions = end_effector_positions(
            client,
            robot_ids,
            [hand_link_index(client, body_id) for body_id in robot_ids],
        )
        if label == "start":
            start_ee = positions
        else:
            goal_ee = positions
    return {
        "endpoint_validity": {"start": True, "goal": True},
        "interpolation_samples": INTERPOLATION_SAMPLES,
        "interior_samples": INTERPOLATION_SAMPLES - 2,
        "invalid_interior_samples": len(interior_invalid),
        "invalid_sample_indices": interior_invalid,
        "inter_robot_collision_samples": len(
            [index for index in pair_samples if 0 < index < INTERPOLATION_SAMPLES - 1]
        ),
        "inter_robot_pairs": sorted(pair_names),
        "table_collision_samples": len(
            [index for index in table_samples if 0 < index < INTERPOLATION_SAMPLES - 1]
        ),
        "self_collision_samples": len(
            [index for index in self_samples if 0 < index < INTERPOLATION_SAMPLES - 1]
        ),
        "collision_witness_sample": witness_index,
        "collision_witness_alpha": witness_index / (INTERPOLATION_SAMPLES - 1),
        "collision_witness_contacts": world_collision_by_sample[witness_index],
        "start_end_effector_positions": start_ee,
        "goal_end_effector_positions": goal_ee,
        "minimum_robot_joint_motion_l2": float(
            np.min(np.linalg.norm(problem["goal"] - problem["start"], axis=1))
        ),
    }


def main() -> None:
    """Build the scene, validate the problems, and save images and data."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("panel dimensions must be positive")

    import pybullet as pb
    from pybullet_utils.bullet_client import BulletClient

    client = BulletClient(connection_mode=pb.DIRECT)
    self_checker = FrankaSelfCollisionChecker()
    try:
        client.resetSimulation()
        add_ground_and_grid(client)
        table_id, _ = add_table_and_mounts(client)
        robot_ids = load_robots(client, urdf_path)
        robot_joints = [joint_indices(client, body_id) for body_id in robot_ids]
        hand_links = [hand_link_index(client, body_id) for body_id in robot_ids]
        tint_robots(client, robot_ids)
        add_axis_triad(client)

        audits = []
        for problem in PROBLEMS:
            audit = audit_problem(
                client,
                robot_ids,
                robot_joints,
                table_id,
                self_checker,
                problem,
            )
            audit["table_id"] = table_id
            image_path = (
                output_dir
                / f"problem_{problem['problem_id'] + 1:02d}_{problem['name']}.png"
            )
            render_problem_image(
                client,
                robot_ids,
                robot_joints,
                hand_links,
                problem,
                audit,
                image_path,
                args.panel_width,
                args.panel_height,
            )
            del audit["table_id"]
            audits.append(audit)
            print(f"saved problem image: {image_path}")

        starts_q = np.stack([problem["start"] for problem in PROBLEMS])
        goals_q = np.stack([problem["goal"] for problem in PROBLEMS])
        starts = np.concatenate((starts_q, np.zeros_like(starts_q)), axis=2)
        goals = np.concatenate((goals_q, np.zeros_like(goals_q)), axis=2)
        problem_file = output_dir / "four_franka_difficult_problems.npz"
        np.savez_compressed(
            problem_file,
            starts=starts,
            goals=goals,
            problem_ids=np.asarray([problem["problem_id"] for problem in PROBLEMS]),
            names=np.asarray([problem["name"] for problem in PROBLEMS]),
            robot_names=np.asarray([robot["name"] for robot in ROBOT_LAYOUT]),
            robot_base_positions=np.asarray(
                [robot["position"] for robot in ROBOT_LAYOUT]
            ),
            robot_base_yaw_degrees=np.asarray(
                [robot["yaw_deg"] for robot in ROBOT_LAYOUT]
            ),
            table_dimensions=np.asarray(TABLE_DIMS),
            table_position=np.asarray(TABLE_POSE),
            interpolation_samples=np.asarray(INTERPOLATION_SAMPLES),
        )
        manifest = {
            "format": "four_franka_shared_table_problem_set",
            "units": "metres_radians_seconds",
            "num_problems": len(PROBLEMS),
            "num_robots": 4,
            "state_dimension_per_robot": 14,
            "robot_order": list(ROBOT_SHORT_NAMES),
            "robot_layout": list(ROBOT_LAYOUT),
            "table": {"dimensions": TABLE_DIMS, "position": TABLE_POSE},
            "urdf": str(urdf_path),
            "self_collision_model": {
                "backend": "MorphIt/cuRobo",
                "sphere_count": int(self_checker.checker.kinematics.total_spheres),
            },
            "difficulty_definition": (
                "Start and goal are valid, but synchronized linear interpolation "
                "in four independent 7D joint spaces has collision samples."
            ),
            "problems": [
                {
                    "problem_id": problem["problem_id"],
                    "name": problem["name"],
                    "title": problem["title"],
                    "description": problem["description"],
                    "start_joint_positions": problem["start"].tolist(),
                    "goal_joint_positions": problem["goal"].tolist(),
                    "audit": audit,
                }
                for problem, audit in zip(PROBLEMS, audits)
            ],
        }
        manifest_path = output_dir / "four_franka_difficult_problems.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"saved problem data: {problem_file}")
        print(f"saved problem manifest: {manifest_path}")
    finally:
        self_checker.close()
        client.disconnect()


if __name__ == "__main__":
    main()
