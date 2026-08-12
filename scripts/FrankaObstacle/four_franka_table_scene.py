#!/usr/bin/env python3
"""Build and render a four-Franka shared-table environment in PyBullet.

The scene places two Panda arms along each long side of a central cuboid table.
All dimensions and poses are expressed in metres in the constants below and
are also written to a JSON manifest beside the rendered image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPOSITORY_ROOT / "assets" / "robots" / "panda" / "panda.urdf"
DEFAULT_CUROBO_CONFIG = (
    REPOSITORY_ROOT
    / "assets"
    / "robots"
    / "panda"
    / "curobo"
    / "panda_morphit_d10_c1000_p1000.yml"
)
EXPECTED_COLLISION_SPHERES = 295
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "franka_obstacle" / "four_franka_table_scene"
)

TABLE_DIMS = (1.80, 0.90, 0.68)
TABLE_POSE = (0.0, 0.0, TABLE_DIMS[2] * 0.5)
PEDESTAL_DIMS = (0.34, 0.30, 0.70)
MOUNT_DIMS = (0.28, 0.25, 0.06)
ROBOT_BASE_Z = PEDESTAL_DIMS[2] + MOUNT_DIMS[2]
ROBOT_X = 0.52
ROBOT_Y = 0.74
GRID_SPACING = 0.25
READY_CONFIGURATION = (0.0, -0.65, 0.0, -2.15, 0.0, 1.55, 0.785)

ROBOT_LAYOUT = (
    {
        "name": "northwest",
        "position": (-ROBOT_X, ROBOT_Y, ROBOT_BASE_Z),
        "yaw_deg": -90.0,
    },
    {
        "name": "northeast",
        "position": (ROBOT_X, ROBOT_Y, ROBOT_BASE_Z),
        "yaw_deg": -90.0,
    },
    {
        "name": "southwest",
        "position": (-ROBOT_X, -ROBOT_Y, ROBOT_BASE_Z),
        "yaw_deg": 90.0,
    },
    {
        "name": "southeast",
        "position": (ROBOT_X, -ROBOT_Y, ROBOT_BASE_Z),
        "yaw_deg": 90.0,
    },
)


def parse_args() -> argparse.Namespace:
    """Parse scene output and display options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--curobo-config", type=Path, default=DEFAULT_CUROBO_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument(
        "--gui", action="store_true", help="Keep an interactive PyBullet window open."
    )
    return parser.parse_args()


def load_robot_geometry_metadata(
    urdf_path: Path, curobo_config_path: Path
) -> dict[str, object]:
    """Validate and describe the shared mesh and 295-sphere robot model."""
    resolved_urdf = urdf_path.resolve()
    resolved_config = curobo_config_path.resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(resolved_urdf)
    if not resolved_config.is_file():
        raise FileNotFoundError(resolved_config)

    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    kinematics = config.get("kinematics", {})
    configured_urdf = Path(kinematics.get("urdf_path", "")).resolve()
    if configured_urdf != resolved_urdf:
        raise ValueError(
            "The PyBullet mesh URDF and cuRobo sphere-model URDF must match: "
            f"{resolved_urdf} != {configured_urdf}"
        )

    collision_spheres = kinematics.get("collision_spheres", {})
    sphere_count = sum(len(spheres) for spheres in collision_spheres.values())
    if sphere_count != EXPECTED_COLLISION_SPHERES:
        raise ValueError(
            f"Expected the active {EXPECTED_COLLISION_SPHERES}-sphere Franka "
            f"model, but {resolved_config} contains {sphere_count} spheres"
        )

    urdf_root = ET.parse(resolved_urdf).getroot()
    visual_meshes = {
        mesh.attrib["filename"]
        for mesh in urdf_root.findall(".//visual/geometry/mesh")
    }
    collision_meshes = {
        mesh.attrib["filename"]
        for mesh in urdf_root.findall(".//collision/geometry/mesh")
    }
    return {
        "urdf": str(resolved_urdf),
        "curobo_config": str(resolved_config),
        "base_link": kinematics.get("base_link"),
        "visual_representation": "URDF link meshes",
        "visual_mesh_count": len(visual_meshes),
        "mesh_collision_representation": "URDF collision link meshes",
        "collision_mesh_count": len(collision_meshes),
        "planning_collision_representation": "MorphIt/cuRobo link spheres",
        "collision_sphere_count": sphere_count,
        "sphere_link_count": len(collision_spheres),
    }


def create_box(client, dimensions, position, color, *, collision=True) -> int:
    """Create a fixed cuboid using full dimensions."""
    half_extents = (0.5 * np.asarray(dimensions, dtype=np.float64)).tolist()
    collision_id = (
        client.createCollisionShape(client.GEOM_BOX, halfExtents=half_extents)
        if collision
        else -1
    )
    visual_id = client.createVisualShape(
        client.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
    )
    return int(
        client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
        )
    )


def quaternion_from_z_axis(direction: np.ndarray) -> list[float]:
    """Return an XYZW quaternion rotating local +Z onto direction."""
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-10:
        return [0.0, 0.0, 0.0, 1.0] if dot > 0.0 else [1.0, 0.0, 0.0, 0.0]
    axis = cross / np.linalg.norm(cross)
    half_angle = 0.5 * math.acos(dot)
    return [*(axis * math.sin(half_angle)).tolist(), math.cos(half_angle)]


def create_cylinder_between(client, start, end, radius, color) -> int:
    """Create a visual cylinder spanning two points."""
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    visual_id = client.createVisualShape(
        client.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
    )
    return int(
        client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=((start_array + end_array) * 0.5).tolist(),
            baseOrientation=quaternion_from_z_axis(delta),
        )
    )


def add_ground_and_grid(client) -> None:
    """Add a neutral floor and a 0.25 m metric reference grid."""
    create_box(client, (3.4, 3.0, 0.025), (0.0, 0.0, -0.0175), [0.87, 0.89, 0.91, 1.0])
    values = np.arange(-1.5, 1.5001, GRID_SPACING)
    for value in values:
        major = np.isclose(value % 0.5, 0.0, atol=1e-8)
        color = [0.48, 0.52, 0.57, 0.65] if major else [0.67, 0.70, 0.74, 0.45]
        width = 0.006 if major else 0.003
        create_box(
            client,
            (3.0, width, 0.004),
            (0.0, float(value), 0.002),
            color,
            collision=False,
        )
        create_box(
            client,
            (width, 3.0, 0.004),
            (float(value), 0.0, 0.002),
            color,
            collision=False,
        )


def add_axis_triad(client) -> None:
    """Draw a metric XYZ triad above the table center so it remains visible."""
    origin = np.array([0.0, 0.0, TABLE_DIMS[2] + 0.035])
    length = 0.42
    axes = (
        (np.array([length, 0.0, 0.0]), [0.92, 0.12, 0.12, 1.0]),
        (np.array([0.0, length, 0.0]), [0.10, 0.72, 0.20, 1.0]),
        (np.array([0.0, 0.0, length]), [0.12, 0.30, 0.95, 1.0]),
    )
    for offset, color in axes:
        endpoint = origin + offset
        create_cylinder_between(client, origin, endpoint, 0.012, color)
        sphere = client.createVisualShape(
            client.GEOM_SPHERE, radius=0.027, rgbaColor=color
        )
        client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=sphere,
            basePosition=endpoint.tolist(),
        )


def add_table_and_mounts(client) -> tuple[int, list[int]]:
    """Create the central box-table and four dedicated robot pedestals."""
    table_id = create_box(client, TABLE_DIMS, TABLE_POSE, [0.50, 0.27, 0.12, 1.0])
    mount_ids = []
    for robot in ROBOT_LAYOUT:
        x, y, _ = robot["position"]
        pedestal_id = create_box(
            client,
            PEDESTAL_DIMS,
            (x, y, PEDESTAL_DIMS[2] * 0.5),
            [0.23, 0.28, 0.33, 1.0],
        )
        mount_id = create_box(
            client,
            MOUNT_DIMS,
            (x, y, PEDESTAL_DIMS[2] + MOUNT_DIMS[2] * 0.5),
            [0.12, 0.15, 0.18, 1.0],
        )
        mount_ids.extend((pedestal_id, mount_id))
    return table_id, mount_ids


def load_robots(client, urdf_path: Path) -> list[int]:
    """Load four fixed-base Frankas from the same URDF link meshes."""
    robot_ids = []
    flags = (
        client.URDF_USE_SELF_COLLISION | client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    )
    for robot in ROBOT_LAYOUT:
        orientation = client.getQuaternionFromEuler(
            (0.0, 0.0, math.radians(robot["yaw_deg"]))
        )
        body_id = int(
            client.loadURDF(
                str(urdf_path),
                basePosition=robot["position"],
                baseOrientation=orientation,
                useFixedBase=True,
                flags=flags,
            )
        )
        movable = [
            joint_index
            for joint_index in range(client.getNumJoints(body_id))
            if client.getJointInfo(body_id, joint_index)[2] == client.JOINT_REVOLUTE
        ]
        for joint_index, position in zip(movable[:7], READY_CONFIGURATION):
            client.resetJointState(body_id, joint_index, position)
        robot_ids.append(body_id)
    return robot_ids


def collision_report(client, robot_ids: list[int], table_id: int) -> dict[str, object]:
    """Report penetrations in the posed scene without changing it."""
    client.performCollisionDetection()
    robot_table = []
    robot_robot = []
    for index, robot_id in enumerate(robot_ids):
        contacts = client.getContactPoints(bodyA=robot_id, bodyB=table_id)
        if contacts:
            robot_table.append(
                {"robot": ROBOT_LAYOUT[index]["name"], "contacts": len(contacts)}
            )
    for first in range(len(robot_ids)):
        for second in range(first + 1, len(robot_ids)):
            contacts = client.getContactPoints(
                bodyA=robot_ids[first], bodyB=robot_ids[second]
            )
            if contacts:
                robot_robot.append(
                    {
                        "robots": [
                            ROBOT_LAYOUT[first]["name"],
                            ROBOT_LAYOUT[second]["name"],
                        ],
                        "contacts": len(contacts),
                    }
                )
    return {
        "posed_scene_collision_free": not robot_table and not robot_robot,
        "robot_table_contacts": robot_table,
        "inter_robot_contacts": robot_robot,
    }


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a Windows UI font with a portable fallback."""
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


def annotate_image(image: Image.Image) -> Image.Image:
    """Add scale, dimensions, axes, and base coordinates above the viewport."""
    source = image.convert("RGB")
    header_height = 230
    canvas = Image.new(
        "RGB", (source.width, source.height + header_height), (28, 35, 44)
    )
    canvas.paste(source, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.line((800, 34, 800, 196), fill=(88, 99, 112), width=2)
    draw.text(
        (48, 32),
        "Four-Franka shared table environment",
        font=font(34, bold=True),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (48, 86),
        "Grid spacing: 0.25 m   |   dimensions in metres",
        font=font(23),
        fill=(221, 229, 239, 255),
    )
    draw.text(
        (48, 130),
        "Central table-box: 1.80 x 0.90 x 0.68",
        font=font(23),
        fill=(243, 205, 155, 255),
    )
    draw.text(
        (48, 174),
        "Four identical mesh robots | 295 planning spheres per arm",
        font=font(23),
        fill=(221, 229, 239, 255),
    )
    draw.text(
        (842, 42),
        "Axis triad at table center: X red  Y green  Z blue",
        font=font(23),
        fill=(221, 229, 239, 255),
    )
    draw.text(
        (842, 91),
        "Opposing bases: 1.48 m   |   same-side spacing: 1.04 m",
        font=font(23),
        fill=(221, 229, 239, 255),
    )
    draw.text(
        (842, 140),
        "Base center to nearest table edge: 0.29 m",
        font=font(23),
        fill=(221, 229, 239, 255),
    )
    draw.text(
        (842, 189),
        "All four arms face inward; posed scene is collision-free",
        font=font(23),
        fill=(157, 232, 180, 255),
    )
    return canvas


def render_scene(client, output_path: Path, width: int, height: int) -> None:
    """Render the complete scene from an elevated three-quarter camera."""
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
    image = annotate_image(Image.fromarray(pixels[..., :3]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def write_manifest(
    output_path: Path,
    geometry: dict[str, object],
    collision: dict[str, object],
) -> None:
    """Save all scene coordinates beside the image for planner integration."""
    manifest = {
        "units": "metres",
        "world_frame": {"x": "red", "y": "green", "z": "blue"},
        "grid_spacing": GRID_SPACING,
        "table": {"type": "cuboid", "dimensions": TABLE_DIMS, "position": TABLE_POSE},
        "pedestal_dimensions": PEDESTAL_DIMS,
        "mount_dimensions": MOUNT_DIMS,
        "robot_geometry": geometry,
        "total_planning_collision_spheres": (
            len(ROBOT_LAYOUT) * int(geometry["collision_sphere_count"])
        ),
        "robot_ready_configuration": READY_CONFIGURATION,
        "robots": [
            {
                **robot,
                "geometry": {
                    "urdf": geometry["urdf"],
                    "curobo_config": geometry["curobo_config"],
                    "collision_sphere_count": geometry["collision_sphere_count"],
                },
            }
            for robot in ROBOT_LAYOUT
        ],
        "clearances": {
            "opposing_base_center_distance": 2.0 * ROBOT_Y,
            "base_center_to_nearest_table_edge": ROBOT_Y - 0.5 * TABLE_DIMS[1],
            "same_side_base_center_distance": 2.0 * ROBOT_X,
        },
        "collision_check": collision,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Construct, validate, render, and optionally display the scene."""
    args = parse_args()
    urdf_path = args.urdf.resolve()
    curobo_config_path = args.curobo_config.resolve()
    output_dir = args.output_dir.resolve()
    geometry = load_robot_geometry_metadata(urdf_path, curobo_config_path)
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")

    import pybullet as pb
    from pybullet_utils.bullet_client import BulletClient

    client = BulletClient(connection_mode=pb.GUI if args.gui else pb.DIRECT)
    try:
        client.configureDebugVisualizer(client.COV_ENABLE_GUI, 0)
        client.setGravity(0.0, 0.0, -9.81)
        client.resetSimulation()
        add_ground_and_grid(client)
        table_id, _ = add_table_and_mounts(client)
        robot_ids = load_robots(client, urdf_path)
        add_axis_triad(client)
        collision = collision_report(client, robot_ids, table_id)
        if not collision["posed_scene_collision_free"]:
            raise RuntimeError(
                f"The nominal four-arm layout has collisions: {collision}"
            )

        image_path = output_dir / "four_franka_table_scene.png"
        manifest_path = output_dir / "four_franka_table_scene.json"
        render_scene(client, image_path, args.width, args.height)
        write_manifest(manifest_path, geometry, collision)
        print(f"saved scene image: {image_path}")
        print(f"saved scene manifest: {manifest_path}")
        print(
            "robot geometry: "
            f"{geometry['visual_mesh_count']} visual meshes, "
            f"{geometry['collision_sphere_count']} planning spheres per arm"
        )
        print(json.dumps(collision, indent=2))

        if args.gui:
            client.resetDebugVisualizerCamera(3.25, 42.0, -31.0, [0.0, 0.0, 0.72])
            input("Press Enter to close the PyBullet scene...")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
