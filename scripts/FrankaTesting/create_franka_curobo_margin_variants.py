"""Create self-collision margin variants from a fitted cuRobo robot model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT_DIR / "assets" / "robots" / "panda" / "curobo" / "panda_morphit_density2.yml"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--margins-mm",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0],
        help="Total radius-sum reduction for each checked link pair.",
    )
    return parser.parse_args()


def margin_label(margin_mm: float) -> str:
    return f"{margin_mm:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    source = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    robot = source.get("robot_cfg", source)
    links = robot["kinematics"]["collision_link_names"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for margin_mm in args.margins_mm:
        variant = yaml.safe_load(args.input.read_text(encoding="utf-8"))
        variant_robot = variant.get("robot_cfg", variant)
        # cuRobo sums the two link buffers for each sphere pair. Splitting the
        # requested pair margin equally gives the stated total reduction.
        per_link_buffer_m = -0.5 * margin_mm / 1000.0
        variant_robot["kinematics"]["self_collision_buffer"] = {
            link: per_link_buffer_m for link in links
        }
        output = args.output_dir / (
            f"{args.input.stem}_selfmargin{margin_label(margin_mm)}mm.yml"
        )
        output.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        outputs.append(
            {
                "path": str(output.resolve()),
                "total_pair_margin_mm": margin_mm,
                "per_link_buffer_m": per_link_buffer_m,
            }
        )
    report = {
        "source": str(args.input.resolve()),
        "world_collision_spheres_changed": False,
        "self_collision_pair_distance_only": True,
        "outputs": outputs,
    }
    report_path = args.output_dir / "panda_morphit_self_margin_variants.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
