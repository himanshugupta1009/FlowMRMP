"""Add targeted self-collision padding to the selected MorphIt sphere model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
ROBOT_CONFIG_DIR = ROOT_DIR / "assets" / "robots" / "panda" / "curobo"
DEFAULT_INPUT = ROBOT_CONFIG_DIR / "panda_morphit_density2.yml"
DEFAULT_OUTPUT = ROBOT_CONFIG_DIR / "panda_morphit_density2_safety_padded.yml"
DEFAULT_REPORT = ROBOT_CONFIG_DIR / "panda_morphit_density2_safety_padded.json"
PADDED_LINKS = ("panda_link0", "panda_link1", "panda_link5", "panda_link6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--padding-mm-per-link", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    robot = config.get("robot_cfg", config)
    kinematics = robot["kinematics"]
    padding_m = args.padding_mm_per_link / 1000.0
    kinematics["self_collision_buffer"] = {
        link: (padding_m if link in PADDED_LINKS else 0.0)
        for link in kinematics["collision_link_names"]
    }
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "world_collision_spheres_changed": False,
        "self_collision_padding_only": True,
        "padded_links": list(PADDED_LINKS),
        "padding_mm_per_link": args.padding_mm_per_link,
        "pair_padding_mm_when_both_links_are_padded": 2.0 * args.padding_mm_per_link,
        "reason": (
            "Cover holdout PyBullet collisions missed by the raw MorphIt fit at "
            "panda_link0/panda_link5 and panda_link1/panda_link6."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
