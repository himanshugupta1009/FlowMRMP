#!/usr/bin/env python3
"""Create the standalone Python environment for Franka FlowMRMP tools.

The normal dependencies are installed from ``requirements-franka.txt``. On
macOS, PyBullet 3.2.7's source distribution does not compile with the current
Apple Clang toolchain, so this installer copies the tested universal CPython
3.9 PyBullet binary vendored in this repository. No CuRobo checkout is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import venv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = REPOSITORY_ROOT / ".venv"
REQUIREMENTS = REPOSITORY_ROOT / "requirements-franka.txt"
MACOS_PYBULLET = REPOSITORY_ROOT / "vendor" / "pybullet_macos_py39"


def parse_args() -> argparse.Namespace:
    """Read the optional environment destination and reinstall flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the selected environment first.",
    )
    return parser.parse_args()


def environment_python(environment: Path) -> Path:
    """Return the Python executable used by a POSIX or Windows virtualenv."""
    executable = "python.exe" if platform.system() == "Windows" else "python"
    directory = "Scripts" if platform.system() == "Windows" else "bin"
    return environment / directory / executable


def run(*command: str | Path) -> None:
    """Run one setup command and stop immediately if it fails."""
    subprocess.run([str(part) for part in command], check=True)


def install_macos_pybullet(python: Path) -> None:
    """Copy the local universal PyBullet build into the new environment."""
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError(
            "The vendored macOS PyBullet extension requires CPython 3.9; "
            f"setup is running under {sys.version.split()[0]}."
        )
    if not MACOS_PYBULLET.is_dir():
        raise FileNotFoundError(MACOS_PYBULLET)

    # Ask the environment itself for its purelib location rather than assuming
    # a particular venv directory layout.
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    site_packages = Path(result.stdout.strip())
    for source in MACOS_PYBULLET.iterdir():
        destination = site_packages / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def main() -> None:
    """Create the environment, install packages, and verify core imports."""
    args = parse_args()
    environment = args.venv.expanduser().resolve()
    if args.recreate and environment.exists():
        shutil.rmtree(environment)
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)

    python = environment_python(environment)
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "-r", REQUIREMENTS)
    if platform.system() == "Darwin":
        install_macos_pybullet(python)

    run(
        python,
        "-c",
        "import h5py, matplotlib, networkx, numba, numpy, pybullet, scipy, torch, tqdm",
    )
    print(f"Standalone Franka environment ready: {environment}")


if __name__ == "__main__":
    main()
