#!/usr/bin/env python3
"""Run a paper plot module after limiting it to environments on disk."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from output_paths import DEFAULT_RESULTS_ROOT, environment_results_dir


def has_runs(system: str, environment: str) -> bool:
    directory = environment_results_dir(DEFAULT_RESULTS_ROOT, system, environment)
    return directory.is_dir() and any(directory.glob(f"{system}_a*_tests*"))


def restrict(module: ModuleType, seen: set[int] | None = None) -> None:
    seen = seen or set()
    if id(module) in seen:
        return
    seen.add(id(module))

    mapping = getattr(module, "SYSTEM_ENVIRONMENTS", None)
    if isinstance(mapping, dict):
        available = {
            system: tuple(env for env in environments if has_runs(system, env))
            for system, environments in mapping.items()
        }
        available = {system: envs for system, envs in available.items() if envs}
        module.SYSTEM_ENVIRONMENTS = available
        if hasattr(module, "SYSTEMS"):
            module.SYSTEMS = tuple(system for system in module.SYSTEMS if system in available)

    for value in vars(module).values():
        if isinstance(value, ModuleType) and value.__name__.startswith("plot_"):
            restrict(value, seen)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_partial_plot.py MODULE [ARGS ...]")
    module_name = Path(sys.argv[1]).stem
    sys.argv = [sys.argv[1], *sys.argv[2:]]
    module = importlib.import_module(module_name)
    restrict(module)
    module.main()


if __name__ == "__main__":
    main()
