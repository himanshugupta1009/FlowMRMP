"""FlowMRMP-local filesystem locations for the Franka planning tools."""

from __future__ import annotations

from pathlib import Path


FLOWMRMP_ROOT = Path(__file__).resolve().parents[2]
CORE_RRT_SRC = FLOWMRMP_ROOT / "mrmp_with_kite_extend" / "src"
FLOW_EB_RRT_SRC = FLOWMRMP_ROOT / "src"

DEFAULT_RAW_TRAJECTORY_DATASET = FLOWMRMP_ROOT / "data" / "dataset200k.h5"
DEFAULT_BENCHMARK_PROBLEMS = (
    FLOWMRMP_ROOT / "data" / "benchmarks" / "franka_reachable_ab_100.npz"
)
DEFAULT_BENCHMARK_PROBLEMS_OUTPUT = (
    FLOWMRMP_ROOT / "data" / "benchmarks" / "franka_reachable_ab_100.npz"
)
DEFAULT_URDF = FLOWMRMP_ROOT / "assets" / "robots" / "panda" / "panda.urdf"
DEFAULT_CHECKPOINT = (
    FLOWMRMP_ROOT
    / "checkpoints"
    / "franka_edge_flow"
    / "franka_edge_flow_k32_200k_pool128_n350000_max50_fullvalid_mps_v1"
    / "best_inference.pt"
)
DEFAULT_FLOW_RESULTS_ROOT = FLOWMRMP_ROOT / "results" / "franka_flow_eb_rrt"
DEFAULT_VANILLA_RESULTS_ROOT = FLOWMRMP_ROOT / "results" / "franka_vanilla_rrt"


def existing_result_dir(planner: str, run_name: str) -> Path:
    """Return a saved-run location contained in FlowMRMP."""
    return FLOWMRMP_ROOT / "results" / planner / run_name


def path_label(path: Path) -> str:
    """Return a stable repository-relative label when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(FLOWMRMP_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)
