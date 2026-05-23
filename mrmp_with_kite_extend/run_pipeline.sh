#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

#====================== 2D pipelines ======================#

# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting swap pipeline"
# "$PYTHON_BIN" pipeline_code/test_pipeline_swap.py
# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished swap pipeline"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting narrow corridor pipeline"
"$PYTHON_BIN" pipeline_code/test_pipeline_narrow_corridor.py
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished narrow corridor pipeline"

# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting small cluttered pipeline"
# "$PYTHON_BIN" pipeline_code/test_pipeline_small_cluttered.py
# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished small cluttered pipeline"

# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting large cluttered pipeline"
# "$PYTHON_BIN" pipeline_code/test_pipeline_large_cluttered.py
# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished large cluttered pipeline"

#====================== 3D pipelines ======================#

# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting swap 3D pipeline"
# "$PYTHON_BIN" pipeline_code/test_pipeline_swap_3d.py
# echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished swap 3D pipeline"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting large cluttered 3D pipeline"
"$PYTHON_BIN" pipeline_code/test_pipeline_large_cluttered_3d.py
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished swap 3D and large cluttered 3D pipelines"
