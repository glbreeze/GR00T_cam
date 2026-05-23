#!/bin/bash
# Convert libero_cam_v2 (openpi-flavor LeRobot, PNG images) -> GR00T-flavor LeRobot (MP4 videos).
# CPU/IO-bound (image decode + h264 encode); no GPU required.
#
# Usage:
#   bash scripts/sbatch_lg/convert_libero_cam_v2.sh
#   NUM_WORKERS=16 bash scripts/sbatch_lg/convert_libero_cam_v2.sh
#   OVERWRITE=1 bash scripts/sbatch_lg/convert_libero_cam_v2.sh

set -euo pipefail

source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/gr00t

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
SRC=/scratch/lg154/cache/lerobot/glbreeze/libero_cam_v2
DST=/scratch/lg154/cache/lerobot/glbreeze/libero_cam_v2_gr00t

NUM_WORKERS=${NUM_WORKERS:-8}
CRF=${CRF:-23}

EXTRA_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

cd "${REPO_ROOT}"
python examples/Libero/convert_openpi_lerobot_to_gr00t.py \
  --src "${SRC}" \
  --dst "${DST}" \
  --num-workers "${NUM_WORKERS}" \
  --crf "${CRF}" \
  "${EXTRA_ARGS[@]}"
