#!/bin/bash
#SBATCH --job-name=convert_robocasa
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/convert_robocasa/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/convert_robocasa/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#
# Convert openpi-flavor RoboCasa LeRobot (PNG images) -> GR00T-flavor (MP4 videos).
# CPU/IO-bound (image decode + h264 encode); no GPU. Mirrors the local
# scripts/convert_robocasa.sh but with HPC paths + a slurm allocation.
#
# Override SRC/DST/MAX_EPISODES via `sbatch --export=...` or env.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/convert_robocasa

source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/gr00t

REPO_ROOT=${REPO_ROOT:-/scratch/lg154/Research/GR00T_cam}
# openpi-flavor source (ROBOCASA24_LEROBOT_ROOT from pi0_robocasa24_all24_baseline);
# world-readable under yz11445's scratch. Override SRC to a local copy if preferred.
SRC=${SRC:-/scratch/yz11445/robocasa24/all24_human_camaware}
DST=${DST:-/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
CRF=${CRF:-23}
MAX_EPISODES=${MAX_EPISODES:-0}

EXTRA_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=(--overwrite)
[[ "${MAX_EPISODES}" -gt 0 ]] && EXTRA_ARGS+=(--max-episodes "${MAX_EPISODES}")

cd "${REPO_ROOT}"
echo "[convert_robocasa] SRC=${SRC} DST=${DST} MAX_EPISODES=${MAX_EPISODES} workers=${NUM_WORKERS}"

exec python examples/RoboCasa/convert_openpi_lerobot_to_gr00t.py \
  --src "${SRC}" \
  --dst "${DST}" \
  --num-workers "${NUM_WORKERS}" \
  --crf "${CRF}" \
  "${EXTRA_ARGS[@]}"
