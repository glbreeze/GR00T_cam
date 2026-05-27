#!/bin/bash
#SBATCH --job-name=rc_smoke
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/_smoke/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/_smoke/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --account=sfscai
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Shanghai-HPC smoke test for the RoboCasa v0.2 gr00t eval pipeline. Adapts
# scripts/sbatch_lg/eval_robocasa.sbatch to this cluster's paths/account and
# trims to: (A) sim env probe, then (B) a tiny baseline server+client run.
# Goal is "does the whole pipeline run end-to-end", not success rate.
#
# Cluster-specific paths (Shanghai), overridable via env:
#   CONDA_BASE=/gpfsnyu/home/lg154/miniconda3
#   GR00T_ENV=/scratch/lg154/conda_envs/gr00t
#   ROBOCASA_ENV=/scratch/lg154/conda_envs/robocasa24
#   CKPT_DIR=<repo>/checkpoints/gr00t_robocasa_baseline_2gpu_b8/checkpoint-30000
#   ENV_NAME=OpenDrawer  NUM_TRIALS=2  MAX_STEPS=300  PORT=19311

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
CONDA_BASE=${CONDA_BASE:-/gpfsnyu/home/lg154/miniconda3}
GR00T_ENV=${GR00T_ENV:-/scratch/lg154/conda_envs/gr00t}
ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/lg154/conda_envs/robocasa24}

CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/gr00t_robocasa_baseline_2gpu_b8/checkpoint-30000}
MODEL_NAME=${MODEL_NAME:-baseline_30k_smoke}
DATA_CONFIG=${DATA_CONFIG:-robocasa}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
DENOISING_STEPS=${DENOISING_STEPS:-4}
USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}
# CamVLA checkpoints (use_camvla_model in config.json) were trained camera-aware
# with geometric augs disabled; the server transform must match. Defaults track
# USE_CAMERA_PARAMS but can be set independently.
DISABLE_GEOMETRIC_AUGS=${DISABLE_GEOMETRIC_AUGS:-${USE_CAMERA_PARAMS}}

ENV_NAME=${ENV_NAME:-OpenDrawer}
NUM_TRIALS=${NUM_TRIALS:-2}
MAX_STEPS=${MAX_STEPS:-300}
REPLAN_STEPS=${REPLAN_STEPS:-16}
PORT=${PORT:-19311}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}

LOG_ROOT=${REPO_ROOT}/logs/eval_robocasa/_smoke
SERVER_LOG=${LOG_ROOT}/server_${SLURM_JOB_ID:-manual}.log
CLIENT_LOG=${LOG_ROOT}/client_${SLURM_JOB_ID:-manual}.log
PROBE_LOG=${LOG_ROOT}/probe_${SLURM_JOB_ID:-manual}.log
SUMMARY_PATH=${LOG_ROOT}/summary_${SLURM_JOB_ID:-manual}.json
mkdir -p "${LOG_ROOT}"

# ---- sanity ----
[[ -f "${CKPT_DIR}/config.json" ]] || { echo "Missing ${CKPT_DIR}/config.json" >&2; exit 1; }
[[ -d "${ROBOCASA_ENV}" ]] || { echo "Missing robocasa env ${ROBOCASA_ENV}" >&2; exit 1; }
[[ -d "${GR00T_ENV}" ]] || { echo "Missing gr00t env ${GR00T_ENV}" >&2; exit 1; }

# ---- per-job scratch + headless rendering ----
JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER}/gr00t_rcsmoke_${SLURM_JOB_ID:-$$}
export TMPDIR=${JOB_TMP}/tmp XDG_CACHE_HOME=${JOB_TMP}/xdg MPLCONFIGDIR=${JOB_TMP}/mpl
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export ROBOCASA_HOME=${ROBOCASA_HOME:-/scratch/lg154/.cache/robocasa24_home}
mkdir -p "${ROBOCASA_HOME}"

echo "===== RC SMOKE ====="; hostname; nvidia-smi -L 2>/dev/null || true
echo "ckpt:   ${CKPT_DIR}"
echo "task:   ${ENV_NAME}  trials=${NUM_TRIALS}  max_steps=${MAX_STEPS}  camera=${USE_CAMERA_PARAMS}"
echo "===================="

# -------- Phase A: sim env probe --------
echo ">>> Phase A: sim env probe (${ENV_NAME})  log: ${PROBE_LOG}"
(
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ROBOCASA_ENV}"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 python -u examples/RoboCasa/eval/_probe_env.py --env-name "${ENV_NAME}" --img-size 128
) > "${PROBE_LOG}" 2>&1
echo "    probe rc=$?  (tail below)"; tail -n 12 "${PROBE_LOG}"
grep -q "PROBE OK" "${PROBE_LOG}" || { echo "PROBE FAILED" >&2; exit 1; }

# -------- Phase B: gr00t server (background) --------
SERVER_CAM_FLAGS=()
[[ "${USE_CAMERA_PARAMS}" == "true" ]] && SERVER_CAM_FLAGS+=("--use-camera-params")
[[ "${DISABLE_GEOMETRIC_AUGS}" == "true" ]] && SERVER_CAM_FLAGS+=("--disable-geometric-augs")
echo ">>> Phase B: start gr00t server on :${PORT}  flags=[${SERVER_CAM_FLAGS[*]:-none}]  log: ${SERVER_LOG}"
(
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${GR00T_ENV}"
  cd "${REPO_ROOT}"
  exec env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/inference_service.py \
    --server --model_path "${CKPT_DIR}" --data_config "${DATA_CONFIG}" \
    --embodiment-tag "${EMBODIMENT_TAG}" --denoising-steps "${DENOISING_STEPS}" \
    --port "${PORT}" --host 0.0.0.0 "${SERVER_CAM_FLAGS[@]}"
) > "${SERVER_LOG}" 2>&1 &
server_pid=$!
trap 'kill -TERM "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
while ! python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', ${PORT}))" 2>/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Server died before port ready. Tail:" >&2; tail -n 60 "${SERVER_LOG}" >&2; exit 1
  fi
  (( SECONDS > deadline )) && { echo "Timed out waiting for :${PORT}" >&2; tail -n 60 "${SERVER_LOG}" >&2; exit 1; }
  sleep 2
done
echo "    server up on :${PORT}"

# -------- Phase B: client (2 trials) --------
echo ">>> Phase B: client  log: ${CLIENT_LOG}"
CAMERA_FLAG=(); [[ "${USE_CAMERA_PARAMS}" == "true" ]] && CAMERA_FLAG+=("--use-camera-params")
(
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ROBOCASA_ENV}"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 python -u examples/RoboCasa/eval/run_robocasa_eval.py \
    --host localhost --port "${PORT}" --model-name "${MODEL_NAME}" \
    --env-name "${ENV_NAME}" --num-trials "${NUM_TRIALS}" --max-steps "${MAX_STEPS}" \
    --replan-steps "${REPLAN_STEPS}" --seed 0 --video-every 0 \
    --summary-out-path "${SUMMARY_PATH}" "${CAMERA_FLAG[@]}"
) > "${CLIENT_LOG}" 2>&1
client_rc=$?
echo "    client rc=${client_rc}  (tail below)"; tail -n 20 "${CLIENT_LOG}"
(( client_rc == 0 )) || { echo "CLIENT FAILED" >&2; exit "${client_rc}"; }

echo "===== SMOKE OK ====="
echo "summary: ${SUMMARY_PATH}"; cat "${SUMMARY_PATH}" 2>/dev/null | head -40
