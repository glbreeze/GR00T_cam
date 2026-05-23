#!/bin/bash
#SBATCH --job-name=eval_gr00t_libero
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/eval_libero/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/eval_libero/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Single-suite LIBERO eval for one gr00t checkpoint. Spins up the gr00t
# inference server in the `gr00t` conda env (model + transforms + bf16) and
# drives rollouts via `examples/Libero/eval/run_libero_eval.py` in the openpi
# venv (which has libero + robosuite + mujoco). Both processes live on the
# same node and talk over ZMQ on localhost.
#
# Required env (set by the dispatcher / caller):
#   CKPT_DIR, MODEL_NAME, SUITE_NAME
#
# Optional overrides (defaults shown):
#   DATA_CONFIG=examples.Libero.custom_data_config:LiberoDataConfig
#   EMBODIMENT_TAG=new_embodiment
#   DENOISING_STEPS=4
#   USE_CAMERA_PARAMS=false      (true for CamVLA modes 3/4)
#   NUM_TRIALS_PER_TASK=50
#   REPLAN_STEPS=5
#   NUM_STEPS_WAIT=10
#   SEED=7
#   PORT=19200
#   SERVER_START_TIMEOUT=900     (seconds; ckpt load can take 1-2 min)
#   TASK_ID_START=0, TASK_ID_END=-1  (-1 = all tasks in the suite)

set -euo pipefail

# ---- required inputs ----
: "${CKPT_DIR:?required: path to a checkpoint dir (must contain config.json)}"
: "${MODEL_NAME:?required: short tag used in output paths}"
: "${SUITE_NAME:?required: libero_spatial | libero_object | libero_goal | libero_10 | libero_90}"

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
OPENPI_ROOT=/scratch/lg154/Research/openpi
LIBERO_SRC=${OPENPI_ROOT}/third_party/libero/libero/libero

# ---- defaults ----
DATA_CONFIG=${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
DENOISING_STEPS=${DENOISING_STEPS:-4}
USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
REPLAN_STEPS=${REPLAN_STEPS:-5}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SEED=${SEED:-7}

PORT=${PORT:-19200}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}

TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}

# ---- output paths ----
LOG_ROOT=${REPO_ROOT}/logs/eval_libero/${MODEL_NAME}/${SUITE_NAME}
SUMMARY_PATH=${LOG_ROOT}/summary/${SLURM_JOB_ID:-manual}.json
SERVER_LOG=${LOG_ROOT}/logs/server_${SLURM_JOB_ID:-manual}.log
CLIENT_LOG=${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}.log
VIDEO_OUT=${LOG_ROOT}/videos/${SLURM_JOB_ID:-manual}

mkdir -p "$(dirname "${SUMMARY_PATH}")" "${VIDEO_OUT}" "$(dirname "${SERVER_LOG}")"

# ---- ckpt sanity ----
if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
  echo "Missing checkpoint config: ${CKPT_DIR}/config.json" >&2
  exit 1
fi

# ---- per-job scratch + env (prevents concurrent evals from stomping shared state) ----
JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER}/gr00t_eval_${SLURM_JOB_ID:-$$}
export TMPDIR=${JOB_TMP}/tmp TMP=${JOB_TMP}/tmp TEMP=${JOB_TMP}/tmp
export XDG_CACHE_HOME=${JOB_TMP}/xdg_cache
export XDG_CONFIG_HOME=${JOB_TMP}/xdg_config
export XDG_STATE_HOME=${JOB_TMP}/xdg_state
export MPLCONFIGDIR=${JOB_TMP}/mplconfig
export LIBERO_CONFIG_PATH=${JOB_TMP}/libero_config
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" \
         "${MPLCONFIGDIR}" "${LIBERO_CONFIG_PATH}"

# LIBERO config the benchmark loader reads (points at the openpi submodule).
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${LIBERO_SRC}
bddl_files: ${LIBERO_SRC}/bddl_files
init_states: ${LIBERO_SRC}/init_files
datasets: ${LIBERO_SRC}/../datasets
assets: ${LIBERO_SRC}/assets
EOF

# Headless rendering on compute nodes (no X server). egl is the default on Linux.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

echo "===== EVAL ENV ====="
echo "host:     $(hostname)"
nvidia-smi -L 2>/dev/null || true
echo "ckpt:     ${CKPT_DIR}"
echo "config:   ${DATA_CONFIG}"
echo "tag:      ${EMBODIMENT_TAG}"
echo "suite:    ${SUITE_NAME}   tasks: [${TASK_ID_START}, ${TASK_ID_END})"
echo "trials:   ${NUM_TRIALS_PER_TASK}/task   replan_steps=${REPLAN_STEPS}"
echo "camera:   USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS}"
echo "summary:  ${SUMMARY_PATH}"
echo "videos:   ${VIDEO_OUT}"
echo "===================="

# -------- 1. start the gr00t inference server (background) --------
echo "Starting gr00t inference server on port ${PORT} (log: ${SERVER_LOG})"
(
  source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
  conda activate /scratch/lg154/conda-envs/gr00t
  cd "${REPO_ROOT}"
  exec env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/inference_service.py \
    --server \
    --model_path "${CKPT_DIR}" \
    --data_config "${DATA_CONFIG}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --denoising-steps "${DENOISING_STEPS}" \
    --port "${PORT}" \
    --host 0.0.0.0
) > "${SERVER_LOG}" 2>&1 &
server_pid=$!
trap 'kill -TERM "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

# Wait for the server's port to accept connections, or fail loudly.
deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
while ! python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', ${PORT}))" 2>/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Server exited before port ${PORT} was ready. Last 80 lines of ${SERVER_LOG}:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for port ${PORT}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
echo "Server is up on tcp://127.0.0.1:${PORT}"

# -------- 2. run the LIBERO client --------
CAMERA_FLAG=()
if [[ "${USE_CAMERA_PARAMS}" == "true" ]]; then
  CAMERA_FLAG+=("--use-camera-params")
fi

echo "Starting LIBERO client (log: ${CLIENT_LOG})"
(
  export PYTHONPATH="${REPO_ROOT}:${OPENPI_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 "${OPENPI_ROOT}/.venv/bin/python" -u examples/Libero/eval/run_libero_eval.py \
    --host localhost \
    --port "${PORT}" \
    --model-name "${MODEL_NAME}" \
    --task-suite-name "${SUITE_NAME}" \
    --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --replan-steps "${REPLAN_STEPS}" \
    --num-steps-wait "${NUM_STEPS_WAIT}" \
    --task-id-start "${TASK_ID_START}" \
    --task-id-end "${TASK_ID_END}" \
    --seed "${SEED}" \
    --video-out-path "${VIDEO_OUT}" \
    --summary-out-path "${SUMMARY_PATH}" \
    "${CAMERA_FLAG[@]}"
) 2>&1 | tee "${CLIENT_LOG}"

echo "Eval complete."
echo "Summary: ${SUMMARY_PATH}"
echo "Videos:  ${VIDEO_OUT}"
