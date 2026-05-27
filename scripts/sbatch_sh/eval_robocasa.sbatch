#!/bin/bash
#SBATCH --job-name=eval_gr00t_robocasa
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=sfscai
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# NYU SHANGHAI HPC RoboCasa v0.2 eval for one gr00t checkpoint, packing MULTIPLE
# task families onto ONE GPU: the inference server (gr00t env) is loaded once and
# reused across every task in ENV_NAMES; rollouts run via
# examples/RoboCasa/eval/run_robocasa_eval.py (robocasa24 env). Counterpart to
# scripts/sbatch_lg/eval_robocasa.sbatch (NY cluster, one task per job).
#
# Sizing for the sfscai QOS (cpu=64, gres/gpu=8 per user): cpus-per-task=8 means
# 8 such jobs fill the CPU cap exactly while using 8 GPUs. submit_eval_robocasa.sh
# groups the 24 task families into 4 jobs by default (4 GPUs/eval -> run two
# checkpoints concurrently within the 8-GPU cap).
#
# Required env (set by the dispatcher / caller):
#   CKPT_DIR, MODEL_NAME, ENV_NAMES   (ENV_NAMES = space-separated task families,
#                                      e.g. "OpenDrawer CloseDrawer PnPCounterToCab")
#   (ENV_NAME, singular, is accepted as a fallback for one task.)
#
# Optional overrides (defaults shown):
#   DATA_CONFIG=robocasa  EMBODIMENT_TAG=new_embodiment  DENOISING_STEPS=4
#   USE_CAMERA_PARAMS=false   DISABLE_GEOMETRIC_AUGS=$USE_CAMERA_PARAMS
#   NUM_TRIALS=50   MAX_STEPS=500   REPLAN_STEPS=16   NUM_PARALLEL_CLIENTS=4
#   VIDEO_EVERY=10  SEED=0  PORT=19300  SERVER_START_TIMEOUT=900
#   CONDA_BASE=/gpfsnyu/home/lg154/miniconda3
#   GR00T_ENV=/scratch/lg154/conda_envs/gr00t
#   ROBOCASA_ENV=/scratch/lg154/conda_envs/robocasa24
#   ROBOCASA_HOME=/scratch/lg154/.cache/robocasa24_home

set -euo pipefail

# ---- required inputs ----
: "${CKPT_DIR:?required: path to a checkpoint dir (must contain config.json)}"
: "${MODEL_NAME:?required: short tag used in output paths}"
ENV_NAMES=${ENV_NAMES:-${ENV_NAME:?required: ENV_NAMES (space-separated) or ENV_NAME}}

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
CONDA_BASE=${CONDA_BASE:-/gpfsnyu/home/lg154/miniconda3}
GR00T_ENV=${GR00T_ENV:-/scratch/lg154/conda_envs/gr00t}
ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/lg154/conda_envs/robocasa24}

# ---- defaults ----
DATA_CONFIG=${DATA_CONFIG:-robocasa}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
DENOISING_STEPS=${DENOISING_STEPS:-4}
USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}
# CamVLA checkpoints train camera-aware with geometric augs off; the eval
# transform must match. Tracks USE_CAMERA_PARAMS by default, overridable.
DISABLE_GEOMETRIC_AUGS=${DISABLE_GEOMETRIC_AUGS:-${USE_CAMERA_PARAMS}}

NUM_TRIALS=${NUM_TRIALS:-50}
MAX_STEPS=${MAX_STEPS:-500}
REPLAN_STEPS=${REPLAN_STEPS:-16}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
# Stagger client env-build starts (seconds x shard index) so N concurrent
# robosuite.make() calls don't hit the node's gpfs/EGL at the same instant.
STAGGER_SECONDS=${STAGGER_SECONDS:-12}
VIDEO_EVERY=${VIDEO_EVERY:-10}
SEED=${SEED:-0}

# Unique per-job port so co-located jobs (multiple on one 8-GPU node) don't
# collide on localhost. Consecutive SLURM job IDs map to distinct ports.
PORT=${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}

LOG_ROOT=${REPO_ROOT}/logs/eval_robocasa/${MODEL_NAME}
SERVER_LOG=${LOG_ROOT}/_server_${SLURM_JOB_ID:-manual}.log
mkdir -p "${LOG_ROOT}"

# ---- sanity ----
[[ -f "${CKPT_DIR}/config.json" ]] || { echo "Missing ${CKPT_DIR}/config.json" >&2; exit 1; }
[[ -d "${ROBOCASA_ENV}" ]] || { echo "Missing robocasa env ${ROBOCASA_ENV}" >&2; exit 1; }
[[ -d "${GR00T_ENV}" ]] || { echo "Missing gr00t env ${GR00T_ENV}" >&2; exit 1; }

# ---- per-job scratch + headless rendering ----
JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER}/gr00t_rceval_${SLURM_JOB_ID:-$$}
export TMPDIR=${JOB_TMP}/tmp TMP=${JOB_TMP}/tmp TEMP=${JOB_TMP}/tmp
export XDG_CACHE_HOME=${JOB_TMP}/xdg_cache XDG_CONFIG_HOME=${JOB_TMP}/xdg_config
export XDG_STATE_HOME=${JOB_TMP}/xdg_state MPLCONFIGDIR=${JOB_TMP}/mplconfig
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export ROBOCASA_HOME=${ROBOCASA_HOME:-/scratch/lg154/.cache/robocasa24_home}
mkdir -p "${ROBOCASA_HOME}"

read -r -a TASK_ARR <<< "${ENV_NAMES}"
echo "===== ROBOCASA EVAL (Shanghai, packed) ====="
echo "host:     $(hostname)"; nvidia-smi -L 2>/dev/null || true
echo "ckpt:     ${CKPT_DIR}"
echo "config:   ${DATA_CONFIG}  tag: ${EMBODIMENT_TAG}  camera=${USE_CAMERA_PARAMS} no_geo_aug=${DISABLE_GEOMETRIC_AUGS}"
echo "tasks (${#TASK_ARR[@]}): ${TASK_ARR[*]}"
echo "per task: trials=${NUM_TRIALS} max_steps=${MAX_STEPS} replan=${REPLAN_STEPS} clients=${NUM_PARALLEL_CLIENTS}"
echo "============================================"

# -------- start the gr00t inference server ONCE (reused across all tasks) --------
SERVER_CAM_FLAGS=()
[[ "${USE_CAMERA_PARAMS}" == "true" ]] && SERVER_CAM_FLAGS+=("--use-camera-params")
[[ "${DISABLE_GEOMETRIC_AUGS}" == "true" ]] && SERVER_CAM_FLAGS+=("--disable-geometric-augs")
echo "Starting gr00t server on :${PORT}  flags=[${SERVER_CAM_FLAGS[*]:-none}]  (log: ${SERVER_LOG})"
(
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${GR00T_ENV}"
  cd "${REPO_ROOT}"
  # Use the GPU SLURM allocated to this job (do NOT hardcode device 0 -- with
  # multiple jobs per node that would collide on one physical GPU).
  exec env PYTHONUNBUFFERED=1 python -u scripts/inference_service.py \
    --server --model_path "${CKPT_DIR}" --data_config "${DATA_CONFIG}" \
    --embodiment-tag "${EMBODIMENT_TAG}" --denoising-steps "${DENOISING_STEPS}" \
    --port "${PORT}" --host 0.0.0.0 "${SERVER_CAM_FLAGS[@]}"
) > "${SERVER_LOG}" 2>&1 &
server_pid=$!
trap 'kill -TERM "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
while ! python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', ${PORT}))" 2>/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Server died before port ready. Tail:" >&2; tail -n 80 "${SERVER_LOG}" >&2; exit 1
  fi
  (( SECONDS > deadline )) && { echo "Timed out waiting for :${PORT}" >&2; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
  sleep 2
done
echo "Server is up on tcp://127.0.0.1:${PORT}"

CAMERA_FLAG=(); [[ "${USE_CAMERA_PARAMS}" == "true" ]] && CAMERA_FLAG+=("--use-camera-params")

# -------- loop over task families, reusing the server --------
overall_rc=0
for ENV_NAME in "${TASK_ARR[@]}"; do
  TASK_ROOT=${LOG_ROOT}/${ENV_NAME}
  SUMMARY_PATH=${TASK_ROOT}/summary/${SLURM_JOB_ID:-manual}.json
  VIDEO_OUT=${TASK_ROOT}/videos/${SLURM_JOB_ID:-manual}
  SHARD_SUMMARY_ROOT=${TASK_ROOT}/summary/shards_${SLURM_JOB_ID:-manual}
  mkdir -p "$(dirname "${SUMMARY_PATH}")" "${VIDEO_OUT}" "${TASK_ROOT}/logs" "${SHARD_SUMMARY_ROOT}"

  n_clients=${NUM_PARALLEL_CLIENTS}
  (( n_clients > NUM_TRIALS )) && n_clients=${NUM_TRIALS}
  echo ">>> [$ENV_NAME] ${NUM_TRIALS} trials across ${n_clients} client(s)"

  client_pids=()
  for i in $(seq 0 $((n_clients - 1))); do
    trial_start=$(( i * NUM_TRIALS / n_clients ))
    trial_end=$(( (i + 1) * NUM_TRIALS / n_clients ))
    shard_trials=$(( trial_end - trial_start ))
    (( shard_trials > 0 )) || continue
    shard_log="${TASK_ROOT}/logs/client_${SLURM_JOB_ID:-manual}_shard${i}.log"
    (
      (( STAGGER_SECONDS > 0 && i > 0 )) && sleep $(( i * STAGGER_SECONDS ))
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${ROBOCASA_ENV}"
      export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
      cd "${REPO_ROOT}"
      PYTHONUNBUFFERED=1 python -u examples/RoboCasa/eval/run_robocasa_eval.py \
        --host localhost --port "${PORT}" --model-name "${MODEL_NAME}" \
        --env-name "${ENV_NAME}" --num-trials "${shard_trials}" --max-steps "${MAX_STEPS}" \
        --replan-steps "${REPLAN_STEPS}" --seed "$(( SEED + trial_start ))" \
        --video-every "${VIDEO_EVERY}" --video-out-path "${VIDEO_OUT}/shard_${i}" \
        --summary-out-path "${SHARD_SUMMARY_ROOT}/shard_${i}.json" "${CAMERA_FLAG[@]}"
    ) > "${shard_log}" 2>&1 &
    client_pids+=("$!")
  done
  task_rc=0
  for pid in "${client_pids[@]}"; do wait "${pid}" || task_rc=$?; done
  (( task_rc == 0 )) || { echo "!!! [$ENV_NAME] a client failed (rc=${task_rc})" >&2; overall_rc=${task_rc}; continue; }

  # merge this task's shard summaries
  "${GR00T_ENV}/bin/python" - "${SHARD_SUMMARY_ROOT}" "${SUMMARY_PATH}" <<'PY'
import json, pathlib, sys
shard_dir, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
paths = sorted(shard_dir.glob("shard_*.json"))
if not paths: raise SystemExit(f"No shard summaries in {shard_dir}")
S = [json.loads(p.read_text()) for p in paths]
eps = [e for s in S for e in s.get("episodes", [])]
te = sum(int(s.get("total_episodes", 0)) for s in S)
ts = sum(int(s.get("total_successes", 0)) for s in S)
f = S[0]
out_path.write_text(json.dumps({
    "model_name": f.get("model_name"), "env_name": f.get("env_name"),
    "num_trials": te, "max_steps": f.get("max_steps"), "replan_steps": f.get("replan_steps"),
    "use_camera_params": f.get("use_camera_params"), "total_episodes": te,
    "total_successes": ts, "success_rate": ts/te if te else 0.0, "episodes": eps,
}, indent=2) + "\n")
print(f"  [{f.get('env_name')}] {ts}/{te} = {100*ts/te if te else 0:.1f}%  -> {out_path}")
PY
done

echo "===== EVAL DONE (rc=${overall_rc}) ====="
echo "Summaries under: ${LOG_ROOT}/<task>/summary/"
exit "${overall_rc}"
