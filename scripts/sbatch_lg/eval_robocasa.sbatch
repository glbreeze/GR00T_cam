#!/bin/bash
#SBATCH --job-name=eval_gr00t_robocasa
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/eval_robocasa/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Single-task-family RoboCasa v0.2 eval for one gr00t checkpoint. Spins up the
# gr00t inference server in the `gr00t` conda env and drives rollouts via
# examples/RoboCasa/eval/run_robocasa_eval.py in the `robocasa24` conda env
# (robosuite 1.5.1 + robocasa v0.2 + kitchen assets). Both live on the same node
# and talk over ZMQ on localhost. Counterpart to eval_libero.sbatch.
#
# Required env (set by the dispatcher / caller):
#   CKPT_DIR, MODEL_NAME, ENV_NAME   (ENV_NAME = one RoboCasa task family,
#                                     e.g. PnPCounterToCab / OpenDrawer / ...)
#
# Optional overrides (defaults shown):
#   DATA_CONFIG=robocasa
#   EMBODIMENT_TAG=new_embodiment
#   DENOISING_STEPS=4
#   USE_CAMERA_PARAMS=false      (true for CamVLA geo/distill checkpoints)
#   NUM_TRIALS=50                (rollouts for this task family, sharded across clients)
#   MAX_STEPS=500
#   REPLAN_STEPS=16
#   NUM_PARALLEL_CLIENTS=4       (shard the trials across N clients against one server)
#   VIDEO_EVERY=10               (0 to disable video)
#   SEED=0
#   PORT=19300
#   SERVER_START_TIMEOUT=900
#   ROBOCASA_ENV=/scratch/lg154/conda-envs/robocasa24
#   ROBOCASA_HOME=/scratch/lg154/.cache/robocasa24_home

set -euo pipefail

# ---- required inputs ----
: "${CKPT_DIR:?required: path to a checkpoint dir (must contain config.json)}"
: "${MODEL_NAME:?required: short tag used in output paths}"
: "${ENV_NAME:?required: a RoboCasa task family, e.g. PnPCounterToCab}"

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/lg154/conda-envs/robocasa24}

# ---- defaults ----
DATA_CONFIG=${DATA_CONFIG:-robocasa}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
DENOISING_STEPS=${DENOISING_STEPS:-4}
USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}

NUM_TRIALS=${NUM_TRIALS:-50}
MAX_STEPS=${MAX_STEPS:-500}
REPLAN_STEPS=${REPLAN_STEPS:-16}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
VIDEO_EVERY=${VIDEO_EVERY:-10}
SEED=${SEED:-0}

PORT=${PORT:-19300}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}

# ---- output paths ----
LOG_ROOT=${REPO_ROOT}/logs/eval_robocasa/${MODEL_NAME}/${ENV_NAME}
SUMMARY_PATH=${LOG_ROOT}/summary/${SLURM_JOB_ID:-manual}.json
SERVER_LOG=${LOG_ROOT}/logs/server_${SLURM_JOB_ID:-manual}.log
CLIENT_LOG=${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}.log
VIDEO_OUT=${LOG_ROOT}/videos/${SLURM_JOB_ID:-manual}
SHARD_SUMMARY_ROOT=${LOG_ROOT}/summary/shards_${SLURM_JOB_ID:-manual}

mkdir -p "$(dirname "${SUMMARY_PATH}")" "${VIDEO_OUT}" "$(dirname "${SERVER_LOG}")" \
         "${SHARD_SUMMARY_ROOT}"

# ---- ckpt sanity ----
if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
  echo "Missing checkpoint config: ${CKPT_DIR}/config.json" >&2
  exit 1
fi
if [[ ! -d "${ROBOCASA_ENV}" ]]; then
  echo "Missing robocasa24 conda env at ${ROBOCASA_ENV}." >&2
  exit 1
fi

# ---- per-job scratch + env (prevents concurrent evals from stomping shared state) ----
JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER}/gr00t_rceval_${SLURM_JOB_ID:-$$}
export TMPDIR=${JOB_TMP}/tmp TMP=${JOB_TMP}/tmp TEMP=${JOB_TMP}/tmp
export XDG_CACHE_HOME=${JOB_TMP}/xdg_cache
export XDG_CONFIG_HOME=${JOB_TMP}/xdg_config
export XDG_STATE_HOME=${JOB_TMP}/xdg_state
export MPLCONFIGDIR=${JOB_TMP}/mplconfig
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}"

# Headless rendering on compute nodes (no X server).
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
# RoboCasa writes some state under $HOME; keep it on a stable per-user cache.
export ROBOCASA_HOME=${ROBOCASA_HOME:-/scratch/lg154/.cache/robocasa24_home}
mkdir -p "${ROBOCASA_HOME}"

echo "===== ROBOCASA EVAL ENV ====="
echo "host:     $(hostname)"
nvidia-smi -L 2>/dev/null || true
echo "ckpt:     ${CKPT_DIR}"
echo "config:   ${DATA_CONFIG}    tag: ${EMBODIMENT_TAG}"
echo "task:     ${ENV_NAME}    trials: ${NUM_TRIALS}  max_steps=${MAX_STEPS}  replan=${REPLAN_STEPS}"
echo "camera:   USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS}"
echo "clients:  ${NUM_PARALLEL_CLIENTS}"
echo "summary:  ${SUMMARY_PATH}"
echo "============================="

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

# -------- 2. run the RoboCasa clients (shard the trials across clients) --------
CAMERA_FLAG=()
if [[ "${USE_CAMERA_PARAMS}" == "true" ]]; then
  CAMERA_FLAG+=("--use-camera-params")
fi

n_clients=${NUM_PARALLEL_CLIENTS}
(( n_clients > NUM_TRIALS )) && n_clients=${NUM_TRIALS}
echo "Evaluating ${ENV_NAME}: ${NUM_TRIALS} trials across ${n_clients} client(s)."

client_pids=()
for i in $(seq 0 $((n_clients - 1))); do
  trial_start=$(( i * NUM_TRIALS / n_clients ))
  trial_end=$(( (i + 1) * NUM_TRIALS / n_clients ))
  shard_trials=$(( trial_end - trial_start ))
  (( shard_trials > 0 )) || continue

  shard_video="${VIDEO_OUT}/shard_${i}"
  shard_summary="${SHARD_SUMMARY_ROOT}/shard_${i}.json"
  shard_log="${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}_shard${i}.log"
  mkdir -p "${shard_video}"

  echo "Client ${i}: ${shard_trials} trials (seed offset ${trial_start})  (log: ${shard_log})"
  (
    source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
    conda activate "${ROBOCASA_ENV}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${REPO_ROOT}"
    PYTHONUNBUFFERED=1 python -u examples/RoboCasa/eval/run_robocasa_eval.py \
      --host localhost \
      --port "${PORT}" \
      --model-name "${MODEL_NAME}" \
      --env-name "${ENV_NAME}" \
      --num-trials "${shard_trials}" \
      --max-steps "${MAX_STEPS}" \
      --replan-steps "${REPLAN_STEPS}" \
      --seed "$(( SEED + trial_start ))" \
      --video-every "${VIDEO_EVERY}" \
      --video-out-path "${shard_video}" \
      --summary-out-path "${shard_summary}" \
      "${CAMERA_FLAG[@]}"
  ) > "${shard_log}" 2>&1 &
  client_pids+=("$!")
done

client_rc=0
for pid in "${client_pids[@]}"; do
  wait "${pid}" || client_rc=$?
done

cat "${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}_shard"*.log > "${CLIENT_LOG}" 2>/dev/null || true
(( client_rc == 0 )) || { echo "At least one client failed (rc=${client_rc})" >&2; exit "${client_rc}"; }

# -------- 3. merge shard summaries -> SUMMARY_PATH --------
/scratch/lg154/conda-envs/gr00t/bin/python - "${SHARD_SUMMARY_ROOT}" "${SUMMARY_PATH}" <<'PY'
import json, pathlib, sys

shard_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
paths = sorted(shard_dir.glob("shard_*.json"))
if not paths:
    raise SystemExit(f"No shard summaries in {shard_dir}")

summaries = [json.loads(p.read_text()) for p in paths]
episodes = [e for s in summaries for e in s.get("episodes", [])]
total_eps = sum(int(s.get("total_episodes", 0)) for s in summaries)
total_succ = sum(int(s.get("total_successes", 0)) for s in summaries)
first = summaries[0]

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "model_name": first.get("model_name"),
    "env_name": first.get("env_name"),
    "num_trials": total_eps,
    "max_steps": first.get("max_steps"),
    "replan_steps": first.get("replan_steps"),
    "use_camera_params": first.get("use_camera_params"),
    "total_episodes": total_eps,
    "total_successes": total_succ,
    "success_rate": total_succ / total_eps if total_eps else 0.0,
    "episodes": episodes,
    "shard_summaries": [str(p) for p in paths],
}, indent=2) + "\n")
print(out_path)
PY

echo "Eval complete."
echo "Summary: ${SUMMARY_PATH}"
echo "Videos:  ${VIDEO_OUT}"
