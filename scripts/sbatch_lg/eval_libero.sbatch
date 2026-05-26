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

# Fan out this many rollout clients (disjoint task ranges) against the single
# policy server. LIBERO eval is sim/render-bound, so one client leaves the GPU
# ~1% utilised and trips the partition's >2h low-util killer. Parallel clients
# overlap their mujoco sims on CPU + EGL render contexts on the GPU, cutting
# wall-clock ~Nx (a 10-task suite finishes well under the 2h window) and lifting
# average GPU util. The zmq.REP server fair-queues across clients (serialised
# inference, parallel sims). Set NUM_PARALLEL_CLIENTS=1 for the old behavior.
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}

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
# Per-shard scratch for the parallel clients; merged back into SUMMARY_PATH.
SHARD_SUMMARY_ROOT=${LOG_ROOT}/summary/shards_${SLURM_JOB_ID:-manual}

mkdir -p "$(dirname "${SUMMARY_PATH}")" "${VIDEO_OUT}" "$(dirname "${SERVER_LOG}")" \
         "${SHARD_SUMMARY_ROOT}"

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

# -------- 2. run the LIBERO clients (sharded across the suite's tasks) --------
CAMERA_FLAG=()
if [[ "${USE_CAMERA_PARAMS}" == "true" ]]; then
  CAMERA_FLAG+=("--use-camera-params")
fi

# Resolve the task window: TASK_ID_END=-1 means "all tasks in the suite". We
# query the benchmark for n_tasks and shard [TASK_ID_START, end) across clients.
n_tasks=$(
  export PYTHONPATH="${REPO_ROOT}:${OPENPI_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
  "${OPENPI_ROOT}/.venv/bin/python" -c "
import contextlib, io
from libero.libero import benchmark
with contextlib.redirect_stdout(io.StringIO()):
    n = benchmark.get_benchmark_dict()['${SUITE_NAME}']().n_tasks
print(n)
"
)
[[ "${n_tasks}" =~ ^[0-9]+$ && "${n_tasks}" -gt 0 ]] \
  || { echo "Bad task count for ${SUITE_NAME}: ${n_tasks}" >&2; exit 1; }

range_start=${TASK_ID_START}
range_end=${TASK_ID_END}
(( range_end < 0 )) && range_end=${n_tasks}
(( range_end > n_tasks )) && range_end=${n_tasks}
span=$(( range_end - range_start ))
(( span > 0 )) || { echo "Empty task range [${range_start}, ${range_end})" >&2; exit 1; }

n_clients=${NUM_PARALLEL_CLIENTS}
(( n_clients > span )) && n_clients=${span}
echo "Suite ${SUITE_NAME} has ${n_tasks} tasks; evaluating [${range_start}, ${range_end}) across ${n_clients} client(s)."

client_pids=()
for i in $(seq 0 $((n_clients - 1))); do
  task_start=$(( range_start + i * span / n_clients ))
  task_end=$(( range_start + (i + 1) * span / n_clients ))
  (( task_start >= task_end )) && continue

  shard_video="${VIDEO_OUT}/shard_${i}"
  shard_summary="${SHARD_SUMMARY_ROOT}/shard_${i}.json"
  shard_log="${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}_shard${i}.log"
  mkdir -p "${shard_video}"

  echo "Client ${i}: tasks [${task_start}, ${task_end})  (log: ${shard_log})"
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
      --task-id-start "${task_start}" \
      --task-id-end "${task_end}" \
      --seed "${SEED}" \
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

# Concatenate shard logs into the canonical client log for convenience.
cat "${LOG_ROOT}/logs/client_${SLURM_JOB_ID:-manual}_shard"*.log > "${CLIENT_LOG}" 2>/dev/null || true

(( client_rc == 0 )) || { echo "At least one client failed (rc=${client_rc})" >&2; exit "${client_rc}"; }

# -------- 3. merge shard summaries -> SUMMARY_PATH --------
"${OPENPI_ROOT}/.venv/bin/python" - "${SHARD_SUMMARY_ROOT}" "${SUMMARY_PATH}" <<'PY'
import json, pathlib, sys

shard_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
paths = sorted(shard_dir.glob("shard_*.json"))
if not paths:
    raise SystemExit(f"No shard summaries in {shard_dir}")

summaries = [json.loads(p.read_text()) for p in paths]
records = [r for s in summaries for r in s.get("records", [])]
total_eps = sum(int(s.get("total_episodes", 0)) for s in summaries)
total_succ = sum(int(s.get("total_successes", 0)) for s in summaries)
first = summaries[0]

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "model_name": first.get("model_name"),
    "task_suite_name": first.get("task_suite_name"),
    "task_range": [
        min(s["task_range"][0] for s in summaries),
        max(s["task_range"][1] for s in summaries),
    ],
    "num_trials_per_task": first.get("num_trials_per_task"),
    "use_camera_params": first.get("use_camera_params"),
    "replan_steps": first.get("replan_steps"),
    "total_episodes": total_eps,
    "total_successes": total_succ,
    "total_success_rate": total_succ / total_eps if total_eps else 0.0,
    "records": sorted(records, key=lambda r: r.get("task_id", 0)),
    "shard_summaries": [str(p) for p in paths],
}, indent=2) + "\n")
print(out_path)
PY

echo "Eval complete."
echo "Summary: ${SUMMARY_PATH}"
echo "Videos:  ${VIDEO_OUT}"
