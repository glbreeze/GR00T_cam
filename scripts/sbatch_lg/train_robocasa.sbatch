#!/bin/bash
# Inner runner for GR00T N1.5 RoboCasa fine-tuning. Mirrors train_libero.sbatch:
# must be invoked from a wrapper sbatch via `exec bash` (it assumes the job is
# already inside a slurm allocation). Not standalone-submittable.
#
# Required from the wrapper:
#   REPO_ROOT, EXP_NAME, DATASET_PATH
# Optional (with defaults shown below):
#   DATA_CONFIG, VIDEO_BACKEND, EMBODIMENT_TAG,
#   NUM_GPUS, NUM_WORKERS, BATCH_SIZE, NUM_TRAIN_STEPS, SAVE_INTERVAL,
#   BASE_MODEL_PATH, OUTPUT_BASE_DIR, RESUME,
#   WANDB_ENABLED, WANDB_PROJECT, WANDB_API_KEY_OVERRIDE,
#   EXTRA_TRAIN_ARGS  (space-separated extra flags forwarded to gr00t_finetune.py).

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set by the wrapper}"
: "${EXP_NAME:?}" "${DATASET_PATH:?}"

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
DATA_CONFIG=${DATA_CONFIG:-robocasa}
VIDEO_BACKEND=${VIDEO_BACKEND:-decord}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-16}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-nvidia/GR00T-N1.5-3B}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-${REPO_ROOT}/checkpoints}
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}"

# --- per-job scratch (isolated TMP / HOME / XDG so concurrent jobs don't collide) ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/gr00t_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${JOB_TMP_ROOT}/tmp" TEMP="${JOB_TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

# --- python env + persistent HF cache ---
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/gr00t
cd "${REPO_ROOT}"

export HF_HOME=${HF_HOME:-/scratch/lg154/.huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
unset TRANSFORMERS_CACHE

# --- compute / runtime knobs ---
THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-${NUM_GPUS}} / NUM_GPUS ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( THREADS_PER_PROC > 0 ? THREADS_PER_PROC : 1 ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- wandb (per-job dirs so concurrent runs don't collide) ---
WANDB_FLAG=()
if [[ "${WANDB_ENABLED:-true}" == "true" ]]; then
  [[ -n "${WANDB_API_KEY_OVERRIDE:-}" ]] && export WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
  export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  export WANDB_PROJECT=${WANDB_PROJECT:-gr00t_robocasa_lg}
  export WANDB_NAME=${WANDB_NAME:-${EXP_NAME}}
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
  WANDB_FLAG=(--report-to wandb)
else
  export WANDB_MODE=disabled
  WANDB_FLAG=(--report-to tensorboard)
fi

# --- pre-flight: required artifacts exist on disk ---
[[ -f "${DATASET_PATH}/meta/info.json" ]] \
  || { echo "Missing LeRobot dataset: ${DATASET_PATH}" >&2; exit 1; }

if [[ "${EXTRA_TRAIN_ARGS:-}" == *--pi3x-root* ]]; then
  pi3x_path=$(echo "${EXTRA_TRAIN_ARGS}" | sed -nE 's/.*--pi3x-root[ =]([^ ]+).*/\1/p')
  [[ -d "${pi3x_path}" ]] \
    || { echo "Missing pi3x targets root: ${pi3x_path}" >&2; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}"

# --- assemble training argv ---
TRAIN_ARGS=(
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --data-config "${DATA_CONFIG}"
  --video-backend "${VIDEO_BACKEND}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --base-model-path "${BASE_MODEL_PATH}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${BATCH_SIZE}"
  --dataloader-num-workers "${NUM_WORKERS}"
  --max-steps "${NUM_TRAIN_STEPS}"
  --save-steps "${SAVE_INTERVAL}"
  "${WANDB_FLAG[@]}"
)
if [[ "${RESUME:-false}" == "true" ]]; then
  TRAIN_ARGS+=(--resume)
fi
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a _extra <<< "${EXTRA_TRAIN_ARGS}"
  TRAIN_ARGS+=("${_extra[@]}")
fi

echo "===== TRAIN ENV CHECK ====="
echo "exp:         ${EXP_NAME}"
echo "dataset:     ${DATASET_PATH}"
echo "data_config: ${DATA_CONFIG}"
echo "base_model:  ${BASE_MODEL_PATH}"
echo "output:      ${OUTPUT_DIR}"
echo "wandb:       ${WANDB_FLAG[*]} (project=${WANDB_PROJECT:-<none>} name=${WANDB_NAME:-<auto>})"
echo "gpus=${NUM_GPUS} bs=${BATCH_SIZE} steps=${NUM_TRAIN_STEPS} workers=${NUM_WORKERS}"
echo "extra:       ${EXTRA_TRAIN_ARGS:-<none>}"
which python; python -V; hostname; nvidia-smi -L || true

exec python scripts/gr00t_finetune.py "${TRAIN_ARGS[@]}"
