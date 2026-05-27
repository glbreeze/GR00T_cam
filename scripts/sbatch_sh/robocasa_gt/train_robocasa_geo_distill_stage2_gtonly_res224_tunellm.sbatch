#!/bin/bash
# LAUNCH WITH `bash <this-script>` (not `sbatch`): the self-submit guard below
# auto-creates the --output/--error dir and re-submits via sbatch (SLURM opens
# the log files before the body runs, so the dir must pre-exist).
#SBATCH --job-name=gr00t_robocasa_geo_distill_stage2_gtonly_res224_tunellm
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_stage2_gtonly_res224_tunellm/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_stage2_gtonly_res224_tunellm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=sfscai
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:a800:2
#SBATCH --export=ALL

# ============================================================================
# GR00T N1.5 RoboCasa -- STAGE 2 (action-focused fine-tune) + LLM UNFROZEN.
# NYU Shanghai (sfscai/A800) self-contained port of the NY recipe
#   scripts/sbatch_lg/train_robocasa_geo_distill_stage2_gtonly_res224.sbatch
# with the ONE experimental delta: --tune-llm (the LLM backbone is unfrozen).
#
# WHY TWO-STAGE + tune-llm (vs the single-stage tunellm run):
# Single-stage tune-llm jointly optimises ZERO-INIT geometry modules AND the
# pretrained LLM from step 0 -> the action head chases two moving targets at
# once (geometry perturbing tokens + LLM weights shifting), giving noisy, slow
# action_loss descent (~0.06 at 10k vs frozen control ~0.025). Stage 1 first
# WARM-STARTS the geometry (LLM frozen) so it produces stable, meaningful
# features; stage 2 then unfreezes the LLM, which adapts to a STABLE geometry
# signal instead of co-adapting with random-init geometry. Removes one moving
# target -> expected to be much smoother. (Compare on RoboCasa EVAL success, not
# just train loss: the unfrozen LLM adds ~1.2B trainable params so lower train
# action_loss is partly just capacity / overfitting risk.)
#
# Recipe = identical to the FROZEN stage-2 (lr 1e-4 cosine, action_loss_weight
# 1.0, pi3x_loss_weight 0.05, loads stage-1 ckpt, NO --trainable-prefixes) EXCEPT:
#   + --tune-llm --llm-learning-rate 2e-5  (LLM @ gentle per-group LR; rest @1e-4)
#   + batch 8 x accum 2 (eff. 32, == frozen stage-2's 16x1) for memory headroom
#     with the LLM unfrozen (frozen stage-2 used 16x1; unfrozen needs more VRAM)
#   + NO --gradient-checkpointing (reentrant ckpt conflicts with the DDP
#     static_graph the unfrozen LLM needs; A800-80GB fits without it). The
#     static_graph fix in DualBrainTrainer._wrap_model auto-triggers on tune_llm.
#
# sfscai deltas vs NY: account/partition sfscai, --gres=gpu:a800:2, conda base
# /gpfsnyu/home/lg154/miniconda3 + env /scratch/lg154/conda_envs/gr00t, 16 CPU /
# 128G (lets this run CONCURRENTLY with the single-stage tunellm run within the
# cpu=64/gpu=8 per-user QOS cap), wandb offline + HF offline. Walltime 12h (run
# ~8h; short walltime backfills past the A800 queue).

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
EXP_NAME=gr00t_robocasa_geo_distill_stage2_gtonly_res224_tunellm

# --- self-submit guard: create the SLURM log dir, then (re)submit ---
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${REPO_ROOT}/logs/${EXP_NAME}"
  echo "[submit] ensured ${REPO_ROOT}/logs/${EXP_NAME} ; submitting via sbatch ..."
  exec sbatch "$0" "$@"
fi

# --- Stage-1 checkpoint to seed from (rsync'd from DTN; geometry warm-started) ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_robocasa_geo_distill_s1_gtonly_res224/checkpoint-5000}

# --- data / model ---
DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
DATA_CONFIG=robocasa
VIDEO_BACKEND=decord
EMBODIMENT_TAG=new_embodiment
BASE_MODEL_PATH="${STAGE1_CKPT}"     # start from the stage-1 (warm geometry) ckpt
OUTPUT_DIR="${REPO_ROOT}/checkpoints/${EXP_NAME}"

# --- ground-truth pointmap cache (same as stage 1) ---
GT_POINT_TARGETS_ROOT="/scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware"

# --- compute / scale ---
NUM_GPUS=${NUM_GPUS:-2}
NUM_WORKERS=${NUM_WORKERS:-8}   # 2 GPU x 8 = 16 dataloader workers = cpus-per-task
BATCH_SIZE=${BATCH_SIZE:-8}     # 8/GPU x 2 GPU x accum 2 = eff. batch 32
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
RESUME=${RESUME:-false}

mkdir -p "${REPO_ROOT}/logs/${EXP_NAME}" "${OUTPUT_DIR}"

# --- per-job scratch (isolated TMP / XDG) ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/gr00t_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${JOB_TMP_ROOT}/tmp" TEMP="${JOB_TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

# --- python env (SH conda) + HF cache on scratch, OFFLINE (no network on nodes) ---
source /gpfsnyu/home/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda_envs/gr00t
cd "${REPO_ROOT}"

export HF_HOME=${HF_HOME:-/scratch/lg154/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- compute / runtime knobs ---
THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-${NUM_GPUS}} / NUM_GPUS ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( THREADS_PER_PROC > 0 ? THREADS_PER_PROC : 1 ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- wandb (OFFLINE; sync from a login node after the run, see footer) ---
WANDB_FLAG=()
if [[ "${WANDB_ENABLED:-true}" == "true" ]]; then
  [[ -n "${WANDB_API_KEY_OVERRIDE:-}" ]] && export WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
  export WANDB_MODE=${WANDB_MODE:-offline}
  if [[ "${WANDB_MODE}" == "offline" ]]; then
    export WANDB_DIR="${REPO_ROOT}/logs/${EXP_NAME}/wandb"   # persistent: survives for `wandb sync`
  else
    export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  fi
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  export WANDB_PROJECT=${WANDB_PROJECT:-gr00t_robocasa_lg}
  export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
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
[[ -d "${GT_POINT_TARGETS_ROOT}/base" && -d "${GT_POINT_TARGETS_ROOT}/wrist" ]] \
  || { echo "Missing GT point cache (need base/ + wrist/): ${GT_POINT_TARGETS_ROOT}" >&2; exit 1; }
# Stage-1 checkpoint must be a real HF checkpoint (rsync it from DTN first).
{ [[ -f "${STAGE1_CKPT}/config.json" ]] \
  && { [[ -f "${STAGE1_CKPT}/model.safetensors.index.json" ]] || [[ -f "${STAGE1_CKPT}/model.safetensors" ]]; }; } \
  || { echo "STAGE1_CKPT is not a HF checkpoint (need config.json + model.safetensors[.index.json]):" >&2
       echo "  ${STAGE1_CKPT}" >&2
       echo "rsync it from DTN first (see the stage-2 setup notes)." >&2; exit 1; }

# --- assemble training argv: stage-2 recipe + LLM UNFROZEN ---
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
  # --- LLM unfreeze (THE experiment) + memory knobs; no grad checkpointing ---
  --tune-llm
  --learning-rate 1e-4
  --llm-learning-rate 2e-5
  --gradient-accumulation-steps 2
  # --- CamVLA geometry stack (must match stage 1) ---
  --use-camvla-model
  --use-ray-embed
  --cross-view-type standard
  --pose-enc-type prope
  --cross-view-aa-order fg
  --cross-view-prope-layer-idx 0
  --cross-view-rope-freq 100.0
  --use-camera-params
  --disable-geometric-augs
  # --- stage-2 GT point-head supervision @224 (action dominates) ---
  --gt-point-root "${GT_POINT_TARGETS_ROOT}"
  --pi3x-cam-subdirs base wrist
  --pi3x-loss-type legacy_conf_mse
  --point-head-output-resolution 224
  --action-loss-weight 1.0
  --pi3x-loss-weight 0.05
)
[[ "${RESUME}" == "true" ]] && TRAIN_ARGS+=(--resume)

echo "===== STAGE-2 + tune-llm ENV CHECK (Shanghai) ====="
echo "exp:         ${EXP_NAME}"
echo "stage1 ckpt: ${STAGE1_CKPT}"
echo "dataset:     ${DATASET_PATH}"
echo "gt cache:    ${GT_POINT_TARGETS_ROOT}"
echo "output:      ${OUTPUT_DIR}"
echo "wandb:       ${WANDB_FLAG[*]} mode=${WANDB_MODE:-<unset>} (project=${WANDB_PROJECT:-<none>} name=${WANDB_NAME:-<auto>})"
[[ "${WANDB_MODE:-}" == "offline" ]] && \
  echo "wandb sync:  after the run, from a LOGIN node: wandb sync ${WANDB_DIR}/wandb/offline-run-*"
echo "gpus=${NUM_GPUS} bs=${BATCH_SIZE} accum=2 (eff 32) steps=${NUM_TRAIN_STEPS} workers=${NUM_WORKERS}"
which python; python -V; hostname; nvidia-smi -L || true
echo "==================================================="

exec python scripts/gr00t_finetune.py "${TRAIN_ARGS[@]}"
