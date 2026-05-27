#!/bin/bash
# LAUNCH WITH `bash <this-script>` (not `sbatch <this-script>`): the self-submit
# guard near the top auto-creates the --output/--error dir and re-submits via
# sbatch. This is required because SLURM opens the log files BEFORE the script
# body runs, so the dir must exist pre-submission -- a plain in-body `mkdir` is
# too late (job would die at t~1s with no log). Direct `sbatch` also works once
# the dir exists from a prior run.
#SBATCH --job-name=gr00t_robocasa_geo_distill_gtonly_res224_tunellm
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly_res224_tunellm/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly_res224_tunellm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --account=sfscai
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:a800:2
#SBATCH --export=ALL

# ============================================================================
# NYU SHANGHAI HPC port of scripts/sbatch_lg/robocasa_gt/
#   train_robocasa_geo_distill_gtonly_res224_tunellm.sbatch (NY cluster).
# Unlike the NY wrapper (which execs sbatch_lg/train_robocasa.sbatch), this is
# SELF-CONTAINED: the NY driver hardcodes NY conda paths, and the SH convention
# (cf. sbatch_sh/eval_robocasa.sbatch) is one standalone script. Deltas vs NY:
#   account/partition  sfscai (was torch_pr_69_tandon_advanced / a100_tandon)
#   gres               gpu:a800:2 -- pin to the A800-80GB nodes (gpu187/188);
#                      only those carry the `a800` gres type on sfscai.
#   conda base         /gpfsnyu/home/lg154/miniconda3 (was /scratch/lg154/miniconda3)
#   conda env          /scratch/lg154/conda_envs/gr00t  (underscore, not -)
#   cpus/mem           16 CPU + 128G. sfscai QOS caps the user at cpu=64 /
#                      gres/gpu=8, and each job takes 2 GPUs -> 16 CPU/job lets
#                      4 such jobs run concurrently (4x16=64 CPU, 4x2=8 GPU, both
#                      at cap); 32 CPU/job would cap concurrency at 2. NUM_WORKERS
#                      is lowered to 8 to match (2 GPU x 8 = 16 = the CPU alloc).
#                      A800 nodes have ~515G but are shared; 128G is ample and
#                      schedules faster than NY's 256G. Walltime cap is 5 days.
#
# HYPOTHESIS TEST: does UNFREEZING THE LLM let the geometry signal help?
# ----------------------------------------------------------------------------
# In the frozen-backbone setup (tune_llm=tune_visual=False), the trainable
# geometry modules (ray_embed + cross_view_fusion) perturb the visual tokens,
# but those perturbed tokens are then consumed by a FROZEN LLM whose weights
# were tuned for the *original* token statistics. The cheapest thing the
# optimizer can do is suppress the geometry modules (keep LayerScale ~0) so the
# frozen LLM sees baseline-like tokens -> geometry ~= baseline. This run
# unfreezes the LLM so it can ADAPT to the geometry-perturbed tokens, testing
# whether that unlocks a real gain.
#
# Deltas vs the FROZEN control (everything else identical):
#   1. --tune-llm                : unfreeze the LLM backbone (THE experiment).
#   2. PER-GROUP LR: --learning-rate 1e-4 keeps the PROVEN action-head/geometry/
#      projector LR, while --llm-learning-rate 2e-5 gives the freshly-unfrozen
#      pretrained LLM a gentler LR so it adapts without early instability /
#      catastrophic forgetting. Both groups share the cosine schedule + warmup;
#      only the peak LR differs. (DualBrainTrainer.create_optimizer splits the
#      LLM params into their own group; --llm-learning-rate unset => single LR.)
#   3. NO --gradient-checkpointing (differs from NY): reentrant grad checkpointing
#      is INCOMPATIBLE with DDP `find_unused_parameters=True`, which the unfrozen
#      LLM requires (its unused lm_head/upper layers have no grad) -- the combo
#      throws "Expected to mark a variable ready only once ... reused parameters
#      in multiple reentrant backward passes". A800 is 80GB so the LLM-unfrozen
#      3B (~38GB params+grads+optim + activations) fits WITHOUT checkpointing,
#      which also runs ~20-30% faster. (If a future smaller-GPU run OOMs, re-add
#      checkpointing AND set gradient_checkpointing_kwargs={"use_reentrant":False}
#      in gr00t_finetune.py's TrainingArguments -- non-reentrant is DDP-safe.)
#   4. BATCH_SIZE 8 + accum 2    : effective batch 16 at half the per-microbatch
#      activation memory (extra A800 headroom).
# Vision tower stays FROZEN (tune_visual unset) to isolate the LLM effect.
#
# sfscai walltime cap is 5 days (>> our 20h); SAVE_INTERVAL=1000 keeps
# intermediate checkpoints eval-able if the job is cut short -- resubmit with
# RESUME=true to continue.

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
EXP_NAME=gr00t_robocasa_geo_distill_gtonly_res224_tunellm

# --- self-submit guard: create the SLURM log dir, then (re)submit ---
# When launched directly (`bash <this>`), SLURM_JOB_ID is unset: make the
# --output/--error dir (sbatch opens those before the job body runs, so it must
# pre-exist) and exec sbatch on ourselves. Inside the real job (SLURM_JOB_ID set)
# this is skipped and training proceeds. No more manual mkdir before submitting.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${REPO_ROOT}/logs/${EXP_NAME}"
  echo "[submit] ensured ${REPO_ROOT}/logs/${EXP_NAME} ; submitting via sbatch ..."
  exec sbatch "$0" "$@"
fi

# --- data / model ---
DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
DATA_CONFIG=robocasa
VIDEO_BACKEND=decord
EMBODIMENT_TAG=new_embodiment
BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B
OUTPUT_DIR="${REPO_ROOT}/checkpoints/${EXP_NAME}"

# --- ground-truth pointmap cache (uncompressed .npy, base/ + wrist/ subdirs) ---
# Same path layout as NY now (moved here out of the repo working tree).
GT_POINT_TARGETS_ROOT="/scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware"

# --- compute / scale ---
NUM_GPUS=${NUM_GPUS:-2}
NUM_WORKERS=${NUM_WORKERS:-8}   # 2 GPU x 8 = 16 dataloader workers = cpus-per-task
BATCH_SIZE=${BATCH_SIZE:-8}          # 8/GPU; accum 2 below restores eff. batch 16
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
RESUME=${RESUME:-false}

mkdir -p "${REPO_ROOT}/logs/${EXP_NAME}" "${OUTPUT_DIR}"

# --- per-job scratch (isolated TMP / XDG so concurrent jobs don't collide) ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/gr00t_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${JOB_TMP_ROOT}/tmp" TEMP="${JOB_TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

# --- python env (SH conda) + persistent HF cache on scratch (home quota is tight) ---
source /gpfsnyu/home/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda_envs/gr00t
cd "${REPO_ROOT}"

export HF_HOME=${HF_HOME:-/scratch/lg154/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
unset TRANSFORMERS_CACHE
# sfscai COMPUTE NODES HAVE NO OUTBOUND NETWORK -> resolve the base model from
# the local cache only; a network HEAD request would otherwise hang/fail. The
# model MUST be pre-downloaded on a login node first (the preflight below
# enforces this). Set HF_HUB_OFFLINE=0 at submit time if your node has network.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- compute / runtime knobs ---
THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-${NUM_GPUS}} / NUM_GPUS ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( THREADS_PER_PROC > 0 ? THREADS_PER_PROC : 1 ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- wandb ---
# Compute nodes have no network, so log OFFLINE and `wandb sync` from a login
# node after the run (command printed in the footer). CRITICAL: in offline mode
# WANDB_DIR must be PERSISTENT scratch -- the per-job ${JOB_TMP_ROOT} is wiped at
# job end, which would destroy the unsynced offline run. The cache/config/data
# dirs stay in job tmp (regenerable). Set WANDB_MODE=online at submit time if the
# node has network.
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
# Base model must already be in the HF cache (no network on compute nodes).
if [[ "${BASE_MODEL_PATH}" != /* ]]; then
  _hf_repo="models--${BASE_MODEL_PATH//\//--}"
  [[ -n "$(find "${HF_HUB_CACHE}/${_hf_repo}/snapshots" -name '*.safetensors' 2>/dev/null | head -1)" ]] \
    || { echo "Base model '${BASE_MODEL_PATH}' not in HF cache (${HF_HUB_CACHE}). With" >&2
         echo "no network on compute nodes, pre-download on a LOGIN node first:" >&2
         echo "  HF_HOME=${HF_HOME} huggingface-cli download ${BASE_MODEL_PATH}" >&2
         exit 1; }
fi

# --- assemble training argv: CamVLA stack + single-stage GT supervision @224, LLM UNFROZEN ---
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
  # --- LLM unfreeze (THE experiment) + memory knobs ---
  # NOTE: no --gradient-checkpointing -- reentrant checkpointing conflicts with
  # the DDP find_unused_parameters the unfrozen LLM needs (see header). A800-80GB
  # fits without it.
  --tune-llm
  --learning-rate 1e-4
  --llm-learning-rate 2e-5
  --gradient-accumulation-steps 2
  # --- CamVLA geometry stack ---
  --use-camvla-model
  --use-ray-embed
  --cross-view-type standard
  --pose-enc-type prope
  --cross-view-aa-order fg
  --cross-view-prope-layer-idx 0
  --cross-view-rope-freq 100.0
  --use-camera-params
  --disable-geometric-augs
  # --- ground-truth point-head supervision @224 ---
  --gt-point-root "${GT_POINT_TARGETS_ROOT}"
  --pi3x-cam-subdirs base wrist
  --pi3x-loss-type legacy_conf_mse
  --pi3x-loss-weight 1.0
  --point-head-output-resolution 224
)
[[ "${RESUME}" == "true" ]] && TRAIN_ARGS+=(--resume)

echo "===== TRAIN ENV CHECK (Shanghai) ====="
echo "exp:         ${EXP_NAME}"
echo "dataset:     ${DATASET_PATH}"
echo "gt cache:    ${GT_POINT_TARGETS_ROOT}"
echo "base_model:  ${BASE_MODEL_PATH}"
echo "output:      ${OUTPUT_DIR}"
echo "wandb:       ${WANDB_FLAG[*]} mode=${WANDB_MODE:-<unset>} (project=${WANDB_PROJECT:-<none>} name=${WANDB_NAME:-<auto>})"
[[ "${WANDB_MODE:-}" == "offline" ]] && \
  echo "wandb sync:  after the run, from a LOGIN node: wandb sync ${WANDB_DIR}/wandb/offline-run-*"
echo "hf_offline:  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} (base model from cache ${HF_HUB_CACHE})"
echo "gpus=${NUM_GPUS} bs=${BATCH_SIZE} accum=2 steps=${NUM_TRAIN_STEPS} workers=${NUM_WORKERS}"
which python; python -V; hostname; nvidia-smi -L || true
echo "======================================"

exec python scripts/gr00t_finetune.py "${TRAIN_ARGS[@]}"
