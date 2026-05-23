#!/usr/bin/env bash
# Fan out a 4-suite LIBERO eval (spatial, object, goal, 10) for ONE gr00t
# checkpoint. Submits 4 sbatch jobs sharing scripts/sbatch_lg/eval_libero.sbatch
# — each suite gets its own GPU and its own JSON summary.
#
# Usage (defaults run the baseline checkpoint at step 30000):
#   bash scripts/sbatch_lg/submit_eval_libero_4suites.sh
#
# Override per-run by exporting env vars first. The full set of vars the
# child sbatch honors:
#
#   CKPT_DIR              full path to checkpoint dir (must contain config.json)
#   MODEL_NAME            short tag used in output paths (logs/eval_libero/$MODEL_NAME/...)
#   DATA_CONFIG           data config string for inference_service.py
#   EMBODIMENT_TAG        usually "new_embodiment" for LIBERO finetunes
#   USE_CAMERA_PARAMS     "true" for CamVLA modes 3/4, else "false"
#   NUM_TRIALS_PER_TASK   default 50
#   REPLAN_STEPS          default 5
#   DENOISING_STEPS       default 4
#   PORT                  default 19200 (per-job ports rotate via $SLURM_JOB_ID would be safer but
#                         only one server runs per job, so a fixed port is fine)
#
# Examples:
#   # Baseline at step 20k:
#   CKPT_DIR=$REPO_ROOT/checkpoints/gr00t_libero_baseline_2gpu_b16/checkpoint-20000 \
#     MODEL_NAME=baseline_20k \
#     bash scripts/sbatch_lg/submit_eval_libero_4suites.sh
#
#   # Geo+distill stage2 at 30k, camera-aware path:
#   CKPT_DIR=$REPO_ROOT/checkpoints/gr00t_libero_geo_distill_s2_2gpu_b16/checkpoint-30000 \
#     MODEL_NAME=geo_distill_s2_30k \
#     USE_CAMERA_PARAMS=true \
#     bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
SBATCH_SCRIPT=${REPO_ROOT}/scripts/sbatch_lg/eval_libero.sbatch

# ---- defaults: baseline checkpoint at step 30000 ----
export CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/gr00t_libero_baseline_2gpu_b16/checkpoint-30000}
export MODEL_NAME=${MODEL_NAME:-baseline_30k}
export USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}
# DATA_CONFIG / EMBODIMENT_TAG / DENOISING_STEPS / NUM_TRIALS_PER_TASK / REPLAN_STEPS
# are passed through to the child sbatch if set; otherwise the child uses its
# defaults.
# -----------------------------------------------------

mkdir -p "${REPO_ROOT}/logs/eval_libero"

if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
  echo "Checkpoint not found: ${CKPT_DIR}/config.json" >&2
  exit 1
fi

echo "Submitting 4-suite eval for:"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS}"

SUITES=(libero_spatial libero_object libero_goal libero_10)
for SUITE in "${SUITES[@]}"; do
  export SUITE_NAME=${SUITE}
  sbatch \
    --job-name="eval_${MODEL_NAME}_${SUITE}" \
    "${SBATCH_SCRIPT}"
done
