#!/usr/bin/env bash
# Fan out a RoboCasa v0.2 eval for ONE gr00t checkpoint: submit one sbatch job
# per task family, each sharing scripts/sbatch_lg/eval_robocasa.sbatch (its own
# GPU, its own JSON summary). Counterpart to submit_eval_libero_4suites.sh.
#
# Usage (defaults: SMOKE subset of 3 task families, 10 trials each, on the
#                  robocasa baseline checkpoint at step 30000):
#   bash scripts/sbatch_lg/submit_eval_robocasa.sh
#
# Run the full 24-task benchmark (50 trials each):
#   TASK_SET=all24 NUM_TRIALS=50 \
#     bash scripts/sbatch_lg/submit_eval_robocasa.sh
#
# Override per-run by exporting env vars first. The child sbatch honors:
#   CKPT_DIR MODEL_NAME DATA_CONFIG EMBODIMENT_TAG USE_CAMERA_PARAMS
#   NUM_TRIALS MAX_STEPS REPLAN_STEPS NUM_PARALLEL_CLIENTS VIDEO_EVERY
#   DENOISING_STEPS SEED PORT ROBOCASA_ENV
#
# Examples:
#   # CamVLA geo+distill checkpoint, full 24-task, camera-aware path:
#   CKPT_DIR=$REPO_ROOT/checkpoints/gr00t_robocasa_geo_distill_gtonly/checkpoint-5000 \
#     MODEL_NAME=geo_distill_gtonly_5k USE_CAMERA_PARAMS=true \
#     TASK_SET=all24 NUM_TRIALS=50 \
#     bash scripts/sbatch_lg/submit_eval_robocasa.sh
#
#   # A specific custom set of task families:
#   TASKS="OpenDrawer CloseDrawer TurnOnMicrowave" \
#     bash scripts/sbatch_lg/submit_eval_robocasa.sh

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
SBATCH_SCRIPT=${REPO_ROOT}/scripts/sbatch_lg/eval_robocasa.sbatch

# ---- defaults: baseline checkpoint at step 30000, SMOKE subset ----
export CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/gr00t_robocasa_baseline_2gpu_b8/checkpoint-30000}
export MODEL_NAME=${MODEL_NAME:-robocasa_baseline_30k}
export USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}

# The 24 RoboCasa v0.2 atomic task families (the dataset's coverage).
ALL24=(
  PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
  PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter
  OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
  OpenDrawer CloseDrawer
  TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
  TurnOnStove TurnOffStove
  CoffeeSetupMug CoffeeServeMug CoffeePressButton
  TurnOnMicrowave TurnOffMicrowave
)
# A small, fast subset for end-to-end validation (one PnP, one door, one drawer).
SMOKE=(PnPCounterToCab OpenDrawer CloseDrawer)

# Task selection precedence: explicit TASKS > TASK_SET={all24,smoke} > smoke.
TASK_SET=${TASK_SET:-smoke}
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<< "${TASKS}"
elif [[ "${TASK_SET}" == "all24" ]]; then
  TASK_LIST=("${ALL24[@]}")
else
  TASK_LIST=("${SMOKE[@]}")
fi

# Smoke runs do fewer trials by default; the full benchmark uses 50.
if [[ "${TASK_SET}" == "all24" && -z "${NUM_TRIALS:-}" ]]; then
  export NUM_TRIALS=50
else
  export NUM_TRIALS=${NUM_TRIALS:-10}
fi

mkdir -p "${REPO_ROOT}/logs/eval_robocasa"

if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
  echo "Checkpoint not found: ${CKPT_DIR}/config.json" >&2
  exit 1
fi

echo "Submitting RoboCasa eval:"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS}"
echo "  NUM_TRIALS=${NUM_TRIALS}  (per task family)"
echo "  tasks (${#TASK_LIST[@]}): ${TASK_LIST[*]}"

for TASK in "${TASK_LIST[@]}"; do
  export ENV_NAME=${TASK}
  sbatch \
    --job-name="rceval_${MODEL_NAME}_${TASK}" \
    "${SBATCH_SCRIPT}"
done
