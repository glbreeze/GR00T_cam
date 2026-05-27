#!/usr/bin/env bash
# NYU SHANGHAI HPC variant of scripts/sbatch_lg/submit_eval_robocasa.sh.
# Eval ONE gr00t checkpoint over a set of RoboCasa v0.2 task families, PACKING
# the tasks into NUM_GROUPS jobs (default 4 jobs x 6 tasks) so one eval uses 4
# GPUs. Each job loads the inference server once and loops over its 6 tasks
# (scripts/sbatch_sh/eval_robocasa.sbatch). Sized for the sfscai QOS
# (cpu=64, gpu=8 per user): 4 jobs x 8 cpu = 32 cpu / 4 gpu per eval, so you can
# run TWO checkpoints (baseline + geo) concurrently within the cap.
#
# Usage:
#   # baseline, full 24-task @ 50 trials, on h20 (4 jobs x 6 tasks):
#   TASK_SET=all24 bash scripts/sbatch_sh/submit_eval_robocasa.sh
#
#   # geo camera-aware, full 24-task @ 50 trials, on h20:
#   CKPT_DIR=$PWD/checkpoints/gr00t_robocasa_geo_distill_gtonly_res224/checkpoint-30000 \
#     MODEL_NAME=geo_gtonly_res224_30k USE_CAMERA_PARAMS=true \
#     TASK_SET=all24 bash scripts/sbatch_sh/submit_eval_robocasa.sh
#
#   # run both back-to-back (uses all 8 h20 GPUs):
#   TASK_SET=all24 bash scripts/sbatch_sh/submit_eval_robocasa.sh
#   CKPT_DIR=... MODEL_NAME=geo_... USE_CAMERA_PARAMS=true TASK_SET=all24 \
#     bash scripts/sbatch_sh/submit_eval_robocasa.sh
#
# Knobs (export to override): CKPT_DIR MODEL_NAME USE_CAMERA_PARAMS NUM_TRIALS
# MAX_STEPS REPLAN_STEPS NUM_PARALLEL_CLIENTS VIDEO_EVERY DENOISING_STEPS SEED
# DATA_CONFIG EMBODIMENT_TAG NUM_GROUPS(=4) TASK_SET(all24|smoke) TASKS
# ACCOUNT(=sfscai) PARTITION(=sfscai) GRES(=gpu:h20:1).

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/GR00T_cam
SBATCH_SCRIPT=${REPO_ROOT}/scripts/sbatch_sh/eval_robocasa.sbatch

# ---- defaults: baseline checkpoint at step 30000 ----
export CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/gr00t_robocasa_baseline_2gpu_b8/checkpoint-30000}
export MODEL_NAME=${MODEL_NAME:-robocasa_baseline_30k}
export USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS:-false}

# ---- SLURM placement (Shanghai); passed on the sbatch CLI to override #SBATCH ----
ACCOUNT=${ACCOUNT:-sfscai}
PARTITION=${PARTITION:-sfscai}
GRES=${GRES:-gpu:h20:1}
NUM_GROUPS=${NUM_GROUPS:-4}

# The 24 RoboCasa v0.2 atomic task families.
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
SMOKE=(PnPCounterToCab OpenDrawer CloseDrawer)

# Task selection: explicit TASKS > TASK_SET={all24,smoke} > smoke.
TASK_SET=${TASK_SET:-smoke}
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<< "${TASKS}"
elif [[ "${TASK_SET}" == "all24" ]]; then
  TASK_LIST=("${ALL24[@]}")
else
  TASK_LIST=("${SMOKE[@]}")
fi

export NUM_TRIALS=${NUM_TRIALS:-50}

mkdir -p "${REPO_ROOT}/logs/eval_robocasa"
[[ -f "${CKPT_DIR}/config.json" ]] || { echo "Checkpoint not found: ${CKPT_DIR}/config.json" >&2; exit 1; }

# Split TASK_LIST into NUM_GROUPS groups (ceil division -> ~6 tasks/group for 24).
n=${#TASK_LIST[@]}
groups=${NUM_GROUPS}
(( groups > n )) && groups=${n}
per=$(( (n + groups - 1) / groups ))   # ceil

echo "Submitting RoboCasa eval (Shanghai, packed):"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  MODEL_NAME=${MODEL_NAME}  USE_CAMERA_PARAMS=${USE_CAMERA_PARAMS}"
echo "  NUM_TRIALS=${NUM_TRIALS}/task  tasks=${n}  -> ${groups} job(s) x ~${per} tasks"
echo "  SLURM: account=${ACCOUNT} partition=${PARTITION} gres=${GRES}"

g=0
for (( start=0; start<n; start+=per )); do
  group=( "${TASK_LIST[@]:start:per}" )
  (( ${#group[@]} > 0 )) || continue
  export ENV_NAMES="${group[*]}"
  echo "  job ${g}: ${ENV_NAMES}"
  sbatch \
    --job-name="rceval_${MODEL_NAME}_g${g}" \
    --account="${ACCOUNT}" --partition="${PARTITION}" --gres="${GRES}" \
    "${SBATCH_SCRIPT}"
  g=$(( g + 1 ))
done
echo "Submitted ${g} job(s). Summaries -> logs/eval_robocasa/${MODEL_NAME}/<task>/summary/"
