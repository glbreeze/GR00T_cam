#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_baseline
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_baseline/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_baseline/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa baseline finetune — counterpart to openpi's
# `pi0_robocasa24_all24_baseline`. PURE baseline: no CamVLA model, no camera
# params, no point head; only `--disable-geometric-augs` (matching openpi's
# disable_geometric_augs=True). action_loss_weight stays at the GR00T default
# (1.0) and the DiT is tuned (full finetune on H200; on 24GB cards add
# `--no-tune_diffusion_model` and lower BATCH_SIZE).
#
# To turn this into a camvla-parity control for a future RoboCasa geo+distill
# run (as the Libero baseline does), set instead:
#   EXTRA_TRAIN_ARGS="--use-camvla-model --cross-view-type none --use-camera-params --disable-geometric-augs"
# The converted dataset already carries the agent_/wrist_ camera columns, so no
# re-conversion is needed.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_baseline

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- env consumed by train_robocasa.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
export DATA_CONFIG=robocasa
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B

export NUM_GPUS=2
export NUM_WORKERS=16
# Per-device batch. 16 x NUM_GPUS=2 = global 32, matching GR00T's default global
# batch (the LR 1e-4 default is tuned for it). Empirically peaks ~24 GB / 80 GB
# on an A100 full 3B finetune, so it fits A100-80GB and H200 with wide headroom.
# (openpi's pi0_robocasa24_all24_baseline uses global 8; we intentionally follow
# GR00T's recipe here rather than strict openpi batch parity.)
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# Pure baseline: deterministic resize only (matches openpi disable_geometric_augs).
export EXTRA_TRAIN_ARGS="--disable-geometric-augs"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_baseline_2gpu_b8

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
