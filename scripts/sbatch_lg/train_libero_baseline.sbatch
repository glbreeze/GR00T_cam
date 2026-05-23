#!/bin/bash
#SBATCH --job-name=gr00t_libero_baseline
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_baseline/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_baseline/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO baseline finetune — apples-to-apples control for the geo+
# distill run. We turn on `--use-camvla-model` so the model code path is
# IDENTICAL to the geo+distill run (Eagle2_5_VLCamVLA instead of Eagle2_5_VL)
# but leave every geometry submodule OFF: no ray embedding, no cross-view
# fusion (`cross_view_type=none`), no PRoPE (`pose_enc_type` stays "null" by
# default), no pi3x distillation. The CamVLA subclass without any modules
# enabled is byte-for-byte equivalent to vanilla Eagle in the forward pass
# (see Eagle2_5_VLCamVLA.extract_feature: every geometry tap is gated).
#
# Data path also matches the geo+distill run: `--use-camera-params` swaps in
# CameraAwareLeRobotDataset (reads agent_/wrist_ extrinsic/intrinsic from the
# parquet — these columns are ignored by the model when the geometry modules
# are off, but the dataset class is the same as the geo+distill run); and
# `--disable-geometric-augs` flips augs to the deterministic 224x224 resize
# the geo+distill run requires. Without both, the comparison would be
# confounded by a different dataset class AND a different augmentation
# distribution, not just by the geometry modules.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_baseline

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- env consumed by train_libero.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/glbreeze/libero_cam_v2_gr00t
export DATA_CONFIG=libero
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# --- CamVLA flags (geometry submodules OFF, but model + data path match the
# --- geo+distill run so the only varying axis is the geometry stack itself) ---
# cross_view_type defaults to "none", pose_enc_type defaults to "null", and
# use_ray_embed defaults to False — we set --cross-view-type none explicitly
# for documentation, the others stay at their defaults.
export EXTRA_TRAIN_ARGS="\
--use-camvla-model \
--cross-view-type none \
--use-camera-params \
--disable-geometric-augs"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics
# Read your wandb key from the environment / .netrc; do NOT hard-code it here.
# If you really need to override at the wrapper level, set WANDB_API_KEY_OVERRIDE.

export EXP_NAME=gr00t_libero_baseline_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
