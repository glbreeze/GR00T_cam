#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO with the full CamVLA stack: ray embedding + standard
# cross-view fusion with PRoPE pose-injection + pi3x distillation. Matches the
# openpi `train_pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1.sbatch`
# profile, single-stage variant (one distillation weight, no follow-up stage 2).
#
# Switches the dataset class to CameraAwareLeRobotDataset via --use-camera-params,
# which reads the openpi `agent_/wrist_` extrinsic/intrinsic columns from the
# parquet and loads pi3x targets (one frame per cam per step) from
# $PI3X_TARGETS_ROOT. Geometric augs are *required* off here because the cache
# is tied to a deterministic 224x224 resize.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill

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

# --- pi3x distillation cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
PI3X_TARGETS_ROOT=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

# --- CamVLA / distillation knobs (forwarded as tyro CLI flags) ---
# Mirrors openpi's `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1`:
#   * use_ray_embed = True
#   * cross_view_type = standard, aa_order = fg, one PRoPE injection after the
#     first global block (cross_view_prope_layer_idx 0).
#   * pi3x distillation enabled, loss_type = pi3x_local_pointmap (the openpi
#     default), loss_weight = 1.0 (single-stage; openpi uses 1.0 for stage 1
#     and 0.05 for stage 2).
#
# Pre-flight enforces --use-camvla-model AND --use-camera-params AND
# --disable-geometric-augs whenever --pi3x-root is set; we set all three.
export EXTRA_TRAIN_ARGS="\
--use-camvla-model \
--use-ray-embed \
--cross-view-type standard \
--pose-enc-type prope \
--cross-view-aa-order fg \
--cross-view-prope-layer-idx 0 \
--cross-view-rope-freq 100.0 \
--use-camera-params \
--disable-geometric-augs \
--pi3x-root ${PI3X_TARGETS_ROOT} \
--pi3x-loss-weight 1.0 \
--pi3x-loss-type pi3x_local_pointmap \
--pi3x-ray-loss-weight 1.0 \
--pi3x-depth-loss-weight 1.0 \
--pi3x-depth-weighting pi3x_inverse"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics
# Read your wandb key from the environment / .netrc; do NOT hard-code it here.

export EXP_NAME=gr00t_libero_geo_distill_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
