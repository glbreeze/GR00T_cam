#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_pi3x_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_pi3x_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_pi3x_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa, full CamVLA stack, SINGLE-STAGE, pi3x (teacher-
# distillation) point supervision -- the FULL-RESOLUTION (224x224) variant.
# pi3x counterpart of train_robocasa_geo_distill_gtonly_res224.sbatch: identical
# in every way except the PointHead is supervised by pi3x teacher predictions
# (--pi3x-root) instead of ground-truth pointmaps (--gt-point-root).
#
#   1. --point-head-output-resolution 224 : PointHead emits full-res (xy, log_z)
#      via a Pi3X-style ConvHead upsampler; the dataset keeps the un-pooled
#      224x224 targets (no avg-pool). Matches openpi's output_resolution=224.
#   2. pi3x cache points at the UNCOMPRESSED .npy re-pack (per-array, mmap-able)
#      under /scratch/lg154, built by convert_point_cache_to_npy.py from
#      yz11445's compressed .npz. Numerically identical targets, but reading one
#      frame mmaps ~0.4MB instead of decompressing the whole episode
#      (~630ms/cam) -- the .npz path starved the GPUs (low-GPU-util kill).
#
# Everything else mirrors the gtonly recipe:
#   * use_ray_embed + standard cross-view fusion + one PRoPE injection (idx 0).
#   * PointHead supervised by pi3x pointmaps (--pi3x-root), legacy_conf_mse loss
#     (matching the existing LIBERO pi3x scripts; pi3x cache shares the
#     {xy,log_z,conf} schema so it flows through the same aux-loss path).
#   * action_loss_weight = 1.0, pi3x_loss_weight = 1.0; full model trainable.
#   * 30000 steps, starting lr 1e-4 (GR00T default), default warmup/scheduler.
#   * geometric augs off (cache tied to a deterministic 224x224 resize).
#   * --pi3x-cam-subdirs base wrist remaps cache subdir base->video.image,
#     wrist->video.wrist_image.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_pi3x_res224

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- env consumed by train_robocasa.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
export DATA_CONFIG=robocasa
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# --- pi3x teacher pointmap cache: UNCOMPRESSED .npy re-pack (mmap-friendly) ---
# Built by scripts/sbatch_lg/convert_point_cache_to_npy.py from yz11445's
# compressed .npz cache. Subdirs base/ + wrist/, layout
# episode_{NNNNNN}/{xy,log_z,conf}.npy, full-res (T,224,224,*) float16.
PI3X_TARGETS_ROOT=/scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware_pi3x

# --- CamVLA stack + single-stage pi3x supervision at 224 (tyro CLI flags) ---
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
--pi3x-cam-subdirs base wrist \
--pi3x-loss-type legacy_conf_mse \
--pi3x-loss-weight 1.0 \
--point-head-output-resolution 224"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_geo_distill_pi3x_res224

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
