#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_gtonly
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa with the full CamVLA stack, SINGLE-STAGE, GROUND-TRUTH
# point supervision. RoboCasa counterpart to train_libero_geo_distill_gtonly.sbatch.
#
# RoboCasa has NO pi3x teacher cache (only a GT pointmap cache exists), so unlike
# Libero there is no pi3x/dual variant -- every RoboCasa geo-distill recipe is
# GT-only. Concretely:
#   * use_ray_embed + standard cross-view fusion + one PRoPE injection (idx 0).
#   * PointHead supervised by GT pointmaps (--gt-point-root), legacy_conf_mse
#     loss (GT depth is metric, so no scale alignment; conf is a constant 10, so
#     every pixel is a valid target).
#   * action_loss_weight = 1.0 (default), pi3x_loss_weight = 1.0 -> action and
#     aux contribute jointly from step 0; full model trainable.
#   * 30000 steps, starting lr 1e-4 (GR00T default -- NOT overridden; matches the
#     one-stage Libero gtonly recipe), default warmup/scheduler.
#
# Geometric augs are *required* off because the GT cache is tied to a
# deterministic 224x224 resize.
#
# CAM-SUBDIR NOTE: the GT cache names its front view "base" (subdirs base/wrist),
# but the dataset's video columns are agent/wrist, so CameraAwareLeRobotDataset's
# default ("agent","wrist") would silently drop the front cam. --pi3x-cam-subdirs
# base wrist remaps cache subdir base -> video.image, wrist -> video.wrist_image.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly

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

# --- ground-truth pointmap cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
# Subdirs base/ + wrist/ (1293 episodes each); {xy, log_z, conf} schema, conf==10.
GT_POINT_TARGETS_ROOT=/scratch/yz11445/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware

# --- CamVLA stack + single-stage GT supervision (forwarded as tyro CLI flags) ---
# No --learning-rate / --lr-scheduler-type / --trainable-prefixes overrides: keep
# the GR00T defaults (lr 1e-4, full unfreeze) that worked for one-stage Libero.
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
--gt-point-root ${GT_POINT_TARGETS_ROOT} \
--pi3x-cam-subdirs base wrist \
--pi3x-loss-type legacy_conf_mse \
--pi3x-loss-weight 1.0"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_geo_distill_gtonly

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
