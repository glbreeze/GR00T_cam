#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_s1_gtonly
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s1_gtonly/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s1_gtonly/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa -- Stage 1 (geometry warmup), GROUND-TRUTH point supervision.
# RoboCasa counterpart to train_libero_geo_distill_stage1_gtonly.sbatch.
#
# RoboCasa has no pi3x teacher cache, so this is GT-only (no pi3x stage1 sibling).
# Concretely:
#   * action_loss_weight = 0.1   (small gradient hint; the frozen DiT trunk still
#                                 propagates that gradient back into cross_view_fusion)
#   * pi3x_loss_weight   = 1.0   (aux point loss dominates)
#   * trainable_prefixes = (ray_embed_module, cross_view_fusion, point_head)
#   * 5000 steps, lr 1e-4 (GR00T default cosine+warmup -- matching the baseline /
#     one-stage recipe; NOT openpi's 2.5e-5).
#
# Stage 2 (train_robocasa_geo_distill_stage2_gtonly.sbatch) loads from this run's
# final ckpt; the geometry-key scanner in GR00T_N1_5.from_pretrained auto-detects
# the trained ray_embed/cross_view_fusion/point_head weights and preserves them
# (does not re-zero via reset_geometry_modules).
#
# CAM-SUBDIR NOTE: GT cache subdirs are base/wrist but the dataset's video columns
# are agent/wrist, so --pi3x-cam-subdirs base wrist remaps base -> video.image,
# wrist -> video.wrist_image (default ("agent","wrist") would drop the front cam).

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s1_gtonly

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
export NUM_TRAIN_STEPS=5000
export SAVE_INTERVAL=1000

# --- ground-truth pointmap cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
GT_POINT_TARGETS_ROOT=/scratch/yz11445/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware

# --- CamVLA stack + stage-1 loss weights / freezing ---
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
--action-loss-weight 0.1 \
--pi3x-loss-weight 1.0 \
--trainable-prefixes \
  backbone.eagle_model.ray_embed_module \
  backbone.eagle_model.cross_view_fusion \
  backbone.eagle_model.point_head"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_geo_distill_s1_gtonly

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
