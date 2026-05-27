#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_s2_pi3x_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s2_pi3x_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s2_pi3x_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa -- Stage 2 (action-focused fine-tune), pi3x (teacher-
# distillation) point supervision, FULL-RESOLUTION (224x224) PointHead. pi3x
# counterpart of train_robocasa_geo_distill_stage2_gtonly_res224.sbatch.
#
# Loads from a Stage-1 (pi3x, res224) checkpoint and does a full unfreeze.
# Differs from the gtonly stage-2 res224 recipe only in the target source
# (--pi3x-root instead of --gt-point-root):
#   1. --point-head-output-resolution 224 (must match Stage 1's head).
#   2. pi3x cache points at the UNCOMPRESSED .npy re-pack (mmap-able).
#
# Stage-2 recipe (mirrors the gtonly stage-2):
#   * action_loss_weight = 1.0   (action dominates)
#   * pi3x_loss_weight   = 0.05  (aux point loss kept as a small regulariser)
#   * pi3x-loss-type legacy_conf_mse (matching the existing LIBERO pi3x scripts).
#   * full unfreeze (no --trainable-prefixes)
#   * 30000 steps, lr 1e-4 (GR00T default cosine+warmup -- NOT openpi's 2.5e-5).
#
# REQUIRES: $STAGE1_CKPT pointing at a Stage-1 (pi3x, res224) checkpoint dir, e.g.
#     export STAGE1_CKPT=${REPO_ROOT}/checkpoints/gr00t_robocasa_geo_distill_s1_pi3x_res224/checkpoint-5000
# The geometry-key scanner in GR00T_N1_5.from_pretrained matches the "point_head"
# substring (incl. the 224 upsampler.* keys), so it skips reset_geometry_modules()
# and preserves the Stage-1 224 geometry weights.
#
# CAM-SUBDIR NOTE: --pi3x-cam-subdirs base wrist remaps pi3x cache subdirs base/wrist
# to video.image/video.wrist_image (must match Stage 1).

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_s2_pi3x_res224

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- Stage 1 checkpoint: required, must point at a real (pi3x, res224) ckpt dir ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_robocasa_geo_distill_s1_pi3x_res224/checkpoint-5000}
if [[ ! -f "${STAGE1_CKPT}/config.json" ]] \
   || [[ ! -f "${STAGE1_CKPT}/model.safetensors.index.json" && ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: STAGE1_CKPT does not look like a HF checkpoint: ${STAGE1_CKPT}" >&2
  echo "       Need config.json + model.safetensors(.index.json)." >&2
  exit 1
fi
echo "Stage 2 (pi3x, res224) starting from: ${STAGE1_CKPT}"

# --- env consumed by train_robocasa.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
export DATA_CONFIG=robocasa
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=${STAGE1_CKPT}

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# --- pi3x teacher pointmap cache (must match the cache used in Stage 1) ---
PI3X_TARGETS_ROOT=/scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware_pi3x

# --- CamVLA stack + stage-2 loss weights (NO --trainable-prefixes: full unfreeze) ---
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
--point-head-output-resolution 224 \
--action-loss-weight 1.0 \
--pi3x-loss-weight 0.05"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_geo_distill_s2_pi3x_res224

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
