#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s2_tune
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_tune/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_tune/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 2 TUNED VARIANT.
#
# Motivation (vs `train_libero_geo_distill_stage2.sbatch`):
#   The action_loss at end of the openpi-faithful s2 (lr=2.5e-5, ray_w=depth_w=1.0)
#   came out HIGHER than baseline (0.061 vs 0.054). Two diagnosed root causes:
#     1) LR too low: pi-0's 2.5e-5 was tuned for paligemma+gemma_expert; GR00T's
#        flow-matching DiT was trained at 1e-4 and is undertrained at 2.5e-5.
#        -> bump peak to 5e-5 (still 2x lower than baseline, half of openpi).
#     2) Depth aux plateaus at ~0.027 in both single-stage and s2 runs while the
#        ray (xy) term trains down to ~3e-4. The depth term is hitting a floor
#        (Pi3X depth noise on LIBERO + hard conf>=0.1 mask) and "stealing"
#        capacity from the action head without making depth meaningfully better.
#        -> reduce --pi3x-depth-loss-weight 1.0 -> 0.3, keep ray_loss_weight 1.0.
#   Loss type stays `legacy_conf_mse` (conf threshold 0.1 in the loss module);
#   change one axis at a time.
#
# Everything else matches `train_libero_geo_distill_stage2.sbatch`:
#   * action_loss_weight = 1.0
#   * pi3x_loss_weight   = 0.05  (the outer multiplier on the aux total)
#   * full unfreeze (no --trainable-prefixes)
#   * cosine_with_min_lr, min_lr_rate=0.1
#   * 30000 steps
#
# REQUIRES: $STAGE1_CKPT pointing at the Stage 1 checkpoint dir (same one used
# by the openpi-faithful stage 2). The geometry-key scanner in
# GR00T_N1_5.from_pretrained auto-detects ray_embed/cross_view_fusion/point_head
# and skips reset_geometry_modules() to preserve the Stage 1 weights.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_tune

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- Stage 1 checkpoint: required, no default (must point at a real ckpt dir) ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_2gpu_b16/checkpoint-5000}
if [[ ! -f "${STAGE1_CKPT}/config.json" ]] \
   || [[ ! -f "${STAGE1_CKPT}/model.safetensors.index.json" && ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: STAGE1_CKPT does not look like a HF checkpoint: ${STAGE1_CKPT}" >&2
  echo "       Need config.json + model.safetensors(.index.json)." >&2
  exit 1
fi
echo "Stage 2 (tune) starting from: ${STAGE1_CKPT}"

# --- env consumed by train_libero.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/glbreeze/libero_cam_v2_gr00t
export DATA_CONFIG=libero
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=${STAGE1_CKPT}

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# --- pi3x distillation cache (must match the cache used in Stage 1) ---
PI3X_TARGETS_ROOT=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

# --- CamVLA stack + TUNED stage-2 loss weights / LR ---
# Diffs from train_libero_geo_distill_stage2.sbatch:
#   --learning-rate 2.5e-5 -> 5e-5     (action head needs more update headroom)
#   --pi3x-depth-loss-weight (default 1.0) explicitly set to 0.3
#   --pi3x-ray-loss-weight   (default 1.0) explicit for clarity
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
--pi3x-loss-type legacy_conf_mse \
--pi3x-ray-loss-weight 1.0 \
--pi3x-depth-loss-weight 0.3 \
--action-loss-weight 1.0 \
--pi3x-loss-weight 0.05 \
--learning-rate 5e-5 \
--lr-scheduler-type cosine_with_min_lr \
--min-lr-rate 0.1"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s2_tune_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
