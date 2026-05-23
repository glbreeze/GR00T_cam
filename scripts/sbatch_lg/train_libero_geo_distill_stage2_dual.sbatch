#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s2_dual
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_dual/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_dual/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 2, DUAL point supervision (pi3x teacher + GT).
#
# Mirrors openpi's `point_target_mix_mode="dual_loss"`: the PointHead is
# supervised against BOTH the pi3x teacher cache and the ground-truth pointmap
# cache, and the two losses are mixed:
#     aux_loss = w_gt * L(pred, gt) + (1 - w_gt) * L(pred, pi3x)
# with w_gt = --point-target-gt-weight. Everything else matches the pi3x
# stage-2 recipe (action_loss_weight=1.0, pi3x_loss_weight=0.05, full unfreeze,
# 30000 steps). The combined and per-channel (aux_gt_loss / aux_pi3x_loss)
# losses are logged to wandb so the mix weight can be tuned.
#
# REQUIRES: $STAGE1_CKPT pointing at a Stage 1 checkpoint directory (use the
# gtonly stage-1 ckpt for an all-GT-warmed geometry stack, or the pi3x stage-1
# ckpt — both expose ray_embed/cross_view_fusion/point_head keys that the
# geometry scanner preserves).

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_dual

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- Stage 1 checkpoint: required, no default (must point at a real ckpt dir) ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_gtonly_2gpu_b16/checkpoint-5000}
if [[ ! -f "${STAGE1_CKPT}/config.json" ]] \
   || [[ ! -f "${STAGE1_CKPT}/model.safetensors.index.json" && ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: STAGE1_CKPT does not look like a HF checkpoint: ${STAGE1_CKPT}" >&2
  echo "       Need config.json + model.safetensors(.index.json)." >&2
  exit 1
fi
echo "Stage 2 (dual) starting from: ${STAGE1_CKPT}"

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

# --- point-target caches (both: enables dual-loss) ---
PI3X_TARGETS_ROOT=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2
GT_POINT_TARGETS_ROOT=/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned

# --- CamVLA stack + stage-2 loss weights (NO --trainable-prefixes: full unfreeze) ---
# --point-target-gt-weight 0.5 weights GT and pi3x equally; raise toward 1.0 to
# trust the simulator GT more, lower toward 0.0 to trust the pi3x teacher more.
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
--gt-point-root ${GT_POINT_TARGETS_ROOT} \
--point-target-gt-weight 0.5 \
--pi3x-loss-type legacy_conf_mse \
--action-loss-weight 1.0 \
--pi3x-loss-weight 0.05 \
--learning-rate 2.5e-5 \
--lr-scheduler-type cosine_with_min_lr \
--min-lr-rate 0.1"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s2_dual_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
