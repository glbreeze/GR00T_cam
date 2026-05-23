#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s2_gtonly_lr1e4
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_gtonly_lr1e4/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_gtonly_lr1e4/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 2 (action-focused fine-tune), GROUND-TRUTH point
# supervision, HIGHER LEARNING RATE (1e-4) variant.
#
# Identical to train_libero_geo_distill_stage2_gtonly.sbatch EXCEPT the peak LR
# is raised from 2.5e-5 to 1e-4 (a clean LR-only A/B). Motivation: the
# single-stage recipe (train_libero_geo_distill_gtonly.sbatch) trained well at
# the GR00T default 1e-4, so this tests whether the staged stage-2 benefits from
# the same peak.
#
# CAVEAT: stage 2 starts from an already-adapted stage-1 checkpoint (warmed
# geometry + LIBERO-adapted action head). The original recipe deliberately uses
# 2.5e-5 here -- a note in the base stage-2 script records that 1e-4 was "4x too
# hot for this recipe". Treat this script as an experiment, not the default;
# compare its eval against the 2.5e-5 stage-2 run before adopting it.
#
# Everything else (action_loss_weight=1.0, pi3x_loss_weight=0.05, full unfreeze,
# 30000 steps, GT supervision via legacy_conf_mse, cosine_with_min_lr schedule)
# matches the 2.5e-5 stage-2 gtonly recipe.
#
# REQUIRES: $STAGE1_CKPT pointing at a Stage 1 (gtonly) checkpoint directory, e.g.
#     export STAGE1_CKPT=${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_gtonly_2gpu_b16/checkpoint-5000
# The geometry-key scanner in GR00T_N1_5.from_pretrained inspects the safetensors
# of this checkpoint; finding ray_embed_module/cross_view_fusion/point_head keys
# skips reset_geometry_modules() and preserves the Stage 1 weights.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_gtonly_lr1e4

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- Stage 1 checkpoint: required, no default (must point at a real ckpt dir) ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_gtonly_2gpu_b16/checkpoint-5000}
if [[ ! -f "${STAGE1_CKPT}/config.json" ]] \
   || [[ ! -f "${STAGE1_CKPT}/model.safetensors.index.json" && ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: STAGE1_CKPT does not look like a HF checkpoint: ${STAGE1_CKPT}" >&2
  echo "       Need config.json + model.safetensors(.index.json)." >&2
  exit 1
fi
echo "Stage 2 (gtonly, lr1e4) starting from: ${STAGE1_CKPT}"

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

# --- ground-truth pointmap cache (must match the cache used in Stage 1) ---
GT_POINT_TARGETS_ROOT=/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned

# --- CamVLA stack + stage-2 loss weights (NO --trainable-prefixes: full unfreeze) ---
# HIGHER-LR variant: peak_lr = 1e-4 (vs 2.5e-5 in the base stage-2 recipe).
#   * min_lr_rate = 0.1   (decay floor = 10% of peak = 1e-5)
#   * warmup_ratio remains the GR00T default 0.05 -> 1500 / 30000 steps.
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
--pi3x-loss-type legacy_conf_mse \
--action-loss-weight 1.0 \
--pi3x-loss-weight 0.05 \
--learning-rate 1e-4 \
--lr-scheduler-type cosine_with_min_lr \
--min-lr-rate 0.1"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s2_gtonly_lr1e4_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
