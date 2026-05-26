#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s2_aux05
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_aux05/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_aux05/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 2 (action-focused fine-tune on top of Stage 1).
#
# AUX-WEIGHT ABLATION: identical to train_libero_geo_distill_stage2.sbatch in
# every respect EXCEPT --pi3x-loss-weight, which is raised 0.05 -> 0.5. This
# isolates the point/aux loss weight as the single changed variable so the
# resulting eval delta can be attributed to it (vs. the full-unfreeze / cosine
# LR schedule that stage 2 also introduces). Motivation: the single-stage
# geo_distill run (aux=1.0 throughout) is the best result so far, while the
# stage-2 runs that decay aux to 0.05 regress on libero_10.
#
# Mirrors openpi `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2`
# otherwise:
#   * action_loss_weight = 1.0   (action dominates)
#   * pi3x_loss_weight   = 0.5   (ABLATION: up from baseline 0.05)
#   * full unfreeze (no --trainable-prefixes) -- backbone + DiT + geometry
#     modules all update together.
#   * 30000 steps.
#
# REQUIRES: $STAGE1_CKPT pointing at a Stage 1 checkpoint directory, e.g.
#     export STAGE1_CKPT=${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_2gpu_b16/checkpoint-5000
# The geometry-key scanner in GR00T_N1_5.from_pretrained inspects the safetensors
# of this checkpoint; finding ray_embed_module/cross_view_fusion/point_head keys
# skips reset_geometry_modules() and preserves the Stage 1 weights.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s2_aux05

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- Stage 1 checkpoint: required, no default (must point at a real ckpt dir) ---
STAGE1_CKPT=${STAGE1_CKPT:-${REPO_ROOT}/checkpoints/gr00t_libero_geo_distill_s1_2gpu_b16/checkpoint-5000}
if [[ ! -f "${STAGE1_CKPT}/config.json" ]] \
   || [[ ! -f "${STAGE1_CKPT}/model.safetensors.index.json" && ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: STAGE1_CKPT does not look like a HF checkpoint: ${STAGE1_CKPT}" >&2
  echo "       Need config.json + model.safetensors(.index.json)." >&2
  exit 1
fi
echo "Stage 2 (aux=0.5) starting from: ${STAGE1_CKPT}"

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

# --- CamVLA stack + stage-2 loss weights (NO --trainable-prefixes: full unfreeze) ---
# LR schedule mirrors openpi's `CosineDecaySchedule(peak_lr=2.5e-5, decay_lr=2.5e-6)`:
#   * peak_lr     = 2.5e-5  (GR00T default 1e-4 was 4x too hot for this recipe)
#   * min_lr_rate = 0.1     (decay floor = 10% of peak = 2.5e-6, matches openpi)
#   * warmup_ratio remains the GR00T default 0.05 -> 1500 / 30000 steps,
#     close to openpi's 1000 absolute steps.
# NOTE: only --pi3x-loss-weight differs from the baseline stage-2 script (0.05 -> 0.5).
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
--action-loss-weight 1.0 \
--pi3x-loss-weight 0.5 \
--learning-rate 1e-4 \
--lr-scheduler-type cosine_with_min_lr \
--min-lr-rate 0.01"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s2_aux05_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
