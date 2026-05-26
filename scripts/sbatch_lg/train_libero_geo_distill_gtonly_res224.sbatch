#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_gtonly_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_gtonly_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_gtonly_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO with the full CamVLA stack, SINGLE-STAGE, GROUND-TRUTH point
# supervision. The one-stage counterpart to the gtonly stage1+stage2 pair, and
# the GT analogue of train_libero_geo_distill.sbatch (one-stage pi3x).
#
# Why one stage: the single-stage pi3x distillation run trained well at the
# GR00T default starting lr 1e-4 with a full unfreeze (no geometry warmup, no
# cosine-to-2.5e-6 schedule). We keep that schedule here and only swap the
# supervision source pi3x -> ground-truth pointmaps. Concretely:
#   * use_ray_embed + standard cross-view fusion + one PRoPE injection (idx 0).
#   * PointHead supervised by GT pointmaps (--gt-point-root), legacy_conf_mse
#     loss (GT depth is metric, so no scale alignment; conf is a constant high
#     value, so every pixel is a valid target). Same loss as the gtonly stages.
#   * action_loss_weight = 1.0 (default), pi3x_loss_weight = 1.0 -> action and
#     aux contribute jointly from step 0; full model trainable.
#   * 30000 steps, starting lr 1e-4 (GR00T default -- intentionally NOT
#     overridden; this is the config that worked for one-stage pi3x), default
#     warmup/scheduler.
#
# Geometric augs are *required* off because the GT cache is tied to a
# deterministic 224x224 resize.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_gtonly_res224

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

# --- ground-truth pointmap cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
# Already frame-aligned to libero_cam_v2 ("_aligned" suffix); same episode/frame
# counts as the dataset. Copy locally if the source path becomes unavailable.
GT_POINT_TARGETS_ROOT=/scratch/lg154/cache/point_targets_npy/libero_gt_cam_v2_aligned

# --- CamVLA stack + single-stage GT supervision (forwarded as tyro CLI flags) ---
# No --learning-rate / --lr-scheduler-type / --trainable-prefixes overrides: we
# deliberately keep the GR00T defaults (lr 1e-4, full unfreeze) that worked for
# the one-stage pi3x recipe.
export EXTRA_TRAIN_ARGS="\
--use-camvla-model \
--point-head-output-resolution 224 \
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
--pi3x-loss-weight 1.0"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_gtonly_res224

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
