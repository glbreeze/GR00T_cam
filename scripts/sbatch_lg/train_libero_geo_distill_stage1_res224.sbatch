#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s1_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --nodelist=ga001
#SBATCH --gres=gpu:a100:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 1 (geometry warmup).
#
# Mirrors openpi `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1`:
#   * action_loss_weight = 0.1   (action loss provides a small gradient hint;
#                                 the frozen DiT trunk still propagates that
#                                 gradient back into cross_view_fusion)
#   * pi3x_loss_weight   = 1.0   (aux distillation dominates)
#   * trainable_prefixes = (ray_embed_module, cross_view_fusion, point_head)
#     -> the pretrained Eagle backbone + action head stay frozen at base
#        values; only the new geometry/aux modules learn.
#   * 5000 steps, lr 2.5e-5 (matches openpi's CosineDecaySchedule peak).
#
# Stage 2 (see train_libero_geo_distill_stage2.sbatch) loads from this run's
# final ckpt; the geometry-key scanner in GR00T_N1_5.from_pretrained auto-
# detects the trained ray_embed/cross_view_fusion/point_head weights and
# preserves them (does not re-zero via reset_geometry_modules).

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_res224

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
export NUM_TRAIN_STEPS=5000
export SAVE_INTERVAL=1000

# --- pi3x distillation cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
PI3X_TARGETS_ROOT=/scratch/lg154/cache/point_targets_npy/libero_pi3x_cam_v2

# --- CamVLA stack + stage-1 loss weights / freezing ---
# Geometry topology identical to the single-stage geo_distill recipe; the only
# stage-1-specific knobs are action-loss/pi3x-loss weights and trainable_prefixes.
# res224: lr/scheduler overrides removed -> GR00T default (start lr 1e-4, cosine, warmup 0.05), matching the baseline.
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
--pi3x-root ${PI3X_TARGETS_ROOT} \
--pi3x-loss-type legacy_conf_mse \
--action-loss-weight 0.1 \
--pi3x-loss-weight 1.0 \
--trainable-prefixes \
  backbone.eagle_model.ray_embed_module \
  backbone.eagle_model.cross_view_fusion \
  backbone.eagle_model.point_head"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s1_res224

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
