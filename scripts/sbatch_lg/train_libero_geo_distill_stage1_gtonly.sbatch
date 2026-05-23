#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_s1_gtonly
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_gtonly/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_gtonly/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- Stage 1 (geometry warmup), GROUND-TRUTH point supervision.
#
# Mirrors openpi `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1_gtonly`:
# identical to the pi3x stage-1 recipe (train_libero_geo_distill_stage1.sbatch)
# except the PointHead is supervised by GROUND-TRUTH pointmaps from the
# simulator instead of pi3x teacher predictions. Concretely:
#   * action_loss_weight = 0.1   (action loss provides a small gradient hint;
#                                 the frozen DiT trunk still propagates that
#                                 gradient back into cross_view_fusion)
#   * pi3x_loss_weight   = 1.0   (aux point loss dominates)
#   * trainable_prefixes = (ray_embed_module, cross_view_fusion, point_head)
#   * 5000 steps, lr 2.5e-5 (matches openpi's CosineDecaySchedule peak).
#
# The GT cache shares the pi3x npz schema ({xy, log_z, conf} per cam per
# episode), so it flows through the same aux-loss path; only the target source
# differs (--gt-point-root instead of --pi3x-root). conf is a constant high
# value, so legacy_conf_mse treats every pixel as a valid target.
#
# Stage 2 (train_libero_geo_distill_stage2_gtonly.sbatch) loads from this run's
# final ckpt; the geometry-key scanner in GR00T_N1_5.from_pretrained auto-
# detects the trained ray_embed/cross_view_fusion/point_head weights and
# preserves them (does not re-zero via reset_geometry_modules).

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_s1_gtonly

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

# --- ground-truth pointmap cache (openpi-flavor, 224x224, per-cam .npz/episode) ---
# Already frame-aligned to libero_cam_v2 ("_aligned" suffix); same episode/frame
# counts as the dataset. Copy locally if the source path becomes unavailable.
GT_POINT_TARGETS_ROOT=/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned

# --- CamVLA stack + stage-1 loss weights / freezing ---
# Geometry topology identical to the pi3x stage-1 recipe; only the target source
# (--gt-point-root) and the gtonly semantics differ.
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
--action-loss-weight 0.1 \
--pi3x-loss-weight 1.0 \
--trainable-prefixes \
  backbone.eagle_model.ray_embed_module \
  backbone.eagle_model.cross_view_fusion \
  backbone.eagle_model.point_head \
--learning-rate 2.5e-5"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_libero_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_libero_geo_distill_s1_gtonly_2gpu_b16

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_libero.sbatch"
