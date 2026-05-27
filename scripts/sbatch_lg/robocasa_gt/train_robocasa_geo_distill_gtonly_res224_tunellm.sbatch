#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_gtonly_res224_tunellm
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly_res224_tunellm/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly_res224_tunellm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# ============================================================================
# HYPOTHESIS TEST: does UNFREEZING THE LLM let the geometry signal help?
# ----------------------------------------------------------------------------
# In the frozen-backbone setup (tune_llm=tune_visual=False), the trainable
# geometry modules (ray_embed + cross_view_fusion) perturb the visual tokens,
# but those perturbed tokens are then consumed by a FROZEN LLM whose weights
# were tuned for the *original* token statistics. The cheapest thing the
# optimizer can do is suppress the geometry modules (keep LayerScale ~0) so the
# frozen LLM sees baseline-like tokens -> geometry ~= baseline (the LIBERO
# pattern we observed). This run unfreezes the LLM so it can ADAPT to the
# geometry-perturbed tokens, testing whether that unlocks a real gain.
#
# This is a one-flag delta on train_robocasa_geo_distill_gtonly_res224.sbatch
# (the FROZEN control, already trained to 30k at checkpoints/
# gr00t_robocasa_geo_distill_gtonly_res224). Compare this run against:
#   * frozen geo  : gr00t_robocasa_geo_distill_gtonly_res224  (same recipe, frozen)
#   * frozen base : gr00t_robocasa_baseline_2gpu_b8           (no geometry, frozen)
#
# Deltas vs the frozen control (everything else identical):
#   1. --tune-llm                : unfreeze the LLM backbone (THE experiment).
#   2. PER-GROUP LR: --learning-rate 1e-4 keeps the PROVEN action-head/geometry/
#      projector LR (the control's 1e-4, validated with the LLM frozen), while
#      --llm-learning-rate 2e-5 gives the freshly-unfrozen pretrained LLM a
#      gentler LR so it adapts without early instability / catastrophic
#      forgetting. Both groups share the cosine schedule + warmup; only the peak
#      LR differs. (gr00t_finetune.py stashes --llm-learning-rate on training_args;
#      DualBrainTrainer.create_optimizer splits the LLM params into their own
#      group. --llm-learning-rate unset => single-LR baseline, unchanged.)
#   3. --gradient-checkpointing  : recompute activations in backward so the
#      unfrozen LLM fits. Costs ~20-30% compute; control's ~6h 30k run -> ~10-14h.
#   4. BATCH_SIZE 8 + accum 2    : effective batch 16 (== control's 16) at half
#      the per-microbatch activation memory, as an OOM safety margin on top of
#      gradient checkpointing.
# Vision tower stays FROZEN (tune_visual unset) to isolate the LLM effect.
#
# OOM fallback (visible at step 0-1 in slurm-*.err): drop BATCH_SIZE to 4 and
# raise gradient_accumulation_steps to 4 (still effective batch 16). Walltime is
# UNLIMITED on a100_tandon, so completion is not at risk; SAVE_INTERVAL=1000
# means intermediate checkpoints are eval-able if you stop early.

set -euo pipefail

mkdir -p /scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_gtonly_res224_tunellm

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam

# --- env consumed by train_robocasa.sbatch ---
export DATASET_PATH=/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t
export DATA_CONFIG=robocasa
export VIDEO_BACKEND=decord
export EMBODIMENT_TAG=new_embodiment
export BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=8          # 8/GPU -> 4/GPU; accum 2 below restores eff. batch 16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000

# --- ground-truth pointmap cache: UNCOMPRESSED .npy re-pack (mmap-friendly) ---
GT_POINT_TARGETS_ROOT=/scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware

# --- CamVLA stack + single-stage GT supervision at 224, LLM UNFROZEN ---
export EXTRA_TRAIN_ARGS="\
--tune-llm \
--gradient-checkpointing \
--learning-rate 1e-4 \
--llm-learning-rate 2e-5 \
--gradient-accumulation-steps 2 \
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
--pi3x-loss-weight 1.0 \
--point-head-output-resolution 224"

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=gr00t_robocasa_lg
export WANDB_ENTITY=NYU-robotics

export EXP_NAME=gr00t_robocasa_geo_distill_gtonly_res224_tunellm

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_robocasa.sbatch"
