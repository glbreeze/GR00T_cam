#!/bin/bash
#SBATCH --job-name=gr00t_libero_geo_distill_2stage_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_2stage_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_libero_geo_distill_2stage_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 LIBERO -- BOTH stages of the pi3x (teacher-distillation) geo-distill
# (res224) recipe in ONE SLURM allocation, so the GPUs are requested only once
# (no re-queue gap between stages when GPUs are tight). pi3x counterpart of
# train_libero_geo_distill_2stage_gtonly_res224.sbatch.
#
# Runs the two standalone res224 pi3x stage scripts back-to-back as child
# processes (single source of truth -- any tweak to a stage script is picked up
# here automatically):
#   Stage 1: train_libero_geo_distill_stage1_res224.sbatch  (5000 steps)
#   Stage 2: train_libero_geo_distill_stage2_res224.sbatch  (30000 steps)
#
# Each stage script ``exec``s into train_libero.sbatch -> python; ``exec`` only
# replaces the *child* shell spawned by ``bash <stage>.sbatch`` here, so control
# returns to this wrapper when the stage's python exits. ``set -e`` aborts before
# Stage 2 if Stage 1 fails. Stage 2's default STAGE1_CKPT already points at
# Stage 1's (res224) checkpoint-5000, so no wiring is needed between them.
#
# Only THIS file's #SBATCH headers define the allocation; the child scripts'
# headers are inert when run via ``bash`` (not ``sbatch``), and all stage
# stdout/stderr lands in this job's combined log.
#
# NOTE: the aux02/aux05 stage-2 res224 variants ALSO build on this pi3x Stage 1
# (their STAGE1_CKPT defaults to gr00t_libero_geo_distill_s1_res224),
# so running this wrapper produces the Stage-1 ckpt they need too.
#
# Walltime: 35000 total steps. h200_tandon has no max walltime; 48h is ample.
# If killed mid-Stage-2, resume with train_libero_geo_distill_stage2_res224.sbatch
# and RESUME=true.

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam
mkdir -p "${REPO_ROOT}/logs/gr00t_libero_geo_distill_2stage_res224"

SBATCH_DIR="${REPO_ROOT}/scripts/sbatch_lg"

echo "##### $(date '+%F %T')  STAGE 1 (pi3x geometry warmup, res224) START #####"
bash "${SBATCH_DIR}/train_libero_geo_distill_stage1_res224.sbatch"
echo "##### $(date '+%F %T')  STAGE 1 DONE -> STAGE 2 (pi3x action fine-tune, res224) START #####"
bash "${SBATCH_DIR}/train_libero_geo_distill_stage2_res224.sbatch"
echo "##### $(date '+%F %T')  STAGE 2 DONE #####"
