#!/bin/bash
#SBATCH --job-name=gr00t_robocasa_geo_distill_2stage_pi3x_res224
#SBATCH --output=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_2stage_pi3x_res224/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/GR00T_cam/logs/gr00t_robocasa_geo_distill_2stage_pi3x_res224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# GR00T N1.5 RoboCasa -- BOTH stages of the pi3x (teacher-distillation) geo-distill
# (res224) recipe in ONE SLURM allocation, so the GPUs are requested only once
# (no re-queue gap between stages when GPUs are tight). pi3x counterpart of
# train_robocasa_geo_distill_2stage_gtonly_res224.sbatch.
#
# It simply runs the two standalone res224 pi3x stage scripts back-to-back as child
# processes (single source of truth -- any tweak to a stage script is picked up
# here automatically):
#   Stage 1: train_robocasa_geo_distill_stage1_pi3x_res224.sbatch  (5000 steps)
#   Stage 2: train_robocasa_geo_distill_stage2_pi3x_res224.sbatch  (30000 steps)
#
# Each stage script ``exec``s into train_robocasa.sbatch -> python; ``exec`` only
# replaces the *child* shell spawned by ``bash <stage>.sbatch`` here, so control
# returns to this wrapper when the stage's python exits. ``set -e`` aborts before
# Stage 2 if Stage 1 fails. Stage 2's default STAGE1_CKPT already points at
# Stage 1's checkpoint-5000, so no wiring is needed between them.
#
# The child scripts' own ``#SBATCH`` headers are ignored (they are run via ``bash``,
# not submitted) -- only THIS file's headers define the allocation, and all
# stage stdout/stderr lands in this job's combined log.
#
# Walltime: 35000 total steps. a100_tandon has no max walltime; 48h is ample
# headroom (lower it to backfill-schedule faster when GPUs are tight). If the job
# is killed mid-Stage-2, resume from the last Stage-2 checkpoint by submitting
# train_robocasa_geo_distill_stage2_pi3x_res224.sbatch with RESUME=true.

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/GR00T_cam
mkdir -p "${REPO_ROOT}/logs/gr00t_robocasa_geo_distill_2stage_pi3x_res224"

# Stage scripts live alongside this wrapper in the robocasa_pi3x/ subfolder.
# (train_robocasa.sbatch -- the shared launcher they exec -- stays in the parent
# sbatch_lg/, referenced by absolute path inside each stage script.)
SBATCH_DIR="${REPO_ROOT}/scripts/sbatch_lg/robocasa_pi3x"

echo "##### $(date '+%F %T')  STAGE 1 (geometry warmup, res224) START #####"
bash "${SBATCH_DIR}/train_robocasa_geo_distill_stage1_pi3x_res224.sbatch"
echo "##### $(date '+%F %T')  STAGE 1 DONE -> STAGE 2 (action fine-tune, res224) START #####"
bash "${SBATCH_DIR}/train_robocasa_geo_distill_stage2_pi3x_res224.sbatch"
echo "##### $(date '+%F %T')  STAGE 2 DONE #####"
