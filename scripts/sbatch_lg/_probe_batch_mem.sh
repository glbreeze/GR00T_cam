#!/bin/bash
# Probe peak GPU memory / OOM for a given per-device batch size on ONE GPU,
# mirroring the RoboCasa baseline finetune args (full 3B finetune,
# --disable-geometric-augs). Runs 4 real optimizer steps. Throwaway.
#   usage: _probe_batch_mem.sh <BATCH_SIZE>
set -uo pipefail
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/gr00t
cd /scratch/lg154/Research/GR00T_cam

export HF_HOME=/scratch/lg154/.huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BS=${1:?need batch size}
OUT=/scratch/lg154/tmp/_bsprobe/bs${BS}
rm -rf "$OUT"; mkdir -p "$OUT"

SMI_LOG=$(mktemp)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$SMI_LOG" 2>/dev/null &
SMIPID=$!

timeout 1200 python scripts/gr00t_finetune.py \
  --dataset-path /scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t \
  --output-dir "$OUT" \
  --data-config robocasa \
  --video-backend decord \
  --embodiment-tag new_embodiment \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --num-gpus 1 \
  --batch-size "$BS" \
  --dataloader-num-workers 8 \
  --max-steps 4 \
  --save-steps 1000 \
  --report-to tensorboard \
  --disable-geometric-augs > "$OUT/train.log" 2>&1
RC=$?

kill $SMIPID 2>/dev/null
PEAK=$(sort -n "$SMI_LOG" | tail -1)
rm -f "$SMI_LOG"

echo "=========================================="
echo "BS=$BS  rc=$RC  peak_mem=${PEAK:-?} MiB / 81920 MiB"
if [ "$RC" -ne 0 ]; then
  echo "--- OOM check ---"
  grep -iE "out of memory|OutOfMemoryError|CUDA out of memory|tried to allocate" "$OUT/train.log" | head -3
  echo "--- last 12 log lines ---"
  tail -12 "$OUT/train.log"
else
  echo "FIT OK (completed 4 steps)"
  grep -iE "'loss'|train_loss|loss=" "$OUT/train.log" | tail -2
fi
rm -rf "$OUT"
