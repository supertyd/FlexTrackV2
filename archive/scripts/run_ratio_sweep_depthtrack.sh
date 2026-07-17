#!/bin/bash
set -u
cd /mnt/task_runtime

CONFIGS="rung0 no_hallucinate no_recon_loss uni_a2r uni_r2a no_ortho"
RATIOS="000 025 050 075 100"
NUM_GPUS=4
THREADS=4
EPOCH=40
SEQ_HOME=/mnt/task_runtime/Depthtrack_workspace/sequences

mkdir -p ablation_logs/ratio_sweep

for CFG in $CONFIGS; do
  YAML="flextrackv2_b224_56_abl_${CFG}"
  for R in $RATIOS; do
    DS="depthtrack_missR${R}"
    LOG="ablation_logs/ratio_sweep/${CFG}_${DS}.log"
    OUT="workspace/results/${DS}/${YAML}"
    if [ -d "$OUT" ] && [ "$(ls -1 "$OUT" 2>/dev/null | wc -l)" -ge 48 ]; then
      echo "SKIP (already done): $CFG $DS"
      continue
    fi
    echo "########## $(date) START $CFG $DS ##########" | tee -a "$LOG"
    python3 RGBT_workspace/test_depthtrack_mgpus.py \
      --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$DS" \
      --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
      --seq_home "$SEQ_HOME" \
      >> "$LOG" 2>&1
    echo "########## $(date) DONE $CFG $DS (exit=$?) ##########" | tee -a "$LOG"
  done
done
echo "ALL DEPTHTRACK RATIO SWEEP RUNS COMPLETE $(date)"
