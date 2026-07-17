#!/bin/bash
# Per-node missing-ratio sweep for ONE config, on RGBT234 + DepthTrack, 5 ratios each.
# Usage: MCI_PYTHON=/coreflow/mambaforge/envs/mci310/bin/python3 run_ratio_sweep_remote.sh <config_name>
set -u
cd /mnt/task_runtime
CFG="${1:?Usage: $0 <config_name>}"
YAML="flextrackv2_b224_56_abl_${CFG}"
RATIOS="000 025 050 075 100"
NUM_GPUS=8
THREADS=8
EPOCH=40
PYTHON_BIN="${MCI_PYTHON:-python3}"

mkdir -p ablation_logs/ratio_sweep

# A run that CUDA-errors on every sequence still exits 0 (per-sequence
# exceptions are caught so the pool never crashes) -- that silently produced
# zero real results across all 8 B200 nodes earlier today. Guard against it:
# after each run, if most sequences failed, stop this node's script instead
# of cascading through every remaining ratio/dataset with junk.
check_real_output() {
  local log="$1" out="$2" min_ok="$3"
  local n_ok
  n_ok=$(ls -1 "$out" 2>/dev/null | wc -l)
  local n_fail
  n_fail=$(grep -c "SEQUENCE FAILED" "$log" 2>/dev/null || echo 0)
  echo "  -> $n_ok result files, $n_fail sequence failures"
  if [ "$n_ok" -lt "$min_ok" ]; then
    echo "FATAL: only $n_ok/$min_ok expected result files for $log -- aborting rest of this node's sweep instead of cascading through junk." | tee -a "$log"
    exit 1
  fi
}

for R in $RATIOS; do
  DS="RGBT234_missR${R}"
  LOG="ablation_logs/ratio_sweep/${CFG}_${DS}.log"
  OUT="workspace/results/${DS}/${YAML}"
  if [ -d "$OUT" ] && [ "$(ls -1 "$OUT" 2>/dev/null | wc -l)" -ge 230 ]; then
    echo "SKIP (already done): $CFG $DS"; continue
  fi
  echo "########## $(date) START $CFG $DS ##########" | tee -a "$LOG"
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$DS" \
    --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
    >> "$LOG" 2>&1
  echo "########## $(date) DONE $CFG $DS (exit=$?) ##########" | tee -a "$LOG"
  check_real_output "$LOG" "$OUT" 230
done

for R in $RATIOS; do
  DS="depthtrack_missR${R}"
  LOG="ablation_logs/ratio_sweep/${CFG}_${DS}.log"
  OUT="workspace/results/${DS}/${YAML}"
  if [ -d "$OUT" ] && [ "$(ls -1 "$OUT" 2>/dev/null | wc -l)" -ge 48 ]; then
    echo "SKIP (already done): $CFG $DS"; continue
  fi
  echo "########## $(date) START $CFG $DS ##########" | tee -a "$LOG"
  "$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
    --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$DS" \
    --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
    --seq_home /mnt/task_runtime/Depthtrack_workspace/sequences \
    >> "$LOG" 2>&1
  echo "########## $(date) DONE $CFG $DS (exit=$?) ##########" | tee -a "$LOG"
  check_real_output "$LOG" "$OUT" 48
done
echo "NODE SWEEP COMPLETE for $CFG $(date)"
