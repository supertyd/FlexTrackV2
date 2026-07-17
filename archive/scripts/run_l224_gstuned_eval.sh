#!/bin/bash
# Evaluate flextrackv2_l224_56 (the "large" model) with the grid-search-tuned
# per-dataset UPT/UPH/INTER/MB thresholds (flextrackv2_l224_56_gstuned.yaml)
# instead of the generic default thresholds it was first evaluated with.
# RGBT234/RGBT234_miss/LasHeR/VisEvent/DepthTrack/DepthTrack_miss use their
# actual grid-search FINAL params; LasHeR_miss/VisEvent_miss have no
# grid-search FINAL on record so they keep the shared default (same as the
# first eval) rather than silently guessing tuned values.
set -u
cd /mnt/task_runtime
YAML="flextrackv2_l224_56_gstuned"
NUM_GPUS=8
THREADS=8
EPOCH=40
PYTHON_BIN="${MCI_PYTHON:-python3}"

mkdir -p ablation_logs

run_rgbt() {
  local ds="$1"
  local log="ablation_logs/${YAML}_eval_${ds}.log"
  echo "########## $(date) START $ds ##########" | tee -a "$log"
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$ds" \
    --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
    >> "$log" 2>&1
  echo "########## $(date) DONE $ds (exit=$?) ##########" | tee -a "$log"
}

run_depth() {
  local ds="$1"
  local log="ablation_logs/${YAML}_eval_${ds}.log"
  echo "########## $(date) START $ds ##########" | tee -a "$log"
  "$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
    --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$ds" \
    --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
    --seq_home /mnt/task_runtime/Depthtrack_workspace/sequences \
    >> "$log" 2>&1
  echo "########## $(date) DONE $ds (exit=$?) ##########" | tee -a "$log"
}

for DS in RGBT234 RGBT234_miss LasHeR LasHeR_miss VisEvent VisEvent_miss; do
  run_rgbt "$DS"
done
for DS in depthtrack depthtrack_miss; do
  run_depth "$DS"
done
echo "GSTUNED EVAL COMPLETE $(date)"
