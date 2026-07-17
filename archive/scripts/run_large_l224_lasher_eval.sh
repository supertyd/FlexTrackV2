#!/bin/bash
# Companion to run_large_l224_worker.sh: adds the LasHeR / LasHeR_miss leg of
# the V56 protocol (compute_ablation_metrics.py already scores these; the
# main worker script just didn't request them). Waits for the epoch-40
# checkpoint so it can run concurrently with the main worker's other evals
# instead of waiting for them to finish first.
#
# Usage: ./run_large_l224_lasher_eval.sh

set -u
CFG=flextrackv2_l224_56
cd /mnt/task_runtime

PYTHON_BIN="${MCI_PYTHON:-python3}"
CKPT="checkpoints/train/flextrackv2/${CFG}/FlexTrackV2_ep0040.pth.tar"

echo "########## [$(hostname)] Waiting for $CKPT at $(date) ##########"
while [ ! -f "$CKPT" ]; do
  sleep 60
done
echo "########## [$(hostname)] Checkpoint found, STARTING LasHeR EVAL at $(date) ##########"

for DS in LasHeR LasHeR_miss; do
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --dataset_name "$DS" \
    --yaml_name "$CFG" --mode parallel --threads 32 --num_gpus 8 --epoch 40 \
    > "ablation_logs/${CFG}_eval_${DS}.log" 2>&1
done

echo "########## [$(hostname)] FINISHED LasHeR EVAL at $(date) — recomputing metrics ##########"
"$PYTHON_BIN" compute_ablation_metrics.py --config "$CFG" >> "ablation_logs/${CFG}_metrics.log" 2>&1
cp "ablation_logs/${CFG}"_eval_LasHeR*.log "ablation_results/${CFG}/" 2>/dev/null

echo "########## [$(hostname)] LasHeR EVAL DONE at $(date) ##########"
