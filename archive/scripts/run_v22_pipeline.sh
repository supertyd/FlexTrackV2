#!/bin/bash
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== V22 Training & Evaluation Pipeline Started ==="

CHECKPOINT="/mnt/task_runtime/output/checkpoints/train/flextrackv2/flextrackv2_b224_22/FlexTrackV2_ep0040.pth.tar"

echo "Monitoring V22 training... Checking for final checkpoint: $CHECKPOINT"

while [ ! -f "$CHECKPOINT" ]; do
    sleep 60
done

echo "=== V22 Training Finished! Final Checkpoint Detected! ==="
echo "=== Starting Full V22 Evaluations across all benchmarks ==="

bash /mnt/task_runtime/run_v22_rest_eval.sh > /mnt/task_runtime/v22_evaluation_pipeline.log 2>&1

echo "=== V22 Training & Evaluation Pipeline Completed Successfully! ==="
