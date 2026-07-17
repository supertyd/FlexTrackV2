#!/bin/bash
# FlexTrackV2-B224 (v22) Training script
set -e
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Starting FlexTrackV2-B224 (v22) Training ==="
CUDA_VISIBLE_DEVICES=2,3 python tracking/train.py --script flextrackv2 --config flextrackv2_b224_22 --save_dir ./output --mode multiple --nproc_per_node 2
echo "=== FlexTrackV2-B224 (v22) Training Completed! ==="
