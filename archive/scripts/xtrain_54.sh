#!/bin/bash
# FlexTrackV2-B224 (v54) Training script on 8 GPUs

cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Starting FlexTrackV2-B224 (v54) Training ==="
/coreflow/venv/bin/python tracking/train.py --script flextrackv2 --config flextrackv2_b224_54 --save_dir ./output --mode multiple --nproc_per_node 8
echo "=== FlexTrackV2-B224 (v54) Training Completed! ==="
