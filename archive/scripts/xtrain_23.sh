#!/bin/bash
# FlexTrackV2-B224 (v23) Training script
set -e
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Starting FlexTrackV2-B224 (v23) Training ==="
python tracking/train.py --script flextrackv2 --config flextrackv2_b224_23 --save_dir ./output --mode multiple --nproc_per_node 4
echo "=== FlexTrackV2-B224 (v23) Training Completed! ==="
