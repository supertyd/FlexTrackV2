#!/bin/bash
# FlexTrackV2-B224 (v55) Training script on 8 GPUs

echo "=== Starting FlexTrackV2-B224 (v55) Training ==="
python tracking/train.py --script flextrackv2 --config flextrackv2_b224_55 --save_dir ./output --mode multiple --nproc_per_node 8
echo "=== FlexTrackV2-B224 (v55) Training Completed! ==="
