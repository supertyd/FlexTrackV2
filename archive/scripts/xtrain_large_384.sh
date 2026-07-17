#!/bin/bash
# FlexTrackV2-Large 384 (v54) Training script on 4 GPUs

echo "=== Starting FlexTrackV2-Large 384 (v54) Training ==="
python tracking/train.py --script flextrackv2 --config flextrackv2_b384_large_54 --save_dir ./output --mode multiple --nproc_per_node 4
echo "=== FlexTrackV2-Large 384 (v54) Training Completed! ==="
