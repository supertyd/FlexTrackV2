#!/usr/bin/env bash
# Evaluate FlexTrack-V2 on DepthTrack. Boxes -> workspace/results/DepthTrack/flextrackv2/.
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
THREADS=${THREADS:-8}
python RGBT_workspace/test_depthtrack_mgpus.py \
    --script_name flextrackv2 --yaml_name flextrackv2 \
    --dataset_name DepthTrack --threads "${THREADS}"
