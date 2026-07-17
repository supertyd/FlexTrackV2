#!/usr/bin/env bash
# Evaluate FlexTrack-V2 on LasHeR. Boxes -> workspace/results/LasHeR/flextrackv2/.
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
THREADS=${THREADS:-8}
python RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --yaml_name flextrackv2 \
    --dataset_name LasHeR --threads "${THREADS}"
