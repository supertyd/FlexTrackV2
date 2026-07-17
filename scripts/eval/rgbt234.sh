#!/usr/bin/env bash
# Evaluate FlexTrack-V2 on RGBT234. Writes boxes to
# workspace/results/RGBT234/flextrackv2/, then score with the RGBT toolkit.
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
THREADS=${THREADS:-8}
python RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --yaml_name flextrackv2 \
    --dataset_name RGBT234 --threads "${THREADS}"
