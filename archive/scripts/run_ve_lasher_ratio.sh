#!/bin/bash
# VisEvent + LasHeR missing-RATIO sweep (rung0 baseline) via the SAME infra as
# the existing RGBT234/DepthTrack sweeps, so all 4 datasets share one setting:
# built-in _missR{000,025,050,075,100}, full test sets, drop-aux / keep-RGB.
set -u
cd /mnt/task_runtime            # save paths are ./workspace/results/... (matches existing)
CFG=flextrackv2_b224_56_abl_rung0

for ds in VisEvent LasHeR; do
  for r in 000 025 050 075 100; do
    dsname="${ds}_missR${r}"
    echo "=== $dsname start $(date '+%H:%M') ==="
    python3 RGBT_workspace/test_rgbt_mgpus.py --yaml_name $CFG --dataset_name $dsname \
        --threads 8 --num_gpus 8 --epoch 40 \
        > ablation_logs/ratio_${dsname}.log 2>&1
    echo "  $dsname done: $(ls ./workspace/results/${dsname}/${CFG}/ 2>/dev/null | wc -l) seqs"
  done
done
echo "=== VE_LASHER_RATIO_DONE ==="
