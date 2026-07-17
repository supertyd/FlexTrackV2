#!/bin/bash
# CORRECTED + FAST missing-rate sweep on FlexTrackV2 (flextrackv2_b224_56 = V54 weights
# + tuned thresholds; patched threshold lookup so R0 == main SOTA results).
# OMP pinned (bug #15: unpinned BLAS/OpenMP thrashes CPU -> GPUs idle).
set -u
cd /mnt/task_runtime
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CFG=flextrackv2_b224_56
T=16

# expected full-set sizes per dataset (skip a combo if already complete)
expected () { case "$1" in RGBT234) echo 234;; LasHeR) echo 245;; VisEvent) echo 320;; depthtrack) echo 50;; esac; }

run () {  # $1=script $2=base_ds $3=dsname
  exp=$(expected "$2")
  have=$(ls "workspace/results/$3/${CFG}/" 2>/dev/null | wc -l)
  if [ "$have" -ge "$exp" ]; then echo "=== $3 already complete ($have/$exp), skip ==="; return; fi
  rm -rf "workspace/results/$3/${CFG}"
  echo "=== $3 start $(date '+%H:%M') ==="
  python3 "RGBT_workspace/$1" --yaml_name $CFG --dataset_name "$3" \
      --threads $T --num_gpus 8 --epoch 40 > "ablation_logs/fix_$3.log" 2>&1
  echo "  $3 done: $(ls workspace/results/$3/${CFG}/ 2>/dev/null | wc -l)/$exp seqs $(date '+%H:%M')"
}

for ds in RGBT234 LasHeR VisEvent; do
  for r in 000 025 050 075 100; do run test_rgbt_mgpus.py "$ds" "${ds}_missR${r}"; done
done
for r in 000 025 050 075 100; do run test_depthtrack_mgpus.py depthtrack "depthtrack_missR${r}"; done

echo "=== MISSRATE_FIXED_SWEEP_DONE $(date '+%H:%M') ==="
