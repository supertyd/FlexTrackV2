#!/bin/bash
# Watch the 8 parallel VOT22RGBD workers (already running, detached) to completion,
# then run a single-GPU fill pass on the MAIN workspace to close any gaps
# (incl. the 10 dirs deleted by mistake), then compute EAO/Acc/Rob.
cd /mnt/task_runtime
export http_proxy="http://proxy.config.pcp.local:3128"
export https_proxy="http://proxy.config.pcp.local:3128"
TRK=flextrackv2_l224_56_gstuned_vot22
BASE=VOT22RGBD_workspace/results/${TRK}/baseline

# 1. wait for parallel workers (pgrep -f: full cmdline, no ps-aux truncation)
while pgrep -f "vot evaluate --workspace /mnt/task_runtime/vot_par/w" >/dev/null 2>&1; do
  sleep 120
done
echo "=== parallel workers finished $(date '+%F %H:%M') · completed $(ls $BASE 2>/dev/null | wc -l)/127 ==="

# 2. fill pass on MAIN workspace (skips complete seqs, computes only missing)
for attempt in 1 2 3; do
  n=$(ls $BASE 2>/dev/null | wc -l)
  [ "$n" -ge 127 ] && break
  echo "=== fill pass $attempt: $n/127 -> running vot evaluate on main workspace (GPU0) ==="
  CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /mnt/task_runtime/VOT22RGBD_workspace "$TRK" \
      > ablation_logs/vot_par/fill_${attempt}.log 2>&1
done

# 3. compute metrics
n=$(ls $BASE 2>/dev/null | wc -l)
echo "=== final completed: $n/127 ==="
if [ "$n" -ge 127 ]; then
  python3 compute_ablation_metrics.py --config flextrackv2_l224_56_gstuned --only VOT22RGBD 2>&1 | tail -15
  echo "=== FINAL VOT22RGBD (gstuned) ==="
  grep -A5 '"VOT22RGBD"' ablation_results/flextrackv2_l224_56_gstuned/metrics.json 2>/dev/null
else
  echo "!!! still short: $n/127 — check ablation_logs/vot_par/*.log"
fi
echo "=== GSTUNED_VOT22_ALLDONE ==="
