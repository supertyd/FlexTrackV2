#!/bin/bash
# Snapshot the training status of all 9 V56 ablation jobs (this node's
# moe_big + the 8 remote Bolt nodes) in parallel. Run from the current node.
#
# Usage:
#   ./monitor_ablations.sh          # one snapshot
#   watch -n 60 ./monitor_ablations.sh   # refresh every 60s (each snapshot
#                                          # itself takes ~10-20s due to the
#                                          # 8 parallel bolt task ssh round trips)

set -u

declare -A NODES=(
  [rung0]=wkc9uvhn57
  [rung0_seedB]=9bjgb4zxu9
  [rung0_seedC]=tcjrta3nur
  [no_hallucinate]=tzfqyhjheu
  [no_recon_loss]=9h58ingztm
  [uni_a2r]=hdrtqt64yv
  [uni_r2a]=ke6h6jg4y4
  [no_ortho]=qad3e9trm9
)

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# --- 8 remote nodes, all in parallel ---
for CFG in "${!NODES[@]}"; do
  ID="${NODES[$CFG]}"
  (
    OUT=$(timeout 25 bolt task ssh "$ID" 2>/dev/null << EOF
echo "GPU_LINE:\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
echo "LAST_LINE:\$(grep -iE 'loss' ablation_logs/flextrackv2_b224_56_abl_${CFG}_train.log 2>/dev/null | grep -v TypedStorage | tail -1)"
echo "ERR_COUNT:\$(grep -c 'Error' ablation_logs/flextrackv2_b224_56_abl_${CFG}_train.log 2>/dev/null || echo 0)"
echo "PROC_COUNT:\$(ps aux | grep run_training | grep -v grep | wc -l)"
EOF
)
    GPU=$(echo "$OUT" | grep "^GPU_LINE:" | sed 's/^GPU_LINE://')
    LAST=$(echo "$OUT" | grep "^LAST_LINE:" | sed 's/^LAST_LINE://')
    ERR=$(echo "$OUT" | grep "^ERR_COUNT:" | sed 's/^ERR_COUNT://')
    PROC=$(echo "$OUT" | grep "^PROC_COUNT:" | sed 's/^PROC_COUNT://')
    echo "${CFG}|${ID}|${GPU}|${LAST}|${ERR}|${PROC}" > "$TMPDIR/$CFG"
  ) &
done
wait

echo "=================================================================================="
printf "%-14s %-12s %-8s %-8s %s\n" "CONFIG" "TASK_ID" "PROC" "ERRORS" "LATEST PROGRESS"
echo "=================================================================================="
for CFG in "${!NODES[@]}"; do
  [ -f "$TMPDIR/$CFG" ] || continue
  IFS='|' read -r name id gpu last err proc < "$TMPDIR/$CFG"
  ITER=$(echo "$last" | grep -oE '\[train: [0-9]+, [0-9]+ / [0-9]+\]' || echo "(no progress line yet)")
  LOSS=$(echo "$last" | grep -oE '0frame_Loss/total: [0-9.]+' || echo "")
  STATUS="OK"
  if [ "${err:-0}" -gt 0 ] 2>/dev/null && [ -z "$ITER" -o "$ITER" = "(no progress line yet)" ]; then
    STATUS="CHECK (errors logged, no progress)"
  fi
  printf "%-14s %-12s %-8s %-8s %s %s\n" "$name" "$id" "${proc:-?}" "${err:-?}" "$ITER" "$LOSS"
done
echo "=================================================================================="
echo "GPU util (8 values per node) and full log tails: re-run with individual"
echo "bolt task ssh <id> commands for detail. Generated at: $(date)"
