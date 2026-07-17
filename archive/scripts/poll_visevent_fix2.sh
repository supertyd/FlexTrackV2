#!/bin/bash
declare -A CFGMAP=(
  [wkc9uvhn57]=moe_small
  [9bjgb4zxu9]=moe_middle
  [tcjrta3nur]=moe_hybrid
  [tzfqyhjheu]=cma_fixed020
  [9h58ingztm]=no_distill
  [hdrtqt64yv]=pmax_015
  [ke6h6jg4y4]=pmax_025
)
while true; do
  ALL_DONE=1
  for N in "${!CFGMAP[@]}"; do
    CFG="${CFGMAP[$N]}"
    OUT=$(timeout 25 bolt task ssh "$N" << EOF 2>/dev/null
echo "PROCS:\$(ps aux | grep -c test_rgbt_mgpus)"
echo "V:\$(ls workspace/results/VisEvent/flextrackv2_b224_56_abl_${CFG}/ 2>/dev/null | wc -l) VM:\$(ls workspace/results/VisEvent_miss/flextrackv2_b224_56_abl_${CFG}/ 2>/dev/null | wc -l)"
EOF
)
    PROCN=$(echo "$OUT" | grep -oE "PROCS:[0-9]+" | grep -oE "[0-9]+")
    COUNTS=$(echo "$OUT" | grep "V:")
    echo "[$N] $CFG procs=$PROCN $COUNTS"
    if [ "${PROCN:-0}" -gt 1 ]; then ALL_DONE=0; fi
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "ALL_VISFIX2_DONE"
    break
  fi
  sleep 90
done
