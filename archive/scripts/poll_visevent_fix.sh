#!/bin/bash
NODES="wkc9uvhn57 9bjgb4zxu9 tcjrta3nur tzfqyhjheu 9h58ingztm hdrtqt64yv ke6h6jg4y4 qad3e9trm9"
STATE=/mnt/tmp/claude-0/-mnt-task-runtime/d59af684-d511-4336-8b76-0e1cdbd03aee/scratchpad/visevent_fix_state
mkdir -p "$STATE"
while true; do
  ALL_DONE=1
  for N in $NODES; do
    OUT=$(timeout 25 bolt task ssh "$N" << 'EOF' 2>/dev/null
ps aux | grep -c test_rgbt_mgpus
tail -3 ablation_logs/visevent_fix.log 2>/dev/null | grep -E "fps|SEQUENCE FAILED|Totally"
EOF
)
    PROCN=$(echo "$OUT" | grep -oE '^[0-9]+$' | head -1)
    LAST=$(echo "$OUT" | grep -E "fps|Totally" | tail -1)
    echo "[$N] procs=$PROCN last=$LAST"
    if [ "${PROCN:-0}" -gt 0 ]; then ALL_DONE=0; fi
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "ALL_REMOTE_VISEVENT_FIX_DONE"
    break
  fi
  sleep 120
done
