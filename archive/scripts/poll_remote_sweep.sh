#!/bin/bash
# Poll all 8 B200 nodes' ratio-sweep driver logs, emit only new lines (each
# becomes a Monitor event). Run this under Monitor.
NODES="wkc9uvhn57 9bjgb4zxu9 tcjrta3nur tzfqyhjheu 9h58ingztm hdrtqt64yv ke6h6jg4y4 qad3e9trm9"
STATE_DIR=/mnt/tmp/claude-0/-mnt-task-runtime/d59af684-d511-4336-8b76-0e1cdbd03aee/scratchpad/poll_state
mkdir -p "$STATE_DIR"
while true; do
  for N in $NODES; do
    OUT=$(timeout 30 bolt task ssh "$N" << 'EOF' 2>/dev/null
cat /mnt/task_runtime/ablation_logs/ratio_sweep_remote_driver.log 2>/dev/null
exit
EOF
)
    echo "$OUT" | grep -E "DONE|START|COMPLETE|Traceback|FATAL|Killed" > "$STATE_DIR/${N}.cur" 2>/dev/null
    if [ -f "$STATE_DIR/${N}.prev" ]; then
      diff "$STATE_DIR/${N}.prev" "$STATE_DIR/${N}.cur" 2>/dev/null | grep "^>" | sed "s/^> /[$N] /"
    fi
    cp "$STATE_DIR/${N}.cur" "$STATE_DIR/${N}.prev"
  done
  sleep 120
done
