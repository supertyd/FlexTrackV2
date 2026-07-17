#!/bin/bash
STATE=/mnt/tmp/claude-0/-mnt-task-runtime/d59af684-d511-4336-8b76-0e1cdbd03aee/scratchpad/l224_gstuned_state
mkdir -p "$STATE"
while true; do
  OUT=$(timeout 30 bolt task ssh qad3e9trm9 << 'EOF' 2>/dev/null
cat /mnt/task_runtime/ablation_logs/l224_gstuned_driver.log 2>/dev/null
EOF
)
  echo "$OUT" | grep -E "DONE|START|COMPLETE|Traceback|FATAL" > "$STATE/cur" 2>/dev/null
  if [ -f "$STATE/prev" ]; then
    diff "$STATE/prev" "$STATE/cur" 2>/dev/null | grep "^>" | sed "s/^> /[qad3e9trm9] /"
  fi
  cp "$STATE/cur" "$STATE/prev"
  sleep 120
done
