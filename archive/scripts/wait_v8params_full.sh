#!/bin/bash
while true; do
  n=$(find /mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_depthtrack -iname '*.txt' 2>/dev/null | wc -l)
  echo "progress: $n/50"
  if [ "$n" -ge 50 ]; then break; fi
  sleep 30
done
echo "=== Full complete, restoring list and running vot analysis ==="
cp /mnt/task_runtime/Depthtrack_workspace/sequences/list.txt.full_50.bak /mnt/task_runtime/Depthtrack_workspace/sequences/list.txt
cd /mnt/task_runtime/Depthtrack_workspace
/coreflow/venv/bin/vot analysis --nocache flextrackv2_v56_depthtrack 2>&1 | tail -5
/coreflow/venv/bin/python3 << 'EOF'
import re, glob, os
analysis_dir = '/mnt/task_runtime/Depthtrack_workspace/analysis'
dirs = sorted(glob.glob(os.path.join(analysis_dir, '2026-*')), key=os.path.getmtime)
path = dirs[-1] + '/report.html'
pattern = r'<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">'
content = open(path).read()
m = re.search(pattern, content, re.DOTALL)
if m:
    print(f'RESULT full(V8-params): Precision={float(m.group(1))*100:.2f}% Recall={float(m.group(2))*100:.2f}% EAO={float(m.group(3))*100:.2f}%')
EOF
echo "=== V8PARAMS_ANALYSIS_DONE ==="
