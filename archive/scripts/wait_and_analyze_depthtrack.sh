#!/bin/bash
set -e
while true; do
  full=$(find /mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_depthtrack -iname '*.txt' 2>/dev/null | wc -l)
  miss=$(find /mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_depthtrack_miss -iname '*.txt' 2>/dev/null | wc -l)
  echo "progress: full=$full/50 miss=$miss/50"
  if [ "$full" -ge 50 ] && [ "$miss" -ge 50 ]; then
    break
  fi
  sleep 30
done

echo "=== Both trackers complete, restoring full sequence list and running vot analysis ==="
cp /mnt/task_runtime/Depthtrack_workspace/sequences/list.txt.full_50.bak /mnt/task_runtime/Depthtrack_workspace/sequences/list.txt

cd /mnt/task_runtime/Depthtrack_workspace
/coreflow/venv/bin/vot analysis --nocache flextrackv2_v56_depthtrack 2>&1 | tail -5
/coreflow/venv/bin/vot analysis --nocache flextrackv2_v56_depthtrack_miss 2>&1 | tail -5

/coreflow/venv/bin/python3 << 'EOF'
import re, glob, os
analysis_dir = '/mnt/task_runtime/Depthtrack_workspace/analysis'
dirs = sorted(glob.glob(os.path.join(analysis_dir, '2026-*')), key=os.path.getmtime)
full_report = dirs[-2] + '/report.html' if len(dirs) >= 2 else None
miss_report = dirs[-1] + '/report.html' if len(dirs) >= 1 else None
pattern = r'<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">'
for name, path in [('full', full_report), ('miss', miss_report)]:
    if path and os.path.exists(path):
        content = open(path).read()
        m = re.search(pattern, content, re.DOTALL)
        if m:
            print(f'RESULT {name}: Precision={float(m.group(1))*100:.2f}% Recall={float(m.group(2))*100:.2f}% EAO={float(m.group(3))*100:.2f}%')
EOF
echo "=== DEPTHTRACK_ANALYSIS_DONE ==="
