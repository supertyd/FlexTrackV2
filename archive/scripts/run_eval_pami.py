import os
import subprocess
env_vot = {'HTTP_PROXY':'http://proxy.config.pcp.local:3128', 'HTTPS_PROXY':'http://proxy.config.pcp.local:3128', 'PYTHONPATH':'/mnt/task_runtime'}
subprocess.run('cd /mnt/task_runtime/Depthtrack_workspace && /coreflow/venv/bin/vot evaluate --workspace /mnt/task_runtime/Depthtrack_workspace flextrackv2_v56_depthtrack', shell=True, env=env_vot)
subprocess.run('cd /mnt/task_runtime/Depthtrack_workspace && /coreflow/venv/bin/vot analysis --nocache --workspace /mnt/task_runtime/Depthtrack_workspace flextrackv2_v56_depthtrack', shell=True, env=env_vot)

import glob, re
analysis_dir = '/mnt/task_runtime/Depthtrack_workspace/analysis'
html_files = glob.glob(os.path.join(analysis_dir, '*', 'report.html'))
if html_files:
    html_path = sorted(html_files, key=os.path.getmtime)[-1]
    with open(html_path, 'r') as f:
        content = f.read()
    pattern = r'<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">.*?</td>.*?<td[^>]*data-value="([0-9\.]+)">'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f'\n🔥 VOT DEPTHTRACK EAO: Precision={float(match.group(1))*100:.2f}%, Recall={float(match.group(2))*100:.2f}%, EAO={float(match.group(3))*100:.2f}%')

print('5. Evaluating RGBT datasets on Local TPAMI Toolkit...')
subprocess.run('cd /mnt/task_runtime && /coreflow/venv/bin/python evaluate_all_visevent_rgbt_pami_v54.py | grep -E "LasHeR|RGBT234"', shell=True, env=env_vot)
