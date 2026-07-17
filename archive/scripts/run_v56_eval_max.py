import os
import subprocess

def run_cmd(cmd, env_add={}):
    env = os.environ.copy()
    env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
    env.update(env_add)
    print(f'Running: {cmd}')
    subprocess.run(cmd, shell=True, env=env)

# Removed cleaning step to allow resuming

print('2. Running Parallel Tracking for DepthTrack & RGBT (V56) - MAX UTILIZATION...')
cmds = [
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=0,1 /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b224_56 --mode parallel --threads 20 --num_gpus 2 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=2,3 /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 20 --num_gpus 2 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=4 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224_56 --mode parallel --threads 10 --num_gpus 1 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=5 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 10 --num_gpus 1 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=6 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224_56 --mode parallel --threads 10 --num_gpus 1 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=7 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 10 --num_gpus 1 --epoch 40'
]
ps = [subprocess.Popen(c, shell=True) for c in cmds]
for p in ps: p.wait()

print('3. Converting DepthTrack results for VOT toolkit...')
run_cmd('cat > /mnt/task_runtime/create_vot_v56.py << "INNEREOF"\nimport os, shutil\nfor dset in ["depthtrack", "depthtrack_miss"]:\n  src = f"/mnt/task_runtime/workspace/results/{dset}/flextrackv2_b224_56"\n  dst = f"/mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_{dset}/rgbd-unsupervised"\n  os.makedirs(dst, exist_ok=True)\n  if not os.path.exists(src): continue\n  for f in os.listdir(src):\n    if not f.endswith(".txt"): continue\n    seq = f.replace(".txt", "")\n    os.makedirs(os.path.join(dst, seq), exist_ok=True)\n    shutil.copy(os.path.join(src, f), os.path.join(dst, seq, f"{seq}_001.txt"))\n    with open(os.path.join(dst, seq, f"{seq}_001_time.value"), "w") as tf: tf.write("0.03\\n" * 5000)\n"INNEREOF"')
run_cmd('/coreflow/venv/bin/python /mnt/task_runtime/create_vot_v56.py')

# Add tracker definition to ini if not exists
run_cmd('grep -q "flextrackv2_v56_depthtrack" /mnt/task_runtime/Depthtrack_workspace/trackers.ini || echo "\n[flextrackv2_v56_depthtrack]\nlabel = flextrackv2_v56_depthtrack\nprotocol = traxpython\ncommand = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot" >> /mnt/task_runtime/Depthtrack_workspace/trackers.ini')

print('4. Evaluating DepthTrack on VOT...')
env_vot = {'HTTP_PROXY':'http://proxy.config.pcp.local:3128', 'HTTPS_PROXY':'http://proxy.config.pcp.local:3128', 'PYTHONPATH':'/mnt/task_runtime'}
run_cmd('cd /mnt/task_runtime/Depthtrack_workspace && /coreflow/venv/bin/vot evaluate --workspace /mnt/task_runtime/Depthtrack_workspace flextrackv2_v56_depthtrack', env_vot)
run_cmd('cd /mnt/task_runtime/Depthtrack_workspace && /coreflow/venv/bin/vot analysis --nocache --workspace /mnt/task_runtime/Depthtrack_workspace flextrackv2_v56_depthtrack', env_vot)

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

print('5. Evaluating RGBT datasets on RGBT Toolkit...')
run_cmd('cd /mnt/task_runtime && /coreflow/venv/bin/python RGBT_toolkit_python/eval_flextrackv2_v56_rgbt_toolkit.py')
print('=== ALL EVALUATIONS DONE ===')
