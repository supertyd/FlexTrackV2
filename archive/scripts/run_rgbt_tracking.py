import os
import subprocess

def run_cmd(cmd, env_add={}):
    env = os.environ.copy()
    env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
    env.update(env_add)
    print(f'Running: {cmd}')
    subprocess.run(cmd, shell=True, env=env)

cmds = [
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=0,1 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224_56 --mode parallel --threads 8 --num_gpus 2 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=2,3 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 8 --num_gpus 2 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=4,5 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224_56 --mode parallel --threads 8 --num_gpus 2 --epoch 40',
    'cd /mnt/task_runtime && CUDA_VISIBLE_DEVICES=6,7 /coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 8 --num_gpus 2 --epoch 40'
]
ps = [subprocess.Popen(c, shell=True) for c in cmds]
for p in ps: p.wait()
print('RGBT Tracking Done.')
