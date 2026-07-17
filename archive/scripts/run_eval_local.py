import os
import subprocess

def run_cmd(cmd, env_add={}):
    env = os.environ.copy()
    env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
    env.update(env_add)
    subprocess.run(cmd, shell=True, env=env)

print('Running Local evaluation script for LasHeR / RGBT (V56 results)...')
with open('/mnt/task_runtime/evaluate_all_visevent_rgbt_pami_v54.py', 'r') as f:
    code = f.read()

code = code.replace('flextrackv2_b224_54', 'flextrackv2_b224_56')
code = code.replace('/mnt/task_wrapper/user_output/artifacts/lasher/testingset', '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset')
code = code.replace('/mnt/task_wrapper/user_output/artifacts/rgbt234', '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234')

with open('/mnt/task_runtime/evaluate_all_visevent_rgbt_pami_v56.py', 'w') as f:
    f.write(code)

run_cmd('cd /mnt/task_runtime && /coreflow/venv/bin/python evaluate_all_visevent_rgbt_pami_v56.py | grep -E "LasHeR|RGBT234"')
