import sys

log_path = '/mnt/task_runtime/v56_eval_final.log'
with open(log_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    if '🔥' in line and ('LasHeR' in line or 'RGBT234' in line):
        print(line.strip())
