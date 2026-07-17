import sys

path = '/mnt/task_runtime/lib/test/parameter/flextrackv2.py'
with open(path, 'r') as f:
    content = f.read()

# Make sure the evaluator finds V56 checkpoint correctly!
if 'flextrackv2_b224_56' not in content:
    content = content.replace(
        'if "flextrackv2_b224_54" in yaml_name:',
        'if "flextrackv2_b224_54" in yaml_name or "flextrackv2_b224_56" in yaml_name:'
    )

with open(path, 'w') as f:
    f.write(content)
print('Fixed checkpoint loader parameter file.')






