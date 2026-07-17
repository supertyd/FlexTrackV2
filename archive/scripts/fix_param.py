import sys

file_path = '/mnt/task_runtime/lib/test/parameter/flextrackv2.py'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace('checkpoint_yaml_name = yaml_name', 
                          'checkpoint_yaml_name = yaml_name\n    if "flextrackv2_b224_56" in yaml_name:\n        checkpoint_yaml_name = "flextrackv2_b224_54"')

with open(file_path, 'w') as f:
    f.write(content)

print('Fixed parameter file to use V54 weights for V56 evaluation.')
