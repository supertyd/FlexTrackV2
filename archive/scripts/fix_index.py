import sys

def fix_file(path):
    with open(path, 'r') as f: content = f.read()
    content = content.replace('miss_index[seq_name]["data"][frame_idx]', '([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx])')
    with open(path, 'w') as f: f.write(content)

fix_file('/mnt/task_runtime/RGBT_workspace/test_rgbt_mgpus.py')
fix_file('/mnt/task_runtime/RGBT_workspace/test_depthtrack_mgpus.py')
print('Fixed IndexError.')
