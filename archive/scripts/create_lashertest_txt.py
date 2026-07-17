import os

path = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
seqs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('.')]
with open(os.path.join(path, 'lashertest.txt'), 'w') as f:
    for s in sorted(seqs):
        f.write(s + '\n')
print('lashertest.txt created.')

path2 = '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234'
seqs2 = [f for f in os.listdir(path2) if os.path.isdir(os.path.join(path2, f)) and not f.startswith('.')]
attr_dir = os.path.join(path2, 'attr_txt')
os.makedirs(attr_dir, exist_ok=True)
with open(os.path.join(attr_dir, 'SequencesName.txt'), 'w') as f:
    for s in sorted(seqs2):
        if s != 'attr_txt':
            f.write(s + '\n')
print('SequencesName.txt created.')
