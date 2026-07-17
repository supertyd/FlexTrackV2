import os
import sys
import subprocess
from huggingface_hub import snapshot_download

os.environ['http_proxy'] = 'http://proxy.config.pcp.local:3128'
os.environ['https_proxy'] = 'http://proxy.config.pcp.local:3128'
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')  # set via env; do not hardcode

target_dir = '/mnt/task_wrapper/user_output/artifacts'
local_path = os.path.join(target_dir, 'lasher')
repo_id = 'xche32/lasher'

def extract_dataset(local_path, repo_id):
    print(f"Extracting dataset for {repo_id} in {local_path}...")
    part_files = sorted([f for f in os.listdir(local_path) if 'lasher.tar.gz.part.' in f])
    if part_files:
        combined_tar = os.path.join(local_path, 'lasher.tar.gz')
        if not os.path.exists(combined_tar):
            print(f"Combining {len(part_files)} lasher parts...")
            with open(combined_tar, 'wb') as outfile:
                for part in part_files:
                    with open(os.path.join(local_path, part), 'rb') as infile:
                        outfile.write(infile.read())
            print("Combined successfully.")
        tar_file = combined_tar
    else:
        tar_file = os.path.join(local_path, 'lasher.tar.gz')
    
    if os.path.exists(tar_file):
        print(f"Extracting {tar_file} inside {local_path}...")
        res = subprocess.run(['tar', '-xf', tar_file, '-C', local_path], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Successfully extracted {tar_file}")
            try:
                os.remove(tar_file)
                print(f"Cleaned up {tar_file}")
                for part in part_files:
                    os.remove(os.path.join(local_path, part))
                print("Cleaned up lasher part files.")
            except Exception as e:
                print(f"Error cleaning up files: {e}")
        else:
            print(f"Error extracting {tar_file}: {res.stderr}", file=sys.stderr)
    else:
        print(f"Tar file not found: {tar_file}", file=sys.stderr)

print(f"Downloading {repo_id} to {local_path}...")
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        local_dir=local_path,
        local_dir_use_symlinks=False,
        ignore_patterns=['*.git*', 'README.md']
    )
    print(f"Successfully downloaded {repo_id}")
    extract_dataset(local_path, repo_id)
except Exception as e:
    print(f"Error downloading {repo_id}: {e}", file=sys.stderr)
