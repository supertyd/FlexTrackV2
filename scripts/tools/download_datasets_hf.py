import os
import sys
import subprocess
from huggingface_hub import snapshot_download

os.environ['http_proxy'] = 'http://proxy.config.pcp.local:3128'
os.environ['https_proxy'] = 'http://proxy.config.pcp.local:3128'
if 'HF_TOKEN' not in os.environ:
    sys.exit('HF_TOKEN environment variable is not set. Set it before running this script.')

target_dir = '/mnt/task_wrapper/user_output/artifacts'
os.makedirs(target_dir, exist_ok=True)

datasets = {
    'xche32/visevent': os.path.join(target_dir, 'visevent'),
    'xche32/lasher': os.path.join(target_dir, 'lasher'),
    'xche32/depthtrack': os.path.join(target_dir, 'depthtrack'),
    'xche32/rgbt234': os.path.join(target_dir, 'rgbt234'),
    'xche32/Depthtrack_workspace': os.path.join(target_dir, 'Depthtrack_workspace'),
}

def extract_dataset(local_path, repo_id):
    print(f"Extracting dataset for {repo_id} in {local_path}...")
    
    # Handle split parts for lasher
    if 'lasher' in repo_id:
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
            # Now untar the combined file
            tar_file = combined_tar
        else:
            print("No split parts found for lasher, checking if combined tar exists...")
            tar_file = os.path.join(local_path, 'lasher.tar.gz')
    else:
        # Normal single tar.gz file
        base_name = repo_id.split('/')[-1]
        tar_file = os.path.join(local_path, f"{base_name}.tar.gz")
    
    if os.path.exists(tar_file):
        print(f"Extracting {tar_file} inside {local_path}...")
        # Use subprocess to run tar as it is much faster than python's tarfile module
        res = subprocess.run(['tar', '-xf', tar_file, '-C', local_path], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Successfully extracted {tar_file}")
            # Clean up the downloaded tar file to save disk space
            try:
                os.remove(tar_file)
                print(f"Cleaned up {tar_file}")
                # Clean up parts if lasher
                if 'lasher' in repo_id:
                    for part in part_files:
                        os.remove(os.path.join(local_path, part))
                    print("Cleaned up lasher part files.")
            except Exception as e:
                print(f"Error cleaning up files: {e}")
        else:
            print(f"Error extracting {tar_file}: {res.stderr}", file=sys.stderr)
    else:
        print(f"Tar file not found: {tar_file}", file=sys.stderr)

failed_repos = []
for repo_id, local_path in datasets.items():
    print(f"Downloading {repo_id} to {local_path}...")
    # Retry: large files (e.g. lasher's 50GB parts) have been observed to
    # truncate mid-transfer through this network's proxy, consistently at
    # the same byte offset -- snapshot_download raises on the resulting
    # size-mismatch, so just retrying the same call a few times is enough
    # to get past a transient truncation.
    downloaded = False
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                local_dir=local_path,
                local_dir_use_symlinks=False,
                ignore_patterns=['*.git*', 'README.md']
            )
            print(f"Successfully downloaded {repo_id}")
            downloaded = True
            break
        except Exception as e:
            print(f"Attempt {attempt}/3 failed downloading {repo_id}: {e}", file=sys.stderr)
    if downloaded:
        extract_dataset(local_path, repo_id)
    else:
        failed_repos.append(repo_id)

if failed_repos:
    print(f"FAILED to download after retries: {failed_repos}", file=sys.stderr)
    sys.exit(1)

print("All downloads and extractions completed!")
