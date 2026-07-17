import os
from huggingface_hub import snapshot_download

os.environ['http_proxy'] = 'http://proxy.config.pcp.local:3128'
os.environ['https_proxy'] = 'http://proxy.config.pcp.local:3128'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

target_dir = '/mnt/task_runtime/data_missing_modality/'
os.makedirs(target_dir, exist_ok=True)
print('Downloading RGBX-Missing dataset...')
snapshot_download(repo_id='taryya/RGBX-Missing', repo_type='dataset', local_dir=target_dir, local_dir_use_symlinks=False)
print('Download completed.')
