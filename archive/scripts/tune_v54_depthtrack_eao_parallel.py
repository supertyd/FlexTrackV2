import os
import sys
import yaml
import subprocess
import time

# Ensure correct PYTHONPATH
os.environ['PYTHONPATH'] = '/mnt/task_runtime'

yaml_path = 'experiments/flextrackv2/flextrackv2_b224_54.yaml'

def load_yaml():
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(cfg):
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

# Define candidates for DepthTrack
upts = [0.60, 0.65, 0.70]
uphs = [0.90, 0.95]
inters = [20, 25, 30]

candidates = []
for upt in upts:
    for uph in uphs:
        for inter in inters:
            candidates.append({"upt": upt, "uph": uph, "inter": inter})

def run_evaluation_concurrent():
    for dataset in ['depthtrack', 'depthtrack_miss']:
        res_dir = f'/mnt/task_runtime/workspace/results/{dataset}/flextrackv2_b224_54'
        subprocess.run(['rm', '-rf', res_dir])
    
    processes = []
    # Test both missing and complete DepthTrack simultaneously using 8 GPUs
    eval_tasks = [
        ('depthtrack', 'RGBT_workspace/test_depthtrack_mgpus.py', '0,1,2,3', 12, 4),
        ('depthtrack_miss', 'RGBT_workspace/test_depthtrack_mgpus.py', '4,5,6,7', 12, 4)
    ]
    
    for dataset, script, gpus, threads, num_gpus in eval_tasks:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpus
        env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
        cmd = [
            sys.executable, script,
            '--script_name', 'flextrackv2',
            '--dataset_name', dataset,
            '--yaml_name', 'flextrackv2_b224_54',
            '--threads', str(threads),
            '--num_gpus', str(num_gpus),
            '--epoch', '40'
        ]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
    for p in processes:
        p.wait()

def get_metrics():
    env = os.environ.copy()
    env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
    cmd = [sys.executable, 'evaluate_all_visevent_rgbt_pami_v54.py']
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    metrics = []
    # Extract just the DepthTrack blocks
    lines = res.stdout.split('\n')
    capture = False
    for line in lines:
        if "DepthTrack" in line or "DepthTrack_miss" in line:
            capture = True
        if capture and ("=" * 20) in line:
            break
        if capture:
            metrics.append(line)
            
    return "\n".join(metrics)

# Main Loop over candidates
print(f"Starting Grid Search for DepthTrack EAO Optimization... Total Configurations: {len(candidates)}")
best_miss_auc = 0.0
best_cand = None

for idx, param in enumerate(candidates):
    print(f"\n==========================================================")
    print(f"--- Iteration {idx+1}/{len(candidates)} ---")
    print(f"Testing Config: UPT={param['upt']}, UPH={param['uph']}, INTER={param['inter']}")
    
    cfg = load_yaml()
    cfg['TEST']['UPT']['DEPTHTRACK'] = float(param['upt'])
    cfg['TEST']['UPT']['DEPTHTRACK_MISS'] = float(param['upt'])
    cfg['TEST']['UPH']['DEPTHTRACK'] = float(param['uph'])
    cfg['TEST']['UPH']['DEPTHTRACK_MISS'] = float(param['uph'])
    cfg['TEST']['INTER']['DEPTHTRACK'] = int(param['inter'])
    cfg['TEST']['INTER']['DEPTHTRACK_MISS'] = int(param['inter'])
    save_yaml(cfg)
    
    start_time = time.time()
    run_evaluation_concurrent()
    print(f"Tracking finished in {time.time() - start_time:.2f} seconds.")
    
    metrics_out = get_metrics()
    print("\n--- AUC / PR Metrics ---")
    print(metrics_out)
    
    # Run VOT converter and analysis for exact EAO if needed
    cmd_conv = "/coreflow/venv/bin/python /mnt/task_runtime/Depthtrack_workspace/convert_flextrackv2_to_vot.py"
    subprocess.run(cmd_conv, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    cmd_anal = "export HTTP_PROXY=http://proxy.config.pcp.local:3128; export HTTPS_PROXY=http://proxy.config.pcp.local:3128; cd /mnt/task_runtime/Depthtrack_workspace && export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH && /coreflow/venv/bin/vot analysis --nocache --name flextrackv2"
    subprocess.run(cmd_anal, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"\nGrid Search completed!")
