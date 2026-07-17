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
        yaml.safe_dump(cfg, f)

# Grid candidates to evaluate
candidates = [
    # Candidate 1: Increase update stability across all sets
    {'rgbt234_upt': 0.45, 'rgbt234_uph': 0.90, 'rgbt234_inter': 15,
     'lasher_upt': 0.50, 'lasher_uph': 0.95, 'lasher_inter': 10,
     'visevent_upt': 0.80, 'visevent_uph': 0.95, 'visevent_inter': 50,
     'rgbt234_miss_upt': 0.45, 'rgbt234_miss_uph': 0.90, 'rgbt234_miss_inter': 15},
     
    # Candidate 2: Allow slightly lower confidence updates but larger interval to prevent drift
    {'rgbt234_upt': 0.40, 'rgbt234_uph': 0.85, 'rgbt234_inter': 20,
     'lasher_upt': 0.45, 'lasher_uph': 0.90, 'lasher_inter': 15,
     'visevent_upt': 0.75, 'visevent_uph': 0.90, 'visevent_inter': 60,
     'rgbt234_miss_upt': 0.40, 'rgbt234_miss_uph': 0.90, 'rgbt234_miss_inter': 20},

    # Candidate 3: Conservative high-fidelity settings
    {'rgbt234_upt': 0.45, 'rgbt234_uph': 0.90, 'rgbt234_inter': 25,
     'lasher_upt': 0.50, 'lasher_uph': 0.95, 'lasher_inter': 20,
     'visevent_upt': 0.80, 'visevent_uph': 0.95, 'visevent_inter': 70,
     'rgbt234_miss_upt': 0.45, 'rgbt234_miss_uph': 0.95, 'rgbt234_miss_inter': 25}
]

def run_all_evaluations_concurrent():
    # Clear result folders first
    for dataset in ['RGBT234', 'RGBT234_miss', 'LasHeR', 'LasHeR_miss', 'VisEvent']:
        res_dir = f'/mnt/task_runtime/workspace/results/{dataset}/flextrackv2_b224_54'
        subprocess.run(['rm', '-rf', res_dir])
    
    processes = []
    
    # We use threads = 12 to prevent CPU core oversubscription (12 * 5 = 60 threads total on 96 cores)
    eval_tasks = [
        ('RGBT234', 'RGBT_workspace/test_rgbt_mgpus.py', '0,1,2,3,4,5,6,7', 12),
        ('RGBT234_miss', 'RGBT_workspace/test_rgbt_mgpus.py', '0,1,2,3,4,5,6,7', 12),
        ('LasHeR', 'RGBT_workspace/test_rgbt_mgpus.py', '0,1,2,3,4,5,6,7', 12),
        ('LasHeR_miss', 'RGBT_workspace/test_rgbt_mgpus.py', '0,1,2,3,4,5,6,7', 12),
        ('VisEvent', 'RGBE_workspace/test_rgbe_mgpus.py', '0,1,2,3,4,5,6,7', 12)
    ]
    
    for dataset, script, gpus, threads in eval_tasks:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpus
        # Inject correct binary paths to PATH environment variable
        env['PATH'] = '/coreflow/venv/bin:/coreflow/mambaforge/bin:' + env.get('PATH', '')
        cmd = [
            sys.executable, script,
            '--script_name', 'flextrackv2',
            '--dataset_name', dataset,
            '--yaml_name', 'flextrackv2_b224_54',
            '--threads', str(threads),
            '--num_gpus', '8',
            '--epoch', '40'
        ]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
    print(f"Spawned {len(processes)} concurrent evaluation tasks with {12*5} total threads. Zero oversubscription!")
    
    for p in processes:
        p.wait()
    print("All concurrent tasks finished!")

def evaluate_all_metrics():
    # 1. RGBT toolkit evaluation
    cmd_rgbt = [sys.executable, 'RGBT_toolkit_python/eval_flextrackv2_vs_flextrack.py']
    res_rgbt = subprocess.run(cmd_rgbt, capture_output=True, text=True)
    
    # 2. VisEvent evaluation
    cmd_visevent = [sys.executable, 'evaluate_all_visevent_rgbt_pami_v54.py']
    res_visevent = subprocess.run(cmd_visevent, capture_output=True, text=True)
    
    return res_rgbt.stdout, res_visevent.stdout

# Main Loop over candidates
for idx, param in enumerate(candidates):
    print(f"\n==========================================================")
    print(f"--- Iteration {idx+1}/{len(candidates)} (Highly Saturated Parallel Tuning) ---")
    print(f"==========================================================")
    print(f"RGBT234: UPT={param['rgbt234_upt']}, UPH={param['rgbt234_uph']}, INTER={param['rgbt234_inter']}")
    print(f"LasHeR: UPT={param['lasher_upt']}, UPH={param['lasher_uph']}, INTER={param['lasher_inter']}")
    print(f"VisEvent: UPT={param['visevent_upt']}, UPH={param['visevent_uph']}, INTER={param['visevent_inter']}")
    print(f"RGBT234_miss: UPT={param['rgbt234_miss_upt']}, UPH={param['rgbt234_miss_uph']}, INTER={param['rgbt234_miss_inter']}")
    
    cfg = load_yaml()
    # Update UPT
    cfg['TEST']['UPT']['RGBT234'] = param['rgbt234_upt']
    cfg['TEST']['UPT']['LASHER'] = param['lasher_upt']
    cfg['TEST']['UPT']['VISEVENT'] = param['visevent_upt']
    cfg['TEST']['UPT']['RGBT234_MISS'] = param['rgbt234_miss_upt']
    
    # Update UPH
    cfg['TEST']['UPH']['RGBT234'] = param['rgbt234_uph']
    cfg['TEST']['UPH']['LASHER'] = param['lasher_uph']
    cfg['TEST']['UPH']['VISEVENT'] = param['visevent_uph']
    cfg['TEST']['UPH']['RGBT234_MISS'] = param['rgbt234_miss_uph']
    
    # Update INTER
    cfg['TEST']['INTER']['RGBT234'] = param['rgbt234_inter']
    cfg['TEST']['INTER']['LASHER'] = param['lasher_inter']
    cfg['TEST']['INTER']['VISEVENT'] = param['visevent_inter']
    cfg['TEST']['INTER']['RGBT234_MISS'] = param['rgbt234_miss_inter']
    
    save_yaml(cfg)
    
    start_time = time.time()
    run_all_evaluations_concurrent()
    print(f"Iteration {idx+1} run took {time.time() - start_time:.2f} seconds.")
    
    rgbt_out, visevent_out = evaluate_all_metrics()
    print("\n--- RGBT Toolkit Output ---")
    print(rgbt_out)
    print("\n--- VisEvent Output ---")
    print(visevent_out)

print("Highly Saturated Parallel Tuning completed successfully!")
