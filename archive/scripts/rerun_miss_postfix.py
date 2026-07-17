import os, sys, time, shutil, subprocess
sys.path.insert(0, '/mnt/task_runtime')
sys.path.insert(0, '/mnt/task_runtime/RGBT_toolkit_python/src')
os.chdir('/mnt/task_runtime')

YAML = 'flextrackv2_b224_56'

def log(m):
    print(f"[rerun_miss] {time.strftime('%H:%M:%S')} {m}", flush=True)

def run(cmd, quiet=True):
    env = os.environ.copy()
    env["PATH"] = "/coreflow/venv/bin:/coreflow/mambaforge/bin:" + env.get("PATH", "")
    print(f">>> {cmd}", flush=True)
    kw = {}
    if quiet:
        kw['stdout'] = subprocess.DEVNULL; kw['stderr'] = subprocess.DEVNULL
    subprocess.run(cmd, shell=True, env=env, **kw)

def track(dataset_name, script):
    res_dir = f"/mnt/task_runtime/workspace/results/{dataset_name}/{YAML}"
    shutil.rmtree(res_dir, ignore_errors=True)
    cmd = (f"cd /mnt/task_runtime && /coreflow/venv/bin/python {script} "
           f"--script_name flextrackv2 --dataset_name {dataset_name} --yaml_name {YAML} "
           f"--mode parallel --threads 32 --num_gpus 8 --epoch 40")
    run(cmd)
    return res_dir

def eval_lasher(res_dir, name):
    from rgbt.dataset.lasher_dataset import LasHeR
    gt = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
    ds = LasHeR(gt_path=gt, seq_name_path=gt+'/lashertest.txt')
    ds(tracker_name=name, result_path=res_dir, bbox_type='ltwh')
    return {'PR': ds.PR()[name][0]*100, 'AUC': ds.SR()[name][0]*100}

def eval_rgbt234(res_dir, name):
    from rgbt.dataset.rgbt234_dataset import RGBT234
    gt = '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234'
    ds = RGBT234(gt_path=gt, seq_name_path=gt+'/attr_txt/SequencesName.txt')
    ds(tracker_name=name, result_path=res_dir, bbox_type='ltwh')
    return {'MPR': ds.MPR()[name][0]*100, 'MSR': ds.MSR()[name][0]*100}

def eval_visevent(res_dir, name):
    from evaluate_lasher_visevent import evaluate_tracker_dataset
    gt = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test'
    seqs = sorted(set(open(os.path.join(gt,'testlist.txt')).read().splitlines()))
    auc, pr = evaluate_tracker_dataset(res_dir, gt, seqs, 'groundtruth.txt', is_visevent=True)
    return {'AUC': auc, 'PR': pr}

JOBS = [
    ('LasHeR_miss',  'RGBT_workspace/test_rgbt_mgpus.py', eval_lasher,   'FlexTrack', {'PR':65.11,'AUC':52.34}),
    ('RGBT234_miss', 'RGBT_workspace/test_rgbt_mgpus.py', eval_rgbt234,  'FlexTrack', {'MPR':84.07,'MSR':62.73}),
    ('VisEvent_miss','RGBE_workspace/test_rgbe_mgpus.py', eval_visevent, 'FlexTrack', {'AUC':58.29,'PR':73.74}),
]

if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else [j[0] for j in JOBS]
    summary = []
    for dsname, script, evalfn, flexdir, flextgt in JOBS:
        if dsname not in which:
            continue
        log(f"=== {dsname}: tracking (post-fix) ===")
        res_dir = track(dsname, script)
        n = len([f for f in os.listdir(res_dir) if f.endswith('.txt')]) if os.path.isdir(res_dir) else 0
        log(f"{dsname}: produced {n} result files")
        mci = evalfn(res_dir, YAML)
        flex_res = f"/mnt/task_runtime/workspace/results/{dsname}/{flexdir}"
        try:
            flex = evalfn(flex_res, flexdir) if os.path.isdir(flex_res) else flextgt
        except Exception as e:
            log(f"{dsname}: FlexTrack recompute failed ({e}), using recorded numbers")
            flex = flextgt
        log(f"{dsname}: FlexTrackV2(postfix)={mci}  FlexTrack={flex}")
        summary.append((dsname, mci, flex))
    log("===== SUMMARY (post-fix) =====")
    for dsname, mci, flex in summary:
        log(f"{dsname}: FlexTrackV2={ {k:round(v,2) for k,v in mci.items()} }  FlexTrack={ {k:round(v,2) for k,v in flex.items()} }")
    log("===== RERUN_MISS_DONE =====")
