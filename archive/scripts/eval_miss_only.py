import os, sys
sys.path.insert(0, '/mnt/task_runtime')
sys.path.insert(0, '/mnt/task_runtime/RGBT_toolkit_python/src')
os.chdir('/mnt/task_runtime')
YAML = 'flextrackv2_b224_56'

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

JOBS = {
    'LasHeR_miss':  (eval_lasher,   {'PR':65.11,'AUC':52.34}),
    'RGBT234_miss': (eval_rgbt234,  {'MPR':84.07,'MSR':62.73}),
    'VisEvent_miss':(eval_visevent, {'AUC':58.29,'PR':73.74}),
}

for dsname in (sys.argv[1:] or list(JOBS)):
    evalfn, flextgt = JOBS[dsname]
    mci = evalfn(f"/mnt/task_runtime/workspace/results/{dsname}/{YAML}", YAML)
    flex_dir = f"/mnt/task_runtime/workspace/results/{dsname}/FlexTrack"
    try:
        flex = evalfn(flex_dir, 'FlexTrack') if os.path.isdir(flex_dir) else flextgt
    except Exception as e:
        flex = flextgt
    print(f"{dsname}: FlexTrackV2={ {k:round(v,2) for k,v in mci.items()} }  FlexTrack={ {k:round(v,2) for k,v in flex.items()} }", flush=True)
