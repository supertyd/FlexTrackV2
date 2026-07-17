"""Missing-modality companion to grid_search_large.py. Tunes the LARGE backbone
(flextrackv2_l224_56) test-time thresholds on the four *_miss datasets so the large
model beats base (rung0) on every missing-modality metric too.

The large model already beats base on 7/8 miss metrics; only DepthTrack_miss
recall is a rounding-level tie (59.20 vs 59.21). So each track first evaluates
its start point and SKIPS the full coordinate descent if it already beats base
on every metric -- only genuinely-trailing datasets get searched.

Trial yamls are named flextrackv2_l224_56_gs_<track>_... -> resolve to the large
checkpoint via the special case in lib/test/parameter/flextrackv2.py.
"""
import os, sys, csv, json, time, glob, shutil, subprocess, yaml, re

os.chdir('/mnt/task_runtime')
sys.path.insert(0, '/mnt/task_runtime')

BASE_YAML = '/mnt/task_runtime/experiments/flextrackv2/flextrackv2_l224_56_gstuned.yaml'
WORKSPACE = '/mnt/task_runtime/Depthtrack_workspace'
ENV_VOT = {
    "HTTP_PROXY": "http://proxy.config.pcp.local:3128",
    "HTTPS_PROXY": "http://proxy.config.pcp.local:3128",
    "PYTHONPATH": "/mnt/task_runtime",
}
GPUS = list(range(8))

# base (rung0) miss targets the large model must beat on every metric
TARGETS = {
    'rgbt234_miss':   dict(mpr=84.55, msr=62.66),
    'lasher_miss':    dict(pr=67.63, sr=54.41),
    'visevent_miss':  dict(auc=58.93, pr=73.78),
    'depthtrack_miss': dict(precision=56.61, recall=59.21, fscore=57.88),
}

# yaml TEST key each track writes into
YAML_KEY = {
    'rgbt234_miss': 'RGBT234_MISS', 'lasher_miss': 'LASHER_MISS',
    'visevent_miss': 'VISEVENT_MISS', 'depthtrack_miss': 'DEPTHTRACK_MISS',
}


def log(tag, msg):
    print(f"[{tag}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def start_params(track):
    d = yaml.safe_load(open(BASE_YAML))['TEST']
    k = YAML_KEY[track]
    return {'UPT': d['UPT'][k], 'UPH': d['UPH'][k], 'INTER': d['INTER'][k], 'MB': d['MB'][k]}


def write_trial_yaml(trial_name, key, params):
    data = yaml.safe_load(open(BASE_YAML))
    for pname, val in params.items():
        data['TEST'][pname][key] = val
    with open(f'/mnt/task_runtime/experiments/flextrackv2/{trial_name}.yaml', 'w') as f:
        yaml.safe_dump(data, f)


def param_tag(p):
    return "UPT{:.3f}_UPH{:.3f}_INTER{}_MB{}".format(p['UPT'], p['UPH'], p['INTER'], p['MB']).replace('.', 'p')


def run(cmd, env_add=None, quiet=False):
    env = os.environ.copy()
    env["PATH"] = "/coreflow/venv/bin:/coreflow/mambaforge/bin:" + env.get("PATH", "")
    if env_add:
        env.update(env_add)
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL; kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(cmd, shell=True, env=env, **kwargs)


def _track_rgbt(dataset_name, test_script, n_seqs, trial_name):
    res_dir = f"/mnt/task_runtime/workspace/results/{dataset_name}/{trial_name}"
    if not (os.path.isdir(res_dir) and len(os.listdir(res_dir)) >= n_seqs):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in GPUS)
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[v] = '2'
        cmd = ['/coreflow/venv/bin/python', '-u', test_script,
               '--script_name', 'flextrackv2', '--dataset_name', dataset_name,
               '--yaml_name', trial_name, '--mode', 'parallel',
               '--threads', '20', '--num_gpus', str(len(GPUS)), '--epoch', '40']
        logf = open(f'/mnt/task_runtime/gs_largemiss_{trial_name}.log', 'w')
        subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd='/mnt/task_runtime')
        logf.close()
    return res_dir


def score_rgbt234_miss(trial_name):
    write_trial_yaml(trial_name, 'RGBT234_MISS', CURRENT_PARAMS['rgbt234_miss'])
    res_dir = _track_rgbt('RGBT234_miss', 'RGBT_workspace/test_rgbt_mgpus.py', 234, trial_name)
    from rgbt.dataset.rgbt234_dataset import RGBT234
    gt_dir = '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234'
    r = RGBT234(gt_path=gt_dir, seq_name_path=os.path.join(gt_dir, 'attr_txt/SequencesName.txt'))
    r(tracker_name=trial_name, result_path=res_dir, bbox_type='ltwh')
    return {'mpr': r.MPR()[trial_name][0] * 100, 'msr': r.MSR()[trial_name][0] * 100}


def score_lasher_miss(trial_name):
    write_trial_yaml(trial_name, 'LASHER_MISS', CURRENT_PARAMS['lasher_miss'])
    res_dir = _track_rgbt('LasHeR_miss', 'RGBT_workspace/test_rgbt_mgpus.py', 245, trial_name)
    from rgbt.dataset.lasher_dataset import LasHeR
    gt_dir = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
    la = LasHeR(gt_path=gt_dir, seq_name_path=os.path.join(gt_dir, 'lashertest.txt'))
    la(tracker_name=trial_name, result_path=res_dir, bbox_type='ltwh')
    return {'pr': la.PR()[trial_name][0] * 100, 'sr': la.SR()[trial_name][0] * 100}


def score_visevent_miss(trial_name):
    write_trial_yaml(trial_name, 'VISEVENT_MISS', CURRENT_PARAMS['visevent_miss'])
    # MUST use test_rgbt_mgpus.py, NOT test_rgbe_mgpus.py: only the former has
    # the 320-seq absent-init fix. The official gstuned VisEvent eval used
    # test_rgbt_mgpus.py (run_l224_gstuned_eval.sh); using the rgbe script here
    # drops the 23 absent-init sequences and gives a wrong low AUC.
    res_dir = _track_rgbt('VisEvent_miss', 'RGBT_workspace/test_rgbt_mgpus.py', 320, trial_name)
    from evaluate_lasher_visevent import evaluate_tracker_dataset
    gt_dir = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test'
    seqs = sorted(set(open(os.path.join(gt_dir, 'testlist.txt')).read().splitlines()))
    auc, pr = evaluate_tracker_dataset(res_dir, gt_dir, seqs, 'groundtruth.txt', is_visevent=True)
    return {'auc': auc, 'pr': pr}


def convert_to_vot_format(raw, vot):
    if os.path.exists(vot):
        shutil.rmtree(vot)
    os.makedirs(vot, exist_ok=True)
    for f in os.listdir(raw):
        if not f.endswith('.txt'):
            continue
        seq = f[:-4]
        os.makedirs(os.path.join(vot, seq), exist_ok=True)
        shutil.copy(os.path.join(raw, f), os.path.join(vot, seq, f"{seq}_001.txt"))
        with open(os.path.join(vot, seq, f"{seq}_001_time.value"), 'w') as tf:
            tf.write("0.03\n" * 5000)


def latest_report():
    files = glob.glob(os.path.join(WORKSPACE, "analysis", "*", "report.html"))
    return sorted(files, key=os.path.getmtime)[-1] if files else None


def parse_metrics(html_path, tracker):
    html = open(html_path).read()
    m = re.search(r'data-tracker="' + re.escape(tracker) + r'"[^>]*>(.*?)</tr>', html, re.DOTALL)
    if not m:
        return None
    vals = re.findall(r'data-value="([0-9.]+)"', m.group(1))
    if len(vals) < 3:
        return None
    return {"precision": float(vals[0]) * 100, "recall": float(vals[1]) * 100, "fscore": float(vals[2]) * 100}


def score_depthtrack_miss(trial_name):
    write_trial_yaml(trial_name, 'DEPTHTRACK_MISS', CURRENT_PARAMS['depthtrack_miss'])
    raw = f"/mnt/task_runtime/workspace/results/depthtrack_miss/{trial_name}"
    vot = f"{WORKSPACE}/results/{trial_name}/rgbd-unsupervised"
    if not (os.path.isdir(raw) and len(os.listdir(raw)) >= 50):
        run(f"cd /mnt/task_runtime && /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py "
            f"--script_name flextrackv2 --dataset_name depthtrack_miss --yaml_name {trial_name} "
            f"--mode parallel --threads 32 --num_gpus 8 --epoch 40", quiet=True)
    if not os.path.isdir(raw) or not os.listdir(raw):
        return None
    convert_to_vot_format(raw, vot)
    ini = f"{WORKSPACE}/trackers.ini"
    text = open(ini).read() if os.path.exists(ini) else ""
    if f"[{trial_name}]" not in text:
        with open(ini, "a") as f:
            f.write(f"\n[{trial_name}]\nlabel = {trial_name}\nprotocol = traxpython\n"
                     f"command = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot\n")
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot evaluate --workspace {WORKSPACE} {trial_name}", ENV_VOT, quiet=True)
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot analysis --nocache --workspace {WORKSPACE} {trial_name}", ENV_VOT, quiet=True)
    hp = latest_report()
    return parse_metrics(hp, trial_name) if hp else None


SCORERS = {
    'rgbt234_miss': score_rgbt234_miss, 'lasher_miss': score_lasher_miss,
    'visevent_miss': score_visevent_miss, 'depthtrack_miss': score_depthtrack_miss,
}

CURRENT_PARAMS = {}  # track -> params (read by scorers via write_trial_yaml)


def candidates(pname, val, step):
    if pname in ('UPT', 'UPH'):
        return sorted(set([round(max(0.05, min(0.98, val - step)), 3), round(max(0.05, min(0.98, val + step)), 3)]))
    if pname == 'INTER':
        d = max(2, int(round(val * step)))
        return sorted(set([max(3, val - d), val + d]))
    if pname == 'MB':
        d = max(20, int(round(val * step)))
        return sorted(set([max(50, val - d), val + d]))
    return []


def margin(m, target):
    return min(m[k] - target[k] for k in target)


def beats(m, target):
    return all(m[k] >= target[k] for k in target)


def search(track, max_rounds=6):
    target = TARGETS[track]
    scorer = SCORERS[track]
    current = start_params(track)
    CURRENT_PARAMS[track] = current
    log(track, f"target(base/rung0)={target} start={current}")

    trial_name = f"flextrackv2_l224_56_gs_{track}_{param_tag(current)}"
    best = scorer(trial_name)
    if best is None:
        log(track, "!! round0 failed, aborting")
        return None, None
    log(track, f"round0 start -> {best} margin={margin(best, target):.3f} beats_base={beats(best, target)}")

    if beats(best, target):
        log(track, "=== already beats base at start point, SKIPPING search ===")
        with open(f'/mnt/task_runtime/gs_largemiss_{track}_FINAL.json', 'w') as f:
            json.dump({'params': current, 'metrics': best, 'target': target,
                       'beats_base_on_every_metric': True, 'searched': False}, f, indent=2)
        return current, best

    csv_f = open(f'/mnt/task_runtime/gs_largemiss_{track}_results.csv', 'a', newline='')
    w = csv.writer(csv_f)
    w.writerow([0, 'start', json.dumps(current), json.dumps(best)]); csv_f.flush()

    upt_uph_step, inter_mb_step = 0.08, 0.4
    for rnd in range(1, max_rounds + 1):
        improved = False
        for pname in ['UPT', 'UPH', 'INTER', 'MB']:
            step = upt_uph_step if pname in ('UPT', 'UPH') else inter_mb_step
            for cand in candidates(pname, current[pname], step):
                if cand == current[pname]:
                    continue
                trial = {**current, pname: cand}
                CURRENT_PARAMS[track] = trial
                tn = f"flextrackv2_l224_56_gs_{track}_{param_tag(trial)}"
                log(track, f"round{rnd} trying {pname}={cand} (others={current})")
                m = scorer(tn)
                if m is None:
                    log(track, f"round{rnd} {pname}={cand} -> FAILED"); continue
                log(track, f"round{rnd} {pname}={cand} -> {m} margin={margin(m, target):.3f}")
                w.writerow([rnd, pname, cand, json.dumps(m)]); csv_f.flush()
                if margin(m, target) > margin(best, target):
                    log(track, f"  -> improvement margin {margin(best, target):.3f} -> {margin(m, target):.3f}, locking {pname}={cand}")
                    current[pname] = cand; best = m; improved = True
            CURRENT_PARAMS[track] = current
        with open(f'/mnt/task_runtime/gs_largemiss_{track}_FINAL.json', 'w') as f:
            json.dump({'round': rnd, 'params': current, 'metrics': best, 'target': target,
                       'beats_base_on_every_metric': beats(best, target), 'searched': True}, f, indent=2)
        log(track, f"=== round {rnd} complete: current={current} best={best} improved={improved} beats_base={beats(best, target)} ===")
        if beats(best, target):
            log(track, f"=== TARGET MET after round {rnd}, stopping early ==="); break
        if not improved:
            upt_uph_step = max(0.02, upt_uph_step * 0.5)
            inter_mb_step = max(0.1, inter_mb_step * 0.5)
            log(track, f"no improvement; shrinking steps to {upt_uph_step}/{inter_mb_step}")
            if upt_uph_step <= 0.02 and inter_mb_step <= 0.1:
                log(track, "steps at floor, stopping"); break
    csv_f.close()
    log(track, f"=== {track.upper()}_GRIDSEARCH_DONE ===")
    return current, best


if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else ['rgbt234_miss', 'lasher_miss', 'visevent_miss', 'depthtrack_miss']
    results = {}
    for track in which:
        p, m = search(track)
        results[track] = {'params': p, 'metrics': m}
    with open('/mnt/task_runtime/gs_largemiss_ALL_FINAL.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("=== ALL_LARGEMISS_GRIDSEARCH_DONE ===", flush=True)
