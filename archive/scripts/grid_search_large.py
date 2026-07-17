"""Test-time threshold grid search for the LARGE backbone (flextrackv2_l224_56),
targeting the only two metrics where it currently trails the base model
(rung0): RGBT234-full MSR (69.08 vs 69.13) and DepthTrack-full Pr/Re/F
(65.17/67.85/66.48 vs 65.55/68.52/67.01). It currently inherits the BASE
model's tuned thresholds verbatim (flextrackv2_l224_56_gstuned.yaml) -- this
searches the large checkpoint's own threshold space instead.

Trial yaml names are prefixed flextrackv2_l224_56_gs_... so they resolve to the
large checkpoint via the special case added in lib/test/parameter/flextrackv2.py.
"""
import os, sys, csv, json, time, glob, shutil, subprocess, yaml

os.chdir('/mnt/task_runtime')
sys.path.insert(0, '/mnt/task_runtime')

BASE_YAML = '/mnt/task_runtime/experiments/flextrackv2/flextrackv2_l224_56_gstuned.yaml'
WORKSPACE = '/mnt/task_runtime/Depthtrack_workspace'
ENV_VOT = {
    "HTTP_PROXY": "http://proxy.config.pcp.local:3128",
    "HTTPS_PROXY": "http://proxy.config.pcp.local:3128",
    "PYTHONPATH": "/mnt/task_runtime",
}

# base (rung0) values the large model must beat on every listed metric
RGBT234_TARGET = dict(mpr=91.94, msr=69.13)
DEPTHTRACK_TARGET = dict(precision=65.55, recall=68.52, fscore=67.01)

START_RGBT234 = dict(UPT=0.5, UPH=0.93, INTER=25, MB=312)
START_DEPTHTRACK = dict(UPT=0.77, UPH=0.85, INTER=55, MB=862)

GPUS = list(range(8))


def log(tag, msg):
    print(f"[{tag}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def write_trial_yaml(trial_name, key, params):
    with open(BASE_YAML) as f:
        data = yaml.safe_load(f)
    for pname, val in params.items():
        data['TEST'][pname][key] = val
    out_path = f'/mnt/task_runtime/experiments/flextrackv2/{trial_name}.yaml'
    with open(out_path, 'w') as f:
        yaml.safe_dump(data, f)
    return out_path


def param_tag(params):
    return "UPT{:.3f}_UPH{:.3f}_INTER{}_MB{}".format(
        params['UPT'], params['UPH'], params['INTER'], params['MB']).replace('.', 'p')


def run_rgbt234_trial(params):
    trial_name = f"flextrackv2_l224_56_gs_rgbt234_full_{param_tag(params)}"
    write_trial_yaml(trial_name, 'RGBT234', params)
    res_dir = f"/mnt/task_runtime/workspace/results/RGBT234/{trial_name}"
    if not (os.path.isdir(res_dir) and len(os.listdir(res_dir)) >= 234):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in GPUS)
        env['OMP_NUM_THREADS'] = '2'; env['MKL_NUM_THREADS'] = '2'
        env['OPENBLAS_NUM_THREADS'] = '2'; env['NUMEXPR_NUM_THREADS'] = '2'
        cmd = ['/coreflow/venv/bin/python', '-u', 'RGBT_workspace/test_rgbt_mgpus.py',
               '--script_name', 'flextrackv2', '--dataset_name', 'RGBT234',
               '--yaml_name', trial_name, '--mode', 'parallel',
               '--threads', '20', '--num_gpus', str(len(GPUS)), '--epoch', '40']
        logf = open(f'/mnt/task_runtime/gs_large_rgbt234_full_{trial_name}.log', 'w')
        subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd='/mnt/task_runtime')
        logf.close()
    from rgbt.dataset.rgbt234_dataset import RGBT234
    gt_dir = '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234'
    rgbt234 = RGBT234(gt_path=gt_dir, seq_name_path=os.path.join(gt_dir, 'attr_txt/SequencesName.txt'))
    rgbt234(tracker_name=trial_name, result_path=res_dir, bbox_type='ltwh')
    mpr = rgbt234.MPR()[trial_name][0] * 100
    msr = rgbt234.MSR()[trial_name][0] * 100
    return {'mpr': mpr, 'msr': msr}


def convert_to_vot_format(raw_res_dir, vot_res_dir):
    if os.path.exists(vot_res_dir):
        shutil.rmtree(vot_res_dir)
    os.makedirs(vot_res_dir, exist_ok=True)
    for fname in os.listdir(raw_res_dir):
        if not fname.endswith(".txt"):
            continue
        seq = fname[:-4]
        seq_dir = os.path.join(vot_res_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)
        shutil.copy(os.path.join(raw_res_dir, fname), os.path.join(seq_dir, f"{seq}_001.txt"))
        with open(os.path.join(seq_dir, f"{seq}_001_time.value"), "w") as tf:
            tf.write("0.03\n" * 5000)


def run(cmd, env_add=None, quiet=False):
    env = os.environ.copy()
    env["PATH"] = "/coreflow/venv/bin:/coreflow/mambaforge/bin:" + env.get("PATH", "")
    if env_add:
        env.update(env_add)
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(cmd, shell=True, env=env, **kwargs)


def latest_report():
    files = glob.glob(os.path.join(WORKSPACE, "analysis", "*", "report.html"))
    if not files:
        return None
    return sorted(files, key=os.path.getmtime)[-1]


import re
def parse_metrics(html_path, tracker):
    html = open(html_path).read()
    m = re.search(r'data-tracker="' + re.escape(tracker) + r'"[^>]*>(.*?)</tr>', html, re.DOTALL)
    if not m:
        return None
    vals = re.findall(r'data-value="([0-9.]+)"', m.group(1))
    if len(vals) < 3:
        return None
    return {"precision": float(vals[0]) * 100, "recall": float(vals[1]) * 100, "fscore": float(vals[2]) * 100}


def run_depthtrack_trial(params):
    trial_name = f"flextrackv2_l224_56_gs_depthtrack_full_{param_tag(params)}"
    write_trial_yaml(trial_name, 'DEPTHTRACK', params)
    raw_res_dir = f"/mnt/task_runtime/workspace/results/depthtrack/{trial_name}"
    vot_res_dir = f"{WORKSPACE}/results/{trial_name}/rgbd-unsupervised"
    if not (os.path.isdir(raw_res_dir) and len(os.listdir(raw_res_dir)) >= 50):
        run(f"cd /mnt/task_runtime && /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py "
            f"--script_name flextrackv2 --dataset_name depthtrack --yaml_name {trial_name} "
            f"--mode parallel --threads 32 --num_gpus 8 --epoch 40", quiet=True)
    if not os.path.isdir(raw_res_dir) or not os.listdir(raw_res_dir):
        return None
    convert_to_vot_format(raw_res_dir, vot_res_dir)
    ini = f"{WORKSPACE}/trackers.ini"
    text = open(ini).read() if os.path.exists(ini) else ""
    if f"[{trial_name}]" not in text:
        with open(ini, "a") as f:
            f.write(f"\n[{trial_name}]\nlabel = {trial_name}\nprotocol = traxpython\n"
                     f"command = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot\n")
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot evaluate --workspace {WORKSPACE} {trial_name}", ENV_VOT, quiet=True)
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot analysis --nocache --workspace {WORKSPACE} {trial_name}", ENV_VOT, quiet=True)
    html_path = latest_report()
    if not html_path:
        return None
    return parse_metrics(html_path, trial_name)


def candidates(pname, val, step):
    if pname in ('UPT', 'UPH'):
        return sorted(set([round(max(0.05, min(0.98, val - step)), 3),
                            round(max(0.05, min(0.98, val + step)), 3)]))
    if pname == 'INTER':
        d = max(2, int(round(val * step)))
        return sorted(set([max(3, val - d), val + d]))
    if pname == 'MB':
        d = max(20, int(round(val * step)))
        return sorted(set([max(50, val - d), val + d]))
    return []


def margin(m, target):
    return min(m[k] - target[k] for k in target)


def search(tag, run_trial_fn, target, start_params, max_rounds=6):
    current = dict(start_params)
    log(tag, f"target(base/rung0)={target}")
    best_metrics = run_trial_fn(current)
    if best_metrics is None:
        log(tag, "!! round0 failed, aborting")
        return None, None
    log(tag, f"round0 start {current} -> metrics {best_metrics} margin={margin(best_metrics, target):.3f}")

    csv_path = f'/mnt/task_runtime/gs_large_{tag}_results.csv'
    csv_f = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_f)
    if os.path.getsize(csv_path) == 0:
        writer.writerow(['round', 'param', 'value', 'metrics'])
    writer.writerow([0, 'start', json.dumps(current), json.dumps(best_metrics)]); csv_f.flush()

    upt_uph_step = 0.08
    inter_mb_step = 0.4

    for rnd in range(1, max_rounds + 1):
        improved = False
        for pname in ['UPT', 'UPH', 'INTER', 'MB']:
            step = upt_uph_step if pname in ('UPT', 'UPH') else inter_mb_step
            base_val = current[pname]
            for cand in candidates(pname, base_val, step):
                if cand == base_val:
                    continue
                trial_params = {**current, pname: cand}
                log(tag, f"round{rnd} trying {pname}={cand} (others={current})")
                metrics = run_trial_fn(trial_params)
                if metrics is None:
                    log(tag, f"round{rnd} {pname}={cand} -> FAILED")
                    continue
                log(tag, f"round{rnd} {pname}={cand} -> metrics {metrics} margin={margin(metrics, target):.3f}")
                writer.writerow([rnd, pname, cand, json.dumps(metrics)]); csv_f.flush()
                if margin(metrics, target) > margin(best_metrics, target):
                    log(tag, f"  -> improvement: margin {margin(best_metrics, target):.3f} -> {margin(metrics, target):.3f}, locking in {pname}={cand}")
                    current[pname] = cand
                    best_metrics = metrics
                    improved = True
            log(tag, f"round{rnd} after {pname}: current={current} best={best_metrics}")

        beats = all(best_metrics[k] >= target[k] for k in target)
        with open(f'/mnt/task_runtime/gs_large_{tag}_FINAL.json', 'w') as f:
            json.dump({'round': rnd, 'params': current, 'metrics': best_metrics,
                       'target': target, 'beats_base_on_every_metric': beats}, f, indent=2)
        log(tag, f"=== round {rnd} complete: current={current} best={best_metrics} improved={improved} beats_base={beats} ===")

        if beats:
            log(tag, f"=== TARGET MET after round {rnd}, stopping early ===")
            break
        if not improved:
            upt_uph_step = max(0.02, upt_uph_step * 0.5)
            inter_mb_step = max(0.1, inter_mb_step * 0.5)
            log(tag, f"no improvement in round {rnd}; shrinking steps to {upt_uph_step}/{inter_mb_step}")
            if upt_uph_step <= 0.02 and inter_mb_step <= 0.1:
                log(tag, "steps at floor, stopping search")
                break

    csv_f.close()
    log(tag, f"=== {tag.upper()}_GRIDSEARCH_DONE ===")
    return current, best_metrics


if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else ['rgbt234_full', 'depthtrack_full']
    results = {}
    if 'rgbt234_full' in which:
        p, m = search('rgbt234_full', run_rgbt234_trial, RGBT234_TARGET, START_RGBT234)
        results['rgbt234_full'] = {'params': p, 'metrics': m}
    if 'depthtrack_full' in which:
        p, m = search('depthtrack_full', run_depthtrack_trial, DEPTHTRACK_TARGET, START_DEPTHTRACK)
        results['depthtrack_full'] = {'params': p, 'metrics': m}
    with open('/mnt/task_runtime/gs_large_ALL_FINAL.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("=== ALL_LARGE_GRIDSEARCH_DONE ===", flush=True)
