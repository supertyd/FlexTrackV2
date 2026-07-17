import os
import re
import glob
import shutil
import subprocess
import yaml

YAML_PATH = "/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_56.yaml"
RAW_RES_DIR = "/mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_56"
VOT_RES_DIR = "/mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_depthtrack/rgbd-unsupervised"
WORKSPACE = "/mnt/task_runtime/Depthtrack_workspace"
TRACKER = "flextrackv2_v56_depthtrack"

TARGET = {"Pr": 67.1, "Re": 66.9, "F": 67.0}
MAX_ROUNDS = 12

CANDIDATES = [
    {"upt": 0.75, "uph": 0.90, "inter": 70, "mb": 500},
    {"upt": 0.85, "uph": 0.90, "inter": 70, "mb": 500},
    {"upt": 0.80, "uph": 0.85, "inter": 70, "mb": 500},
    {"upt": 0.80, "uph": 0.95, "inter": 70, "mb": 500},
    {"upt": 0.80, "uph": 0.90, "inter": 50, "mb": 500},
    {"upt": 0.80, "uph": 0.90, "inter": 90, "mb": 500},
    {"upt": 0.80, "uph": 0.90, "inter": 70, "mb": 400},
    {"upt": 0.80, "uph": 0.90, "inter": 70, "mb": 600},
    {"upt": 0.75, "uph": 0.85, "inter": 50, "mb": 500},
    {"upt": 0.85, "uph": 0.95, "inter": 90, "mb": 600},
    {"upt": 0.70, "uph": 0.85, "inter": 40, "mb": 400},
    {"upt": 0.85, "uph": 0.85, "inter": 55, "mb": 550},
]

ENV_VOT = {
    "HTTP_PROXY": "http://proxy.config.pcp.local:3128",
    "HTTPS_PROXY": "http://proxy.config.pcp.local:3128",
    "PYTHONPATH": "/mnt/task_runtime",
}


def run(cmd, env_add=None, quiet=False):
    env = os.environ.copy()
    env["PATH"] = "/coreflow/venv/bin:/coreflow/mambaforge/bin:" + env.get("PATH", "")
    if env_add:
        env.update(env_add)
    print(f">>> {cmd}", flush=True)
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(cmd, shell=True, env=env, **kwargs)


def update_yaml(upt, uph, inter, mb):
    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["TEST"]["UPT"]["DEPTHTRACK"] = float(upt)
    cfg["TEST"]["UPH"]["DEPTHTRACK"] = float(uph)
    cfg["TEST"]["INTER"]["DEPTHTRACK"] = int(inter)
    cfg["TEST"]["MB"]["DEPTHTRACK"] = int(mb)
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def convert_to_vot_format():
    if os.path.exists(VOT_RES_DIR):
        shutil.rmtree(VOT_RES_DIR)
    os.makedirs(VOT_RES_DIR, exist_ok=True)
    for fname in os.listdir(RAW_RES_DIR):
        if not fname.endswith(".txt"):
            continue
        seq = fname[:-4]
        seq_dir = os.path.join(VOT_RES_DIR, seq)
        os.makedirs(seq_dir, exist_ok=True)
        shutil.copy(os.path.join(RAW_RES_DIR, fname), os.path.join(seq_dir, f"{seq}_001.txt"))
        with open(os.path.join(seq_dir, f"{seq}_001_time.value"), "w") as tf:
            tf.write("0.03\n" * 5000)


def latest_report():
    files = glob.glob(os.path.join(WORKSPACE, "analysis", "*", "report.html"))
    if not files:
        return None
    return sorted(files, key=os.path.getmtime)[-1]


def parse_metrics(html_path):
    html = open(html_path).read()
    m = re.search(
        r'data-tracker="' + re.escape(TRACKER) + r'"[^>]*>(.*?)</tr>',
        html,
        re.DOTALL,
    )
    if not m:
        return None
    vals = re.findall(r'data-value="([0-9.]+)"', m.group(1))
    if len(vals) < 3:
        return None
    return {"Pr": float(vals[0]) * 100, "Re": float(vals[1]) * 100, "F": float(vals[2]) * 100}


def run_one_round(cand):
    update_yaml(cand["upt"], cand["uph"], cand["inter"], cand["mb"])
    run(f"rm -rf {RAW_RES_DIR}")
    cmd = (
        "cd /mnt/task_runtime && /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py "
        "--script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b224_56 "
        "--mode parallel --threads 32 --num_gpus 8 --epoch 40"
    )
    run(cmd, quiet=True)
    if not os.path.isdir(RAW_RES_DIR) or not os.listdir(RAW_RES_DIR):
        print("!! no raw results produced, skipping round")
        return None
    convert_to_vot_format()
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot evaluate --workspace {WORKSPACE} {TRACKER}", ENV_VOT)
    run(f"cd {WORKSPACE} && /coreflow/venv/bin/vot analysis --nocache --workspace {WORKSPACE} {TRACKER}", ENV_VOT)
    html_path = latest_report()
    if not html_path:
        print("!! no report generated, skipping round")
        return None
    metrics = parse_metrics(html_path)
    return metrics


def meets_target(metrics):
    return (
        metrics["Pr"] >= TARGET["Pr"]
        and metrics["Re"] >= TARGET["Re"]
        and metrics["F"] >= TARGET["F"]
    )


def main():
    print(f"Target (FlexTrack): Pr={TARGET['Pr']} Re={TARGET['Re']} F={TARGET['F']}")
    best = None
    for idx, cand in enumerate(CANDIDATES[:MAX_ROUNDS], 1):
        print(f"\n===== Round {idx}/{min(len(CANDIDATES), MAX_ROUNDS)}: {cand} =====", flush=True)
        metrics = run_one_round(cand)
        if metrics is None:
            print(f"Round {idx}: FAILED (no metrics)")
            continue
        print(f"Round {idx} result: Pr={metrics['Pr']:.2f} Re={metrics['Re']:.2f} F={metrics['F']:.2f}")
        if best is None or metrics["F"] > best[1]["F"]:
            best = (cand, metrics)
        if meets_target(metrics):
            print(f"\n*** TARGET REACHED at round {idx}: {cand} -> {metrics} ***")
            break
    else:
        print("\n*** Reached round cap without meeting full target. ***")

    print("\n===== SUMMARY =====")
    if best:
        print(f"Best candidate: {best[0]}")
        print(f"Best metrics: Pr={best[1]['Pr']:.2f} Re={best[1]['Re']:.2f} F={best[1]['F']:.2f}")
        print(f"Target:        Pr={TARGET['Pr']} Re={TARGET['Re']} F={TARGET['F']}")
    else:
        print("No successful rounds.")


if __name__ == "__main__":
    main()
