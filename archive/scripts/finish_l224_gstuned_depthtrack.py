"""Convert flextrackv2_l224_56_gstuned's raw DepthTrack box predictions into VOT
format and register+run the VOT tracker labels, matching the pattern used by
run_l224_gstuned_parallel_eval.sh. Run this after the raw depthtrack/
depthtrack_miss predictions finish."""
import os, shutil, subprocess, json

CFG = "flextrackv2_l224_56_gstuned"
pairs = [("depthtrack", f"{CFG}_full"), ("depthtrack_miss", f"{CFG}_miss")]
WS = "/mnt/task_runtime/Depthtrack_workspace"

for dset, label in pairs:
    src = f"/mnt/task_runtime/workspace/results/{dset}/{CFG}"
    dst = f"{WS}/results/{label}/rgbd-unsupervised"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith(".txt"):
            continue
        seq = f.replace(".txt", "")
        os.makedirs(os.path.join(dst, seq), exist_ok=True)
        shutil.copy(os.path.join(src, f), os.path.join(dst, seq, f"{seq}_001.txt"))
        with open(os.path.join(dst, seq, f"{seq}_001_time.value"), "w") as tf:
            tf.write("0.03\n" * 5000)

# register tracker labels in trackers.ini if not already present
ini = f"{WS}/trackers.ini"
text = open(ini).read() if os.path.exists(ini) else ""
for _, label in pairs:
    if f"[{label}]" not in text:
        with open(ini, "a") as f:
            f.write(f"\n[{label}]\nlabel = {label}\nprotocol = traxpython\n"
                     f"command = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot\n")
        print(f"registered {label}")

for _, label in pairs:
    subprocess.run(["vot", "analysis", "--nocache", "--format", "json",
                     "--workspace", WS, label], timeout=1200)

import glob
for _, label in pairs:
    candidates = sorted(glob.glob(f"{WS}/analysis/*/results.json"), key=os.path.getmtime, reverse=True)
    found = None
    for path in candidates:
        try:
            data = json.load(open(path))
            if label in data.get("trackers", {}):
                exp = list(data["results"].values())[0]
                pr = exp["results"][0][0]
                if pr:
                    found = {"precision": round(pr[0]*100,2), "recall": round(pr[1]*100,2), "fscore": round(pr[2]*100,2)}
                    break
        except Exception:
            continue
    print(label, found)
