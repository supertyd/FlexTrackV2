#!/bin/bash
cd /mnt/task_runtime
CFG="flextrackv2_b224_56_abl_moe_big"
for DS in RGBT234 RGBT234_miss VisEvent VisEvent_miss; do
  python RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --dataset_name "$DS" \
    --yaml_name "$CFG" --mode parallel --threads 4 --num_gpus 1 --epoch 40 \
    > "ablation_logs/${CFG}_eval_${DS}.log" 2>&1
done
python RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack \
  --yaml_name "$CFG" --mode parallel --threads 8 --num_gpus 2 --epoch 40 \
  > "ablation_logs/${CFG}_eval_depthtrack.log" 2>&1
python RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack_miss \
  --yaml_name "$CFG" --mode parallel --threads 8 --num_gpus 2 --epoch 40 \
  > "ablation_logs/${CFG}_eval_depthtrack_miss.log" 2>&1
python3 - "$CFG" << 'PYEOF'
import os, shutil, sys
cfg = sys.argv[1]
pairs = [("depthtrack", f"{cfg}_full"), ("depthtrack_miss", f"{cfg}_miss")]
for dset, label in pairs:
    src = f"/mnt/task_runtime/workspace/results/{dset}/{cfg}"
    dst = f"/mnt/task_runtime/Depthtrack_workspace/results/{label}/rgbd-unsupervised"
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(src):
        continue
    for f in os.listdir(src):
        if not f.endswith(".txt"):
            continue
        seq = f.replace(".txt", "")
        os.makedirs(os.path.join(dst, seq), exist_ok=True)
        shutil.copy(os.path.join(src, f), os.path.join(dst, seq, f"{seq}_001.txt"))
        with open(os.path.join(dst, seq, f"{seq}_001_time.value"), "w") as tf:
            tf.write("0.03\n" * 5000)
    ini = "/mnt/task_runtime/Depthtrack_workspace/trackers.ini"
    entry = f"[{label}]"
    text = open(ini).read() if os.path.exists(ini) else ""
    if entry not in text:
        with open(ini, "a") as f:
            f.write(f"\n[{label}]\nlabel = {label}\nprotocol = traxpython\ncommand = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot\n")
PYEOF
vot evaluate --workspace Depthtrack_workspace "${CFG}_full" "${CFG}_miss" >> "ablation_logs/${CFG}_eval_depthtrack.log" 2>&1
python3 compute_ablation_metrics.py --config "$CFG" >> "ablation_logs/${CFG}_metrics.log" 2>&1
cp ablation_logs/${CFG}_eval_*.log ablation_results/${CFG}/ 2>/dev/null
echo "MOE_BIG_EVAL_DONE"
