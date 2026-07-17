#!/bin/bash
# Trains + tests the isolated V56-large config (flextrackv2_l224_56), using the
# exact same 6-dataset V56 protocol as ABLATIONS.md / run_ablation_worker.sh.
#
# Unlike run_ablation_worker.sh, this does NOT git push results (no GitHub
# write credentials assumed) — results stay on local disk under
# ablation_results/flextrackv2_l224_56/ for retrieval via `bolt task scp`.
#
# On p6-b200 (Blackwell/sm_100) nodes the default venv's PyTorch has no
# sm_100 kernels; use the mci310 conda env instead (see ABLATIONS.md /
# run_ablation_worker.sh). Override MCI_PYTHON/MCI_TORCHRUN/NPROC as needed;
# defaults assume a plain 8-GPU non-Blackwell node.
#
# Usage: ./run_large_l224_worker.sh

set -u
CFG=flextrackv2_l224_56
cd /mnt/task_runtime

PYTHON_BIN="${MCI_PYTHON:-python3}"
TORCHRUN_BIN="${MCI_TORCHRUN:-torchrun}"
NPROC="${NPROC:-8}"

mkdir -p ablation_logs ablation_results

echo "########## [$(hostname)] STARTING TRAIN: $CFG at $(date) ##########"
"$TORCHRUN_BIN" --nproc_per_node "$NPROC" lib/train/run_training.py \
  --script flextrackv2 --config "$CFG" --save_dir . \
  > "ablation_logs/${CFG}_train.log" 2>&1
TRAIN_STATUS=$?
echo "########## [$(hostname)] FINISHED TRAIN: $CFG at $(date), exit=${TRAIN_STATUS} ##########"

mkdir -p "ablation_results/${CFG}"
if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "TRAIN FAILED for $CFG — skipping evaluation"
  printf '{"config": "%s", "status": "train_failed", "host": "%s"}\n' "$CFG" "$(hostname)" \
    > "ablation_results/${CFG}/metrics.json"
  exit 1
fi

echo "########## [$(hostname)] STARTING EVAL: $CFG at $(date) ##########"
for DS in RGBT234 RGBT234_miss LasHeR LasHeR_miss VisEvent VisEvent_miss; do
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --dataset_name "$DS" \
    --yaml_name "$CFG" --mode parallel --threads 32 --num_gpus 8 --epoch 40 \
    > "ablation_logs/${CFG}_eval_${DS}.log" 2>&1
done

"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack \
  --yaml_name "$CFG" --mode parallel --threads 32 --num_gpus 8 --epoch 40 \
  > "ablation_logs/${CFG}_eval_depthtrack.log" 2>&1
"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack_miss \
  --yaml_name "$CFG" --mode parallel --threads 32 --num_gpus 8 --epoch 40 \
  > "ablation_logs/${CFG}_eval_depthtrack_miss.log" 2>&1

"$PYTHON_BIN" - "$CFG" << 'PYEOF'
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

echo "########## [$(hostname)] FINISHED EVAL: $CFG at $(date) ##########"

"$PYTHON_BIN" compute_ablation_metrics.py --config "$CFG" >> "ablation_logs/${CFG}_metrics.log" 2>&1
cp "ablation_logs/${CFG}"_eval_*.log "ablation_results/${CFG}/" 2>/dev/null

echo "########## [$(hostname)] ALL DONE at $(date) — results in ablation_results/${CFG}/ ##########"
