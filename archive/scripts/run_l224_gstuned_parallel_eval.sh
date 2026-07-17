#!/bin/bash
# Parallel version of run_l224_gstuned_eval.sh: instead of evaluating the
# 8 V56-protocol datasets one at a time (which leaves 8 GPUs mostly idle
# during single-dataset tracking inference), launch all 8 concurrently in
# the background so the same 8 GPUs are shared by all of them at once.
#
# Usage: ./run_l224_gstuned_parallel_eval.sh
set -u
cd /mnt/task_runtime
YAML="flextrackv2_l224_56_gstuned"
NUM_GPUS=8
THREADS=8
EPOCH=40
PYTHON_BIN="${MCI_PYTHON:-python3}"

mkdir -p ablation_logs "ablation_results/${YAML}"

echo "########## [$(hostname)] STARTING PARALLEL GSTUNED EVAL at $(date) ##########"

for DS in RGBT234 RGBT234_miss LasHeR LasHeR_miss VisEvent VisEvent_miss; do
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --yaml_name "$YAML" --dataset_name "$DS" \
    --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
    > "ablation_logs/${YAML}_eval_${DS}.log" 2>&1 &
done

"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --yaml_name "$YAML" --dataset_name depthtrack \
  --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
  --seq_home /mnt/task_runtime/Depthtrack_workspace/sequences \
  > "ablation_logs/${YAML}_eval_depthtrack.log" 2>&1 &
"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --yaml_name "$YAML" --dataset_name depthtrack_miss \
  --threads $THREADS --num_gpus $NUM_GPUS --epoch $EPOCH --mode parallel \
  --seq_home /mnt/task_runtime/Depthtrack_workspace/sequences \
  > "ablation_logs/${YAML}_eval_depthtrack_miss.log" 2>&1 &

echo "########## [$(hostname)] Launched 8 parallel eval jobs, waiting at $(date) ##########"
wait
echo "########## [$(hostname)] All parallel eval jobs finished at $(date) ##########"

"$PYTHON_BIN" - "$YAML" << 'PYEOF'
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

export PATH="/coreflow/mambaforge/envs/mci310/bin:$PATH"
vot evaluate --workspace Depthtrack_workspace "${YAML}_full" "${YAML}_miss" >> "ablation_logs/${YAML}_eval_depthtrack.log" 2>&1

echo "########## [$(hostname)] FINISHED GSTUNED EVAL at $(date) ##########"

"$PYTHON_BIN" compute_ablation_metrics.py --config "$YAML" >> "ablation_logs/${YAML}_metrics.log" 2>&1
cp "ablation_logs/${YAML}"_eval_*.log "ablation_results/${YAML}/" 2>/dev/null

echo "########## [$(hostname)] ALL DONE at $(date) — results in ablation_results/${YAML}/ ##########"
