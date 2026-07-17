#!/bin/bash
# Standalone re-run of just the evaluation stage for an ablation config whose
# training already finished (and whose first evaluation attempt failed due to
# missing RGBT_workspace scripts / Depthtrack_workspace config / vot-toolkit).
# Mirrors run_ablation_worker.sh's eval section exactly.
#
# Usage: MCI_PYTHON=/path/to/python3 ./rerun_ablation_eval.sh <config_name>

set -u
cd /mnt/task_runtime

CFG="${1:?Usage: $0 <config_name>}"

PYTHON_BIN="${MCI_PYTHON:-python3}"
case "$PYTHON_BIN" in
  /*) export PATH="$(dirname "$PYTHON_BIN"):$PATH" ;;
esac

# vot-toolkit's Workspace.download_dataset() only skips its network fetch
# if <sequences_dir>/list.txt already exists; on ablation nodes the real
# DepthTrack sequences live under the HF download path, not under
# Depthtrack_workspace/ itself, so without this symlink it decides the local
# dataset is incomplete and tries (and fails) to download vot's own
# votrgbd2021 dataset from data.votchallenge.net -- a host unreachable
# through this network's proxy regardless of proxy env vars.
mkdir -p Depthtrack_workspace
if [ ! -e Depthtrack_workspace/sequences ]; then
  ln -s /mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/Depthtrack_workspace/sequences \
    Depthtrack_workspace/sequences
fi

export http_proxy="${http_proxy:-http://proxy.config.pcp.local:3128}"
export https_proxy="${https_proxy:-http://proxy.config.pcp.local:3128}"

# Use every GPU on the box instead of just GPU 0 -- run_sequence() round-robins
# worker threads across num_gpus already, it was just never told about more
# than 1. Keep the same 4-threads-per-GPU ratio the original single-GPU
# invocation used.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NUM_GPUS:-0}" -gt 0 ] || NUM_GPUS=1
THREADS=$((NUM_GPUS * 4))

# Without this, each of the $THREADS worker processes has PyTorch/numpy's
# BLAS/OpenMP layer default to using ALL cores on the box for its own CPU-side
# ops (image decode/resize) -- $THREADS processes x all-cores-each turns into
# massive oversubscription (observed load averages many multiples of nproc)
# that starves the GPUs of work instead of feeding them faster.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ablation_logs "ablation_results/${CFG}"

echo "########## [$(hostname)] STARTING EVAL (rerun): $CFG at $(date) -- ${NUM_GPUS} GPUs / ${THREADS} threads ##########"
for DS in RGBT234 RGBT234_miss LasHeR LasHeR_miss VisEvent VisEvent_miss; do
  "$PYTHON_BIN" RGBT_workspace/test_rgbt_mgpus.py \
    --script_name flextrackv2 --dataset_name "$DS" \
    --yaml_name "$CFG" --mode parallel --threads "$THREADS" --num_gpus "$NUM_GPUS" --epoch 40 \
    > "ablation_logs/${CFG}_eval_${DS}.log" 2>&1
done

"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack \
  --yaml_name "$CFG" --mode parallel --threads "$THREADS" --num_gpus "$NUM_GPUS" --epoch 40 \
  > "ablation_logs/${CFG}_eval_depthtrack.log" 2>&1
"$PYTHON_BIN" RGBT_workspace/test_depthtrack_mgpus.py \
  --script_name flextrackv2 --dataset_name depthtrack_miss \
  --yaml_name "$CFG" --mode parallel --threads "$THREADS" --num_gpus "$NUM_GPUS" --epoch 40 \
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

echo "########## [$(hostname)] FINISHED EVAL (rerun): $CFG at $(date) ##########"

"$PYTHON_BIN" compute_ablation_metrics.py --config "$CFG" >> "ablation_logs/${CFG}_metrics.log" 2>&1
cp "ablation_logs/${CFG}"_eval_*.log "ablation_results/${CFG}/" 2>/dev/null

echo "RERUN_EVAL_DONE for $CFG"
