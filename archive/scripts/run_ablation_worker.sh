#!/bin/bash
# Runs a subset of the V56 TPAMI ablation configs on THIS machine: for each
# config, train -> evaluate on the 6 documented datasets -> write a small
# per-config results file -> commit+push just that file to the `ablations`
# branch. See ABLATIONS.md for the full ablation plan and dataset protocol.
#
# Usage: ./run_ablation_worker.sh <config_name> [config_name ...]
# Example: ./run_ablation_worker.sh flextrackv2_b224_56_abl_moe_hybrid flextrackv2_b224_56_abl_cma_fixed020

set -u
cd /mnt/task_runtime

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <config_name> [config_name ...]"
  exit 1
fi

# On p6-b200 (Blackwell/sm_100) nodes the default venv's PyTorch 2.1.1+cu118
# cannot run on the GPU at all (no sm_100 kernel support, and no py3.8-compatible
# PyTorch build ever added Blackwell support). Those nodes must use the separate
# mci310 conda env (python 3.10 + PyTorch 2.7.1+cu126) set up per
# ABLATIONS.md's "p6-b200 / Blackwell environment" notes. Set MCI_PYTHON/
# MCI_TORCHRUN to override; defaults keep working unchanged on non-B200 nodes.
PYTHON_BIN="${MCI_PYTHON:-python3}"
TORCHRUN_BIN="${MCI_TORCHRUN:-torchrun}"

# `vot evaluate` (and the traxpython subprocess it spawns for each tracker)
# resolve via PATH, not via $PYTHON_BIN directly. On nodes using a non-default
# env (MCI_PYTHON set to an absolute path, e.g. B200 nodes' mci310 conda env),
# put that env's bin/ first on PATH so both the `vot` binary and the python
# it launches trackers with come from the same env. No-op on default nodes.
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

# Without this, each parallel eval worker process has PyTorch/numpy's
# BLAS/OpenMP layer default to using ALL cores for its own CPU-side ops --
# multiplied across many worker processes this causes massive oversubscription
# (load averages many multiples of nproc) that starves the GPUs of work.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ablation_logs ablation_results

push_result() {
  local cfg="$1"
  git add "ablation_results/${cfg}/"
  git commit -q -m "Ablation results: ${cfg}" || return 0
  for attempt in 1 2 3; do
    git pull --rebase origin ablations >> "ablation_logs/${cfg}_push.log" 2>&1
    if git push origin ablations >> "ablation_logs/${cfg}_push.log" 2>&1; then
      echo "pushed results for ${cfg}"
      return 0
    fi
    echo "push attempt ${attempt} failed for ${cfg}, retrying..." >> "ablation_logs/${cfg}_push.log"
    sleep 5
  done
  echo "WARNING: failed to push results for ${cfg} after 3 attempts — check ablation_logs/${cfg}_push.log"
}

# Training seed: override per-run with MCI_SEED (used for variance/seed-repeat
# experiments); defaults to run_training.py's own default (42).
SEED="${MCI_SEED:-42}"

for CFG in "$@"; do
  echo "########## [$(hostname)] STARTING TRAIN: $CFG (seed=$SEED) at $(date) ##########"
  "$TORCHRUN_BIN" --nproc_per_node 8 lib/train/run_training.py \
    --script flextrackv2 --config "$CFG" --save_dir . --seed "$SEED" \
    > "ablation_logs/${CFG}_train.log" 2>&1
  TRAIN_STATUS=$?
  echo "########## [$(hostname)] FINISHED TRAIN: $CFG at $(date), exit=${TRAIN_STATUS} ##########"

  if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "TRAIN FAILED for $CFG — skipping evaluation, moving to next config"
    mkdir -p "ablation_results/${CFG}"
    printf '{"config": "%s", "status": "train_failed", "host": "%s"}\n' "$CFG" "$(hostname)" \
      > "ablation_results/${CFG}/metrics.json"
    push_result "$CFG"
    continue
  fi

  mkdir -p "ablation_results/${CFG}"

  # Delegate the whole evaluation phase to the unified eval script, which has
  # the battle-tested multi-GPU / thread-cap / DepthTrack-symlink / proxy /
  # per-sequence-isolation logic and now covers RGBT234, LasHeR, VisEvent
  # (each + _miss) and DepthTrack (+ _miss). VOT22RGBD is handled separately
  # on the local node (its 20GB dataset is local-only).
  MCI_PYTHON="$PYTHON_BIN" bash rerun_ablation_eval.sh "$CFG"

  push_result "$CFG"
done

echo "########## [$(hostname)] ALL ASSIGNED CONFIGS COMPLETE at $(date) ##########"
