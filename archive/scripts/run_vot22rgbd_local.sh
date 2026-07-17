#!/bin/bash
# VOT22RGBD evaluation for one config, LOCAL only (its 20GB data + workspace
# live only on this node). VOT-RGBD2022 uses the MultiStart protocol (stack
# vot2022/rgbd): the metric is EAO + Accuracy + Robustness, produced by the
# VOT toolkit driving the tracker LIVE with restart anchors (not precomputed
# unsupervised boxes like DepthTrack). The live trax tracker
# lib/test/vot/flextrackv2.py selects its config/checkpoint from FlexTrackV2_YAML /
# FlexTrackV2_EPOCH, which we pass per-config via the trackers.ini entry.
#
# Requires the config's checkpoint at
#   checkpoints/train/flextrackv2/<config>/FlexTrackV2_ep0040.pth.tar
# (pull from the B200 node first).
#
# Usage: [CUDA_VISIBLE_DEVICES=N] ./run_vot22rgbd_local.sh <config_name>
set -u
cd /mnt/task_runtime
CFG="${1:?Usage: $0 <config_name>}"
LABEL="${CFG}_vot22"
WS=/mnt/task_runtime/VOT22RGBD_workspace

# vot binary + tracker run on the default (py3.8) venv locally.
export http_proxy="${http_proxy:-http://proxy.config.pcp.local:3128}"
export https_proxy="${https_proxy:-http://proxy.config.pcp.local:3128}"

mkdir -p ablation_logs "ablation_results/${CFG}"

# Register a per-config trax tracker that loads THIS config's checkpoint.
python3 - "$CFG" "$LABEL" << 'PYEOF'
import sys
cfg, label = sys.argv[1], sys.argv[2]
ini = "/mnt/task_runtime/VOT22RGBD_workspace/trackers.ini"
text = open(ini).read() if __import__("os").path.exists(ini) else ""
if f"[{label}]" not in text:
    with open(ini, "a") as f:
        f.write(f"\n[{label}]\nlabel = {label}\nprotocol = traxpython\n"
                f"command = flextrackv2\npaths = /mnt/task_runtime/lib/test/vot\n"
                f"env_FlexTrackV2_YAML = {cfg}\nenv_FlexTrackV2_EPOCH = 40\n")
        print(f"registered tracker [{label}]")
PYEOF

echo "########## VOT22RGBD (multistart/EAO) eval: $CFG at $(date) ##########"
# Live evaluation with restart anchors; skips sequences already done, so safe to re-run.
vot evaluate --workspace "$WS" "$LABEL" > "ablation_logs/${CFG}_eval_vot22rgbd.log" 2>&1
echo "vot evaluate exit=$?"

# Analysis (EAO/Acc/Rob) + merge into the config's metrics.json (other fields untouched).
python3 compute_ablation_metrics.py --config "$CFG" --only VOT22RGBD >> "ablation_logs/${CFG}_metrics.log" 2>&1
echo "VOT22RGBD_DONE for $CFG"
grep -A4 '"VOT22RGBD"' "ablation_results/${CFG}/metrics.json" 2>/dev/null
