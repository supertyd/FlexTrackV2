#!/bin/bash
# One-time bootstrap for a FRESH Bolt node before running run_ablation_worker.sh.
# Run this after `bolt task submit ... --git https://github.com/supertyd/FlexTrack-V2.git@ablations`
# has booted the pod and you have a shell on it (bolt task ssh <id>).
#
# Requires HF_TOKEN to be set in the environment (pass it via the Bolt config's
# environment_variables, or `export HF_TOKEN=...` before running this script —
# never hardcode it in a file).
#
# Usage: HF_TOKEN=... ./provision_ablation_node.sh

set -e
cd /mnt/task_runtime

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Export it first (see this repo's Bolt task config)." >&2
  exit 1
fi

echo "########## Downloading LasHeR / VisEvent / DepthTrack / RGBT234 / Depthtrack_workspace ##########"
python download_datasets_hf.py

echo "########## Downloading missing-modality annotation data ##########"
python download_missing.py

echo "########## Regenerating lib/test/evaluation/local.py (generic paths) ##########"
# Must run BEFORE writing lib/train/admin/local.py below --
# create_default_local_file_ITP_train() (called by this script) unconditionally
# overwrites lib/train/admin/local.py with generic ./data/<name> placeholder
# paths, which would clobber the corrected dataset paths if run after.
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir . || \
  echo "NOTE: create_default_local_file.py failed or needs manual review — check lib/test/evaluation/local.py before evaluating"

echo "########## Writing lib/train/admin/local.py (dataset paths, corrected nesting) ##########"
mkdir -p lib/train/admin
cat > lib/train/admin/local.py << 'PYEOF'
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/task_runtime/workspace'
        self.tensorboard_dir = '/mnt/task_runtime/tensorboard'
        self.pretrained_networks = '/mnt/task_runtime/pretrained_networks'
        self.lasot_dir = '/mnt/my_netdisk/FlexTrackV2/data/lasot'
        self.vasttrack_dir = '/mnt/my_netdisk/FlexTrackV2/data/vasttrack'
        self.got10k_dir = '/mnt/my_netdisk/FlexTrackV2/data/got10k/train'
        self.lasot_lmdb_dir = '/mnt/my_netdisk/FlexTrackV2/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/my_netdisk/FlexTrackV2/data/got10k_lmdb'
        self.trackingnet_dir = '/mnt/my_netdisk/FlexTrackV2/data/trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/my_netdisk/FlexTrackV2/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/my_netdisk/FlexTrackV2/data/coco'
        self.coco_lmdb_dir = '/mnt/my_netdisk/FlexTrackV2/data/coco_lmdb'
        self.imagenet1k_dir = '/mnt/my_netdisk/FlexTrackV2/data/imagenet1k'
        self.imagenet22k_dir = '/mnt/my_netdisk/FlexTrackV2/data/imagenet22k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/my_netdisk/FlexTrackV2/data/vid'
        self.imagenet_lmdb_dir = '/mnt/my_netdisk/FlexTrackV2/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.depthtrack_dir = '/mnt/task_wrapper/user_output/artifacts/depthtrack/depthtrack/train/DepthTrackTraining'
        self.lasher_dir = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/trainingset'
        self.visevent_dir = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/train'
        self.refcoco_dir = '/mnt/my_netdisk/FlexTrackV2/data/refcoco'
        self.tnl2k_dir = '/mnt/my_netdisk/FlexTrackV2/data/tnl2k_train'
        self.otb99_dir = '/mnt/my_netdisk/FlexTrackV2/data/otb_lang'
PYEOF

echo "########## NOTE: pretrained checkpoint must be copied from the source node ##########"
echo "Run this FROM the source node (not here), then re-check this file exists:"
echo "  bolt task scp <SOURCE_TASK_ID>:/mnt/task_runtime/train/FlexTrackV2_ep0300.pth.tar <THIS_TASK_ID>:/mnt/task_runtime/train/FlexTrackV2_ep0300.pth.tar"
ls -la train/FlexTrackV2_ep0300.pth.tar 2>/dev/null || echo "  -> NOT YET PRESENT on this node."

echo "########## Provisioning complete. Verify the checkpoint above, then run: ##########"
echo "  ./run_ablation_worker.sh <config1> <config2> <config3>"
