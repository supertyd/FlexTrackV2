#!/usr/bin/env bash
# Train FlexTrack-V2 from the backbone pre-train.
#
# Prerequisites:
#   1. Dataset paths configured in lib/train/admin/local.py
#      (copy lib/train/admin/local.py.example and edit).
#   2. Backbone pre-train weights at the path given by MODEL.ENCODER.PRETRAIN_TYPE
#      in experiments/flextrackv2/flextrackv2.yaml (default:
#      pretrained/FlexTrackV2_backbone_pretrain.pth.tar).
#
# Checkpoints are written to output/checkpoints/train/flextrackv2/<config>/.
set -e
NPROC=${NPROC:-8}
CONFIG=${1:-flextrackv2}

python tracking/train.py \
    --script flextrackv2 \
    --config "${CONFIG}" \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node "${NPROC}"
