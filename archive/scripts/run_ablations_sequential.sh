#!/bin/bash
cd /mnt/task_runtime
CONFIGS=(
  flextrackv2_b224_56_abl_moe_big
  flextrackv2_b224_56_abl_moe_small
  flextrackv2_b224_56_abl_moe_middle
  flextrackv2_b224_56_abl_moe_hybrid
  flextrackv2_b224_56_abl_cma_fixed020
  flextrackv2_b224_56_abl_no_distill
  flextrackv2_b224_56_abl_pmax_015
  flextrackv2_b224_56_abl_pmax_025
  flextrackv2_b224_56_abl_pmax_050
)
for CFG in "${CONFIGS[@]}"; do
  echo "########## STARTING TRAIN: $CFG at $(date) ##########"
  torchrun --nproc_per_node 8 lib/train/run_training.py \
    --script flextrackv2 --config "$CFG" --save_dir . \
    > "ablation_logs/${CFG}_train.log" 2>&1
  echo "########## FINISHED TRAIN: $CFG at $(date), exit=$? ##########"
done
echo "########## ALL 9 ABLATION TRAININGS COMPLETE at $(date) ##########"
