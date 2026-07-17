#!/bin/bash
cd /mnt/task_runtime
CFGS="flextrackv2_b224_56_abl_moe_small flextrackv2_b224_56_abl_moe_middle flextrackv2_b224_56_abl_moe_hybrid flextrackv2_b224_56_abl_cma_fixed020 flextrackv2_b224_56_abl_no_distill flextrackv2_b224_56_abl_pmax_015 flextrackv2_b224_56_abl_pmax_025 flextrackv2_b224_56_abl_no_hallucinate_no_ortho flextrackv2_b224_56_abl_no_recon_loss_no_ortho flextrackv2_b224_56_abl_uni_a2r_no_ortho flextrackv2_b224_56_abl_pmax_000 flextrackv2_b224_56_abl_pmax_010 flextrackv2_b224_56_abl_pmax_065 flextrackv2_b224_56_abl_pmax_080 flextrackv2_l224_56_gstuned"
while true; do
  ALL_OK=1
  for CFG in $CFGS; do
    V=$(ls workspace/results/VisEvent/$CFG 2>/dev/null | wc -l)
    if [ "$V" -lt 320 ]; then ALL_OK=0; fi
  done
  echo "check: all_ok=$ALL_OK"
  if [ "$ALL_OK" -eq 1 ]; then
    echo "ALL_PULLS_DONE"
    break
  fi
  sleep 60
done
