#!/bin/bash
# FlexTrackV2-B224 (v10) Distributed Training and Evaluation

echo "=== Starting Distributed Training on 8 GPUs for FlexTrackV2-B224 (v10) ==="
python tracking/train.py --script flextrackv2 --config flextrackv2_b224_10 --save_dir ./output --mode multiple --nproc_per_node 8

echo "=== Starting Multi-GPU Parallel Evaluation across 6 Benchmarks ==="
# 1. VisEvent
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40
# 2. LasHeR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40
# 3. VisEvent_miss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40
# 4. LasHeR_miss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40
# 5. RGBT234
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40
# 6. RGBT234_miss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_10 --threads 32 --epoch 40

echo "=== Running Metric Aggregation and Verification ==="
python /root/.gemini/antigravity-ide/brain/bafff6f9-9ebb-4cdb-995d-499717936d3a/scratch/eval_rgbt_10.py

echo "=== FlexTrackV2-B224 (v10) Execution Completed! ==="
