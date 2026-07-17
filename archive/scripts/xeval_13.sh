#!/bin/bash
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH

echo "=== Running VisEvent Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== Running LasHeR Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== Running VisEvent Miss Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== Running LasHeR Miss Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== Running RGBT234 Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== Running RGBT234 Miss Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_13 --threads 8 --epoch 40

echo "=== All Evaluations Completed! ==="
