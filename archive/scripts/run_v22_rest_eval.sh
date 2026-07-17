#!/bin/bash
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Running DepthTrack Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0 python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b224_22 --mode sequential --epoch 40

echo "=== Running DepthTrack Miss Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0 python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack_miss --yaml_name flextrackv2_b224_22 --mode sequential --epoch 40

echo "=== Running VisEvent Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== Running VisEvent Miss Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== Running LasHeR Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== Running LasHeR Miss Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== Running RGBT234 Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== Running RGBT234 Miss Evaluation for V22 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_22 --threads 8 --epoch 40

echo "=== All V22 Evaluations Completed! ==="
