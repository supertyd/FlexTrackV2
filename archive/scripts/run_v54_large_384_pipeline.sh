#!/bin/bash
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Starting V54 Large 384 Training ==="
bash /mnt/task_runtime/xtrain_large_384.sh > /mnt/task_runtime/training_pipeline_v54_large.log 2>&1

echo "=== V54 Large 384 Training Finished! Creating Evaluation Script ==="
cat << "INNER_EOF" > /mnt/task_runtime/run_v54_large_rest_eval.sh
#!/bin/bash
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:\$PYTHONPATH
source /coreflow/venv/bin/activate

# Force evaluation to use Epoch 40 checkpoint for all datasets
export FlexTrackV2_EPOCH=40

# Optimize CPU threads for PyTorch multiprocessing to prevent CPU thrashing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Running DepthTrack Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0 python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b384_large_54 --mode parallel --threads 8 --num_gpus 4 --epoch 40

echo "=== Running DepthTrack Miss Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0 python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack_miss --yaml_name flextrackv2_b384_large_54 --mode parallel --threads 8 --num_gpus 4 --epoch 40

echo "=== Running VisEvent Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== Running VisEvent Miss Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== Running LasHeR Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== Running LasHeR Miss Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== Running RGBT234 Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== Running RGBT234 Miss Evaluation for V54 Large ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b384_large_54 --threads 24 --num_gpus 4 --epoch 40

echo "=== All V54 Large Evaluations Completed! ==="
INNER_EOF

chmod +x /mnt/task_runtime/run_v54_large_rest_eval.sh

echo "=== Launching V54 Large Rest Eval ==="
bash /mnt/task_runtime/run_v54_large_rest_eval.sh > /mnt/task_runtime/v54_large_evaluation_pipeline.log 2>&1

echo "=== All V54 Large Steps Completed Successfully ==="
