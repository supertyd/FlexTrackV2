#!/bin/bash
cd /mnt/task_runtime
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[gpumax] $(date +%H:%M:%S) RGBT234_miss 续跑(已有59条,skip已完成)"
/coreflow/venv/bin/python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 48 --num_gpus 8 --epoch 40 > /mnt/task_runtime/gpumax_rgbt234.log 2>&1
echo "[gpumax] $(date +%H:%M:%S) RGBT234_miss done: $(ls workspace/results/RGBT234_miss/flextrackv2_b224_56/|grep -c '\.txt$')/234"

echo "[gpumax] $(date +%H:%M:%S) VisEvent_miss 重跑(修复后)"
/coreflow/venv/bin/python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 48 --num_gpus 8 --epoch 40 > /mnt/task_runtime/gpumax_visevent.log 2>&1
echo "[gpumax] $(date +%H:%M:%S) VisEvent_miss done: $(ls workspace/results/VisEvent_miss/flextrackv2_b224_56/|grep -c '\.txt$')/320"

echo "[gpumax] $(date +%H:%M:%S) === 三个miss最终评测 ==="
/coreflow/venv/bin/python eval_miss_only.py 2>&1 | grep -E "FlexTrackV2="
echo "[gpumax] GPUMAX_ALL_DONE"
