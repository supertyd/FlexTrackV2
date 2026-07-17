# torchrun --nproc_per_node 8 lib/train/run_training.py --script flextrackv2 --config flextrackv2_b224 --save_dir .

python -m torch.distributed.launch --nproc_per_node 8 lib/train/run_training.py --script flextrackv2 --config flextrackv2_b224 --save_dir .
python tracking/train.py --script flextrackv2 --config flextrackv2_b224 --save_dir ./output --mode multiple --nproc_per_node 8


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224 --threads 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224 --threads 8


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224 --threads 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224 --threads 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224 --threads 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR --yaml_name flextrackv2_b224 --threads 8


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent_miss --yaml_name flextrackv2_b224 --threads 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name LasHeR_miss --yaml_name flextrackv2_b224 --threads 8


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234 --yaml_name flextrackv2_b224 --threads 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --dataset_name RGBT234_miss --yaml_name flextrackv2_b224 --threads 8