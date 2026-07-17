import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)
DMM = os.path.abspath(os.path.join(prj, "data_missing_modality"))  # missing-modality annotation root (repo-relative)

from lib.test.tracker.flextrackv2 import FlexTrackV2Tracker
import lib.test.parameter.flextrackv2 as rgbt_prompt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
import json
import re

_RATIO_SUFFIX_RE = re.compile(r"_missR(\d{3})$")
_RGBDROP_RE = re.compile(r"_missRGB(\d{3})$")  # inverse sweep: RGB dropped, aux kept


def resolve_miss_json_path(dataset_name, official_path):
    """Ratio-sweep synthetic annotations ('<base>_missR###') resolve to
    generate_missing_ratio_json.py output; anything else uses the fixed
    official DepthTrack-miss annotation."""
    mg = _RGBDROP_RE.search(dataset_name)
    if mg:
        base = dataset_name[:mg.start()]
        return (DMM + "/synthetic_ratio_rgbdrop/"
                "{}_rgbdropR{}.json".format(base.lower(), mg.group(1)), True)
    m = _RATIO_SUFFIX_RE.search(dataset_name)
    if m:
        base = dataset_name[:m.start()]
        ratio_str = m.group(1)
        return (
            DMM + "/synthetic_ratio/"
            "{}_missR{}.json".format(base.lower(), ratio_str),
            True,
        )
    return official_path, False


def genConfig(seq_path, set_type):
    RGB_img_list = sorted([seq_path + '/color/' + p for p in os.listdir(seq_path + '/color') if p.endswith('.jpg')])
    T_img_list = sorted([seq_path + '/depth/' + p for p in os.listdir(seq_path + '/depth') if p.endswith('.png')])
    RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
    T_gt = np.copy(RGB_gt)
    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='flextrackv2'):
    seq_txt = seq_name
    save_name = '{}'.format(yaml_name)
    save_path = f'./workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if "miss" in dataset_name:
        import collections
        json_path, _ = resolve_miss_json_path(
            dataset_name,
            DMM + "/Missing_data_annotation/depthtrack-miss/missing_results_depthtrack.json",
        )
        if os.path.exists(json_path):
            with open(json_path, 'r') as load_f:
                miss_index = json.load(load_f)
        else:
            # HARD FAIL rather than silently faking a full-modality run — see
            # the same guard in test_rgbt_mgpus.py.
            raise FileNotFoundError(
                "Missing-modality annotation not found for {}: {}. "
                "Refusing to run a _miss eval without it (would fake full-modality).".format(dataset_name, json_path))

    try:
        params = rgbt_prompt_params.parameters(yaml_name, epoch)
        dtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbcolormap')

        if script_name == 'flextrackv2':
            mmtrack = FlexTrackV2Tracker(params, dataset_name)

            class ViPT_Depth(object):
                def __init__(self, tracker):
                    self.tracker = tracker
                def initialize(self, image, region):
                    self.tracker.initialize(image, {'init_bbox': region})
                def track(self, image, missing=[1, 1]):
                    outputs = self.tracker.track(image, missing)
                    return outputs['target_bbox'], outputs['best_score']

            tracker = ViPT_Depth(mmtrack)

        seq_path = seq_home + '/' + seq_name
        print('——————————Process sequence: '+seq_name +'——————————————')
        RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
        if len(RGB_img_list) == len(RGB_gt):
            result = np.zeros_like(RGB_gt)
        else:
            result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
        result[0] = np.copy(RGB_gt[0])
        last_region = result[0]

        if "miss" not in dataset_name:
            for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
                if frame_idx == 0:
                    image = get_x_frame(rgb_path, T_path, dtype=dtype, depth_clip=True)
                    tracker.initialize(image, RGB_gt[0].tolist())
                elif frame_idx > 0:
                    image = get_x_frame(rgb_path, T_path, dtype=dtype, depth_clip=True)
                    region, confidence = tracker.track(image, [1.0, 1.0])
                    last_region = region
                    result[frame_idx] = np.array(region)
        else:
            for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
                if frame_idx == 0:
                    image = get_x_frame(rgb_path, T_path, dtype=dtype, depth_clip=True)
                    tracker.initialize(image, RGB_gt[0].tolist())
                elif frame_idx > 0:
                    miss_state = ([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0]
                    image = get_x_frame(rgb_path, T_path, dtype=dtype, depth_clip=True)
                    if miss_state == [1.0, 1.0]:
                        pass
                    elif miss_state == [1.0, 0.0]:
                        image[:, :, 3:] = 0
                    elif miss_state == [0.0, 1.0]:
                        image[:, :, :3] = 0
                    elif miss_state == [0.0, 0.0]:
                        # both modalities missing: freeze at the last tracked box and
                        # skip the tracker entirely, instead of feeding an all-zero
                        # frame and taking the garbage prediction. Matches the
                        # RGBT/LasHeR/VisEvent miss protocol in test_rgbt_mgpus.py.
                        result[frame_idx] = last_region
                        continue

                    region, confidence = tracker.track(image, miss_state)
                    last_region = region
                    result[frame_idx] = np.array(region)

        np.savetxt(save_path, result, delimiter=',', fmt='%.2f')
        print(f"Finished {seq_name}")
    except Exception as e:
        # One bad sequence must not take down the whole multiprocessing.Pool
        # .starmap() batch -- see the identical fix in test_rgbt_mgpus.py.
        print('SEQUENCE FAILED: {} , error: {}'.format(seq_name, repr(e)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on DepthTrack dataset.')
    parser.add_argument('--script_name', type=str, default='flextrackv2', help='Name of tracking method.')
    parser.add_argument('--yaml_name', type=str, default='flextrackv2_b224_8', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='depthtrack', help='Name of dataset (depthtrack, depthtrack_miss).')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=2, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=40, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--seq_home', default='/mnt/task_runtime/Depthtrack_workspace/sequences', type=str,
                        help='sequence root; override for VOT22RGBD (same color/depth/groundtruth.txt layout)')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # Fail fast if a _miss eval has no annotation file (never fake full-modality).
    if "miss" in dataset_name:
        _jp, _ = resolve_miss_json_path(
            dataset_name,
            DMM + "/Missing_data_annotation/depthtrack-miss/missing_results_depthtrack.json",
        )
        if not os.path.exists(_jp):
            raise SystemExit("FATAL: missing-modality annotation absent for {}: {}. Aborting (won't fake full-modality).".format(dataset_name, _jp))
    seq_list = None

    seq_home = args.seq_home
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f)) and not f.startswith('.')]
    seq_list.sort()

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
