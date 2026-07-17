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

# Ratio-sweep synthetic missing-modality annotations (see
# generate_missing_ratio_json.py). Dataset names of the form
# "<Base>_missR###" (e.g. RGBT234_missR025) resolve to a synthetic JSON
# instead of the single fixed official "_miss" pattern.
_RATIO_SUFFIX_RE = re.compile(r"_missR(\d{3})$")
_RGBDROP_RE = re.compile(r"_missRGB(\d{3})$")  # inverse sweep: RGB dropped, aux kept


def resolve_miss_json_path(dataset_name, official_paths):
    """Return (json_path, is_synthetic) for a dataset_name that contains
    'miss'. official_paths maps exact dataset_name -> official JSON path."""
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
    if dataset_name in official_paths:
        return official_paths[dataset_name], False
    # generic fallback for names like "<Base>_miss" not in the explicit map
    base = dataset_name[:-5] if dataset_name.endswith("_miss") else dataset_name
    return (
        DMM + "/Missing_data_annotation/"
        + base.lower() + "-miss/missing_results_" + base.lower() + ".json",
        False,
    )


def genConfig(seq_path, set_type):
    if set_type.startswith('RGBT234'):
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    elif set_type.startswith('LasHeR'):
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
    

    elif "VisEvent" in set_type:
        RGB_img_list = sorted([seq_path + "/vis_imgs/" + p for p in os.listdir(seq_path + "/vis_imgs") if p.endswith(".bmp") or p.endswith(".jpg")])
        T_img_list = sorted([seq_path + "/event_imgs/" + p for p in os.listdir(seq_path + "/event_imgs") if p.endswith(".bmp") or p.endswith(".jpg")])
        try:
            RGB_gt = np.loadtxt(seq_path + "/groundtruth.txt", delimiter=",")
        except:
            RGB_gt = np.loadtxt(seq_path + "/groundtruth.txt")
        T_gt = np.copy(RGB_gt)
    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt'):
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'./workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    save_folder = f'./workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)
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
        _official_paths = {
            "LasHeR_miss": DMM + "/Missing_data_annotation/LasHeR245-Miss/missing_results_lasher245.json",
            "RGBT234_miss": DMM + "/Missing_data_annotation/RGBT234-Miss/missing_results_rgbt234.json",
            "VisEvent_miss": DMM + "/Missing_data_annotation/visevent-miss/missing_results_visevent.json",
        }
        json_path, _ = resolve_miss_json_path(dataset_name, _official_paths)
        if os.path.exists(json_path):
            with open(json_path,'r') as load_f:
                miss_index = json.load(load_f)
        else:
            # HARD FAIL: a missing-modality eval with no annotation file would
            # silently degrade to a full-modality run (dummy all-[1,1] index),
            # producing bogus "_miss" numbers indistinguishable from full. That
            # exact footgun corrupted an earlier B200 run. Never fall back.
            raise FileNotFoundError(
                "Missing-modality annotation not found for {}: {}. "
                "Refusing to run a _miss eval without it (would fake full-modality).".format(dataset_name, json_path))




    try:
        if script_name == 'flextrackv2':
            params = rgbt_prompt_params.parameters(yaml_name, epoch)
            mmtrack = FlexTrackV2Tracker(params,dataset_name)  # "GTOT" # dataset_name
            tracker = ViPT_RGBT(tracker=mmtrack)

        seq_path = seq_home + '/' + seq_name
        print('——————————Process sequence: '+seq_name +'——————————————')
        RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
        if len(RGB_img_list) == len(RGB_gt):
            result = np.zeros_like(RGB_gt)
        else:
            result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)

        # A few VisEvent sequences have the target absent (gt box all-zero)
        # in frame 0 -- initializing there crashes with "Too small bounding
        # box" and the whole sequence used to be silently dropped. Init on
        # the first frame with a real (w>0,h>0) box instead; frames before
        # that stay zero, matching ground truth, and are excluded from
        # scoring anyway via absent_label.txt. No-op for every other
        # dataset, where frame 0 is always valid.
        init_idx = 0
        for i in range(len(RGB_gt)):
            if RGB_gt[i][2] > 0 and RGB_gt[i][3] > 0:
                init_idx = i
                break

        result[init_idx] = np.copy(RGB_gt[init_idx])
        last_region = result[init_idx]
        toc = 0
        if "miss" not in dataset_name:
            for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
                if frame_idx < init_idx:
                    continue
                tic = cv2.getTickCount()
                if frame_idx == init_idx:
                    # initialization  cuz missing modalities does not include first frame missing so keep the same with other trackers.
                    image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    tracker.initialize(image, RGB_gt[init_idx].tolist())  # xywh
                else:


                    image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    region, confidence = tracker.track(image,[1.0,1.0])  # xywh

                    last_region = region
                    result[frame_idx] = np.array(region)
                toc += cv2.getTickCount() - tic
        else:
            for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
                if frame_idx < init_idx:
                    continue
                tic = cv2.getTickCount()
                if frame_idx == init_idx:
                    # initialization  cuz missing modalities does not include first frame missing so keep the same with other trackers.
                    image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    tracker.initialize(image, RGB_gt[init_idx].tolist())  # xywh
                elif frame_idx > init_idx:
                    if (([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0])==[1.0,1.0]:
                        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                        region, confidence = tracker.track(image,(([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0]))  # xywh
                        last_region = result[frame_idx] = np.array(region)

                    elif (([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0])==[1.0,0.0]:
                        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                        image[:,:,3:] = 0
                        region, confidence = tracker.track(image,(([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0]))  # xywh
                        last_region = result[frame_idx] = np.array(region)
                    elif (([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0])==[0.0,1.0]:
                        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                        image[:,:,:3] = 0
                        region, confidence = tracker.track(image,(([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0]))  # xywh
                        last_region = result[frame_idx] = np.array(region)  # updatse last_region for double-missing case
                    elif (([1.0,1.0] if frame_idx >= len(miss_index[seq_name]["data"]) else miss_index[seq_name]["data"][frame_idx]) if frame_idx < len(miss_index[seq_name]["data"]) else [1.0, 1.0])==[0.0,0.0]:
                            result[frame_idx] = last_region
                toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        if not debug:
            np.savetxt(save_path, result)
        print('{} , fps:{}'.format(seq_name, frame_idx / toc))
    except Exception as e:
        # One bad sequence (corrupt annotation, non-sequence directory picked up
        # by a naive listdir, a too-small init box, ...) must not take down the
        # whole multiprocessing.Pool.starmap() batch -- that silently drops every
        # sequence still queued/in-flight in the other workers along with it.
        print('SEQUENCE FAILED: {} , error: {}'.format(seq_name, repr(e)))


class ViPT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB,missing):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB,missing)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='flextrackv2', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='flextrackv2_b224_8', help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='LasHeR_miss', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=80, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # Fail fast (before spawning the worker pool) if a _miss eval has no
    # annotation file, rather than letting every worker fail one-by-one or,
    # worse, silently faking a full-modality run.
    if "miss" in dataset_name:
        _miss_paths = {
            "LasHeR_miss": DMM + "/Missing_data_annotation/LasHeR245-Miss/missing_results_lasher245.json",
            "RGBT234_miss": DMM + "/Missing_data_annotation/RGBT234-Miss/missing_results_rgbt234.json",
            "VisEvent_miss": DMM + "/Missing_data_annotation/visevent-miss/missing_results_visevent.json",
        }
        _jp, _is_synth = resolve_miss_json_path(dataset_name, _miss_paths)
        if not os.path.exists(_jp):
            raise SystemExit("FATAL: missing-modality annotation absent for {}: {}. Aborting (won't fake full-modality).".format(dataset_name, _jp))
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/home/lz/Videos/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f)) and not f.startswith('.')]
        seq_list.sort()
    elif dataset_name.startswith('RGBT234'):
        seq_home = '/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234'
        # 'attr_txt' is a per-attribute annotation directory shipped alongside
        # the real sequence directories in this dataset layout, not a sequence
        # itself -- it has no visible/infrared subfolders and previously crashed
        # run_sequence() for the whole pool if it fell in the same chunk.
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f)) and not f.startswith('.') and f != 'attr_txt']
        seq_list.sort()
    elif dataset_name.startswith('LasHeR'):
        seq_home = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f)) and not f.startswith('.')]
        seq_list.sort()
    elif dataset_name.startswith("VisEvent"):
        seq_home = "/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test"
        with open(join(seq_home, "testlist.txt"), "r") as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

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
