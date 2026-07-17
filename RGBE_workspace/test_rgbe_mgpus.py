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
import lib.test.parameter.flextrackv2 as rgbe_prompt_params
from lib.train.dataset.depth_utils import get_x_frame
import multiprocessing
import torch
import time
import json


def genConfig(seq_path, set_type):
    if 'VisEvent' in set_type:
        RGB_img_list = sorted([seq_path + '/vis_imgs/' + p for p in os.listdir(seq_path + '/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        E_img_list = sorted([seq_path + '/event_imgs/' + p for p in os.listdir(seq_path + '/event_imgs') if os.path.splitext(p)[1] == '.bmp'])

        RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    return RGB_img_list, E_img_list, RGB_gt, absent_label


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60, debug=0, script_name='prompt'):
    # Bug #12 fix (previously applied to test_rgbt/test_depthtrack but NOT here):
    # never let one bad sequence's exception propagate out of a pool.starmap
    # worker, which silently aborts every still-queued sequence in the batch.
    # Isolate each sequence so a single failure drops only that sequence.
    try:
        return _run_sequence_impl(seq_name, seq_home, dataset_name, yaml_name, num_gpu, epoch, debug, script_name)
    except Exception as e:
        import traceback
        print(f'[SEQ-FAILED] {seq_name}: {e}')
        traceback.print_exc()
        return None


def _run_sequence_impl(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60, debug=0, script_name='prompt'):
    import os
    import collections
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    seq_txt = seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'./workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    if "miss" in dataset_name:
        json_path = DMM + "/Missing_data_annotation/visevent-miss/missing_results_visevent.json"
        if os.path.exists(json_path):
            with open(json_path,'r') as load_f:
                miss_index = json.load(load_f)
        else:
            print("Warning: {} not found. Using dummy missing index.".format(json_path))
            miss_index = collections.defaultdict(lambda: {"data": [[1.0, 1.0]] * 10000})    

    if script_name == 'flextrackv2':
        params = rgbe_prompt_params.parameters(yaml_name,debug)
        ostrack = FlexTrackV2Tracker(params,dataset_name)  # "VisEvent"
        tracker = ViPT_RGBE(tracker=ostrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+ seq_name +'——————————————')
    RGB_img_list, E_img_list, RGB_gt, absent_label = genConfig(seq_path, dataset_name)
    if absent_label[0] == 0: # first frame is absent in some seqs
        first_present_idx = absent_label.argmax()
        RGB_img_list = RGB_img_list[first_present_idx:]
        E_img_list = E_img_list[first_present_idx:]
        RGB_gt = RGB_gt[first_present_idx:]
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    last_region = result[0]

    toc = 0
    if "miss" not in dataset_name:
        for frame_idx, (rgb_path, E_path) in enumerate(zip(RGB_img_list, E_img_list)):
            tic = cv2.getTickCount()
            if frame_idx == 0:
                # initialization  cuz missing modalities does not include first frame missing so keep the same with other trackers.
                image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                tracker.initialize(image, RGB_gt[0].tolist())  # xywh
            elif frame_idx > 0:
    

                image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                region, confidence = tracker.track(image)  # xywh

                last_region = region
                result[frame_idx] = np.array(region)
            toc += cv2.getTickCount() - tic
    else:
        for frame_idx, (rgb_path, E_path) in enumerate(zip(RGB_img_list, E_img_list)):
            tic = cv2.getTickCount()
            if frame_idx == 0:
                # initialization  cuz missing modalities does not include first frame missing so keep the same with other trackers.
                image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                tracker.initialize(image, RGB_gt[0].tolist())  # xywh
            elif frame_idx > 0:
                if miss_index[seq_name]["data"][frame_idx]==[1.0,1.0]:
                    image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    region, confidence = tracker.track(image,miss_index[seq_name]["data"][frame_idx])  # xywh
                    last_region = result[frame_idx] = np.array(region)

                elif miss_index[seq_name]["data"][frame_idx]==[1.0,0.0]:
                    image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    image[:,:,3:] = 0
                    region, confidence = tracker.track(image,miss_index[seq_name]["data"][frame_idx])  # xywh
                    last_region = result[frame_idx] = np.array(region)
                elif miss_index[seq_name]["data"][frame_idx]==[0.0,1.0]:
                    image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
                    image[:,:,:3] = 0
                    region, confidence = tracker.track(image,miss_index[seq_name]["data"][frame_idx])  # xywh
                    last_region = result[frame_idx] = np.array(region)  # updatse last_region for double-missing case

                elif miss_index[seq_name]["data"][frame_idx]==[0.0,0.0]:
                        result[frame_idx] = last_region
            toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class ViPT_RGBE(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, missing=[1,1]):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB,missing)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBE dataset.')
    parser.add_argument('--script_name', type=str, default='flextrackv2', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='flextrackv2_b224_3', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='VisEvent', help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=120, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    # path initialization
    seq_list = None

    if 'VisEvent' in dataset_name:
        seq_home = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test'
        with open(join(seq_home, 'testlist.txt'), 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
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


