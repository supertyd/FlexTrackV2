"""Inference-time MISSING-RATE degradation curve on VisEvent.
Same trained model, but at inference the auxiliary (event) modality is dropped
per-frame with probability `rate` (RGB always present; first frame always full).
Test-time thresholds stay on the standard VisEvent_miss config for consistency.
Saves to workspace/results/VisEvent_mr/<tag>/<yaml>/<seq>.txt so each rate is separate.
"""
import os, sys, argparse, time, json
from os.path import join, dirname, abspath
import numpy as np
import multiprocessing, torch, cv2
prj = join(dirname(__file__), '..')
if prj not in sys.path: sys.path.append(prj)
from lib.test.tracker.flextrackv2 import FlexTrackV2Tracker
import lib.test.parameter.flextrackv2 as rgbe_prompt_params
from lib.train.dataset.depth_utils import get_x_frame

def genConfig(seq_path):
    RGB = sorted([seq_path+'/vis_imgs/'+p for p in os.listdir(seq_path+'/vis_imgs') if p.endswith('.bmp')])
    E   = sorted([seq_path+'/event_imgs/'+p for p in os.listdir(seq_path+'/event_imgs') if p.endswith('.bmp')])
    gt  = np.loadtxt(seq_path+'/groundtruth.txt', delimiter=',')
    absent = np.loadtxt(seq_path+'/absent_label.txt')
    return RGB, E, gt, absent

def run_sequence(seq_name, seq_home, yaml_name, rate, tag, num_gpu, epoch):
    try:
        wid = int(multiprocessing.current_process().name.split('-')[-1]) - 1
        torch.cuda.set_device(wid % num_gpu)
    except Exception:
        pass
    save_dir = f'./workspace/results/VisEvent_mr/{tag}/{yaml_name}'
    save_path = f'{save_dir}/{seq_name}.txt'
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}'); return
    try:
        params = rgbe_prompt_params.parameters(yaml_name, epoch)
        tracker_core = FlexTrackV2Tracker(params, 'VisEvent_miss')      # thresholds: standard miss config
        dtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
        RGB, E, gt, absent = genConfig(f'{seq_home}/{seq_name}')
        n = len(gt) if len(RGB) == len(gt) else len(RGB)
        result = np.zeros((n, 4), dtype=gt.dtype); result[0] = gt[0]
        # per-frame drop pattern (reproducible): event dropped w.p. rate, RGB always kept, frame0 full
        rs = np.random.RandomState((abs(hash(seq_name)) + int(round(rate*1000))) & 0xffffffff)
        drop = rs.random_sample(n) < rate
        for i in range(min(n, len(RGB))):
            rgb_p, e_p = RGB[i], E[i]
            if i == 0:
                img = get_x_frame(rgb_p, e_p, dtype=dtype)
                tracker_core.initialize(img, {'init_bbox': gt[0].tolist()}); continue
            img = get_x_frame(rgb_p, e_p, dtype=dtype)
            if drop[i]:
                img[:, :, 3:] = 0                              # zero the event channels
                out = tracker_core.track(img, [1.0, 0.0])
            else:
                out = tracker_core.track(img, [1.0, 1.0])
            result[i] = np.array(out['target_bbox'])
        np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
        print(f'done {seq_name} (drop {drop.mean():.2f})')
    except Exception as e:
        import traceback; print(f'[SEQ-FAILED] {seq_name}: {e}'); traceback.print_exc()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml_name', required=True)
    ap.add_argument('--rate', type=float, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--threads', type=int, default=torch.cuda.device_count())
    ap.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    ap.add_argument('--epoch', type=int, default=40)
    ap.add_argument('--stride', type=int, default=1, help='take every k-th seq (subset preview)')
    a = ap.parse_args()
    seq_home = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test'
    seqs = [l.strip() for l in open(join(seq_home, 'testlist.txt')) if l.strip()]
    seqs.sort()
    seqs = seqs[::a.stride]
    print(f'evaluating {len(seqs)} seqs (stride={a.stride})')
    args = [(s, seq_home, a.yaml_name, a.rate, a.tag, a.num_gpus, a.epoch) for s in seqs]
    t0 = time.time()
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=a.threads) as pool:
        pool.starmap(run_sequence, args)
    print(f'rate={a.rate} tag={a.tag} done in {time.time()-t0:.0f}s')
