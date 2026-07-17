"""Eval-only probe: replace the EVENT auxiliary with a classical CANNY EDGE map
computed from the RGB frame, feed to the (event-trained) model, evaluate on VisEvent.
Tests the hypothesis that the event modality is ~redundant because event ~= edges,
and edges are derivable from RGB. Compare against RGB+event (R0) and RGB-only (event zeroed).
CAVEAT: the model was trained on event images, so edge input is out-of-distribution;
read as a probe, not a trained ablation.
"""
import os, sys, argparse, time
from os.path import join, dirname
import numpy as np
import multiprocessing, torch, cv2
prj = join(dirname(__file__), '..')
if prj not in sys.path: sys.path.append(prj)
from lib.test.tracker.flextrackv2 import FlexTrackV2
import lib.test.parameter.flextrackv2 as rgbe_prompt_params

def genConfig(seq_path):
    RGB = sorted([seq_path+'/vis_imgs/'+p for p in os.listdir(seq_path+'/vis_imgs') if p.endswith('.bmp')])
    E   = sorted([seq_path+'/event_imgs/'+p for p in os.listdir(seq_path+'/event_imgs') if p.endswith('.bmp')])
    gt  = np.loadtxt(seq_path+'/groundtruth.txt', delimiter=',')
    return RGB, E, gt

def rgb_edge_frame(rgb_path):
    """6-ch input: [RGB | Canny-edge-as-3ch]. Aux = classical edge of the RGB."""
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    edge3 = cv2.merge((edge, edge, edge))
    return cv2.merge((rgb, edge3))

def run_sequence(seq_name, seq_home, yaml_name, tag, num_gpu, epoch):
    try:
        wid = int(multiprocessing.current_process().name.split('-')[-1]) - 1
        torch.cuda.set_device(wid % num_gpu)
    except Exception: pass
    save_dir = f'./workspace/results/VisEvent_{tag}/{yaml_name}'
    save_path = f'{save_dir}/{seq_name}.txt'
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path): print(f'-1 {seq_name}'); return
    try:
        params = rgbe_prompt_params.parameters(yaml_name, epoch)
        tracker = FlexTrackV2(params, 'VisEvent')          # use VisEvent (gs) thresholds
        RGB, E, gt = genConfig(f'{seq_home}/{seq_name}')
        n = len(gt) if len(RGB) == len(gt) else len(RGB)
        result = np.zeros((n, 4), dtype=gt.dtype); result[0] = gt[0]
        for i in range(min(n, len(RGB))):
            img = rgb_edge_frame(RGB[i])                 # RGB + Canny-edge aux
            if i == 0:
                tracker.initialize(img, {'init_bbox': gt[0].tolist()})
            else:
                out = tracker.track(img, [1.0, 1.0])
                result[i] = np.array(out['target_bbox'])
        np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
        print(f'done {seq_name}')
    except Exception as e:
        import traceback; print(f'[SEQ-FAILED] {seq_name}: {e}'); traceback.print_exc()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml_name', required=True)
    ap.add_argument('--tag', default='edge')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    ap.add_argument('--epoch', type=int, default=40)
    ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    seq_home = '/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test'
    seqs = sorted(l.strip() for l in open(join(seq_home, 'testlist.txt')) if l.strip())[::a.stride]
    print(f'edge-eval {len(seqs)} seqs (stride={a.stride})')
    args = [(s, seq_home, a.yaml_name, a.tag, a.num_gpus, a.epoch) for s in seqs]
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=a.threads) as pool:
        pool.starmap(run_sequence, args)
    print(f'edge-eval done in {time.time():.0f}')
