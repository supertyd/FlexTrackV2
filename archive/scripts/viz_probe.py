"""Probe: build a tracker, hook all MoEFusion modules, run one LasHeR frame
full vs aux-missing, confirm we can capture fused features + hallucinated_aux."""
import os, sys
sys.path.insert(0, '/mnt/task_runtime')
os.chdir('/mnt/task_runtime')
import numpy as np, torch
from lib.train.dataset.depth_utils import get_x_frame
import lib.test.parameter.flextrackv2 as P
from lib.test.tracker.flextrackv2 import FlexTrackV2
from lib.models.flextrackv2.moe_fusion import MoEFusion

CFG = 'flextrackv2_b224_56_abl_rung0'
SEQ_HOME = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
SEQ = '1boycoming'

params = P.parameters(CFG, 40)
tracker = FlexTrackV2(params, 'LasHeR')

# hook every MoEFusion
caps = []
def mk(name):
    def hook(mod, inp, out):
        x = inp[0]
        y = out[0] if isinstance(out, tuple) else out
        # recompute hallucinated_aux / real aux for fidelity
        xr = x[:, :, :mod.input_size]; xa = x[:, :, mod.input_size:]
        hal_aux = mod.rgb_to_aux_hallucinater(xr)
        caps.append(dict(name=name, y=y.detach(), x_aux=xa.detach(), hal_aux=hal_aux.detach(),
                         L=y.shape[1], C=y.shape[-1], input_size=mod.input_size))
    return hook
n_moe = 0
for nm, m in tracker.network.named_modules():
    if isinstance(m, MoEFusion):
        m.register_forward_hook(mk(nm)); n_moe += 1
print(f"hooked {n_moe} MoEFusion modules")

# load frames
vis = sorted(os.listdir(f'{SEQ_HOME}/{SEQ}/visible'))
inf = sorted(os.listdir(f'{SEQ_HOME}/{SEQ}/infrared'))
gt = np.loadtxt(f'{SEQ_HOME}/{SEQ}/visible.txt', delimiter=',')
print(f"seq {SEQ}: {len(vis)} frames, gt {gt.shape}")

def frame(i, zero_aux=False):
    img = get_x_frame(f'{SEQ_HOME}/{SEQ}/visible/{vis[i]}',
                      f'{SEQ_HOME}/{SEQ}/infrared/{inf[i]}',
                      dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
    img = np.array(img)
    if zero_aux:
        img[:, :, 3:] = 0
    return img

tracker.initialize(frame(0), {'init_bbox': gt[0].tolist()})

# full pass on frame 5
caps.clear()
tracker.track(frame(5), [1.0, 1.0])
print("\nFULL pass captures:")
for c in caps:
    print(f"  {c['name']:40s} y={tuple(c['y'].shape)} input_size={c['input_size']}")

print("\nlast-layer fused y shape:", tuple(caps[-1]['y'].shape))
print("token split: first 196 = search(14x14), last 49 = template(7x7)?  L =", caps[-1]['L'])
