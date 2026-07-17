import os, sys
sys.path.insert(0,'/mnt/task_runtime'); os.chdir('/mnt/task_runtime')
import numpy as np, torch, torch.nn.functional as F
from lib.train.dataset.depth_utils import get_x_frame
import lib.test.parameter.flextrackv2 as P
from lib.test.tracker.flextrackv2 import FlexTrackV2
from lib.models.flextrackv2.moe_fusion import MoEFusion
SH='/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'; SEQ='1boycoming'
params=P.parameters('flextrackv2_b224_56_abl_rung0',40); tk=FlexTrackV2(params,'LasHeR')
caps=[]
def hook(m,i,o):
    y=o[0] if isinstance(o,tuple) else o
    caps.append(y.detach()[0])
for _,m in tk.network.named_modules():
    if isinstance(m,MoEFusion): m.register_forward_hook(hook)
vis=sorted(os.listdir(f'{SH}/{SEQ}/visible')); inf=sorted(os.listdir(f'{SH}/{SEQ}/infrared'))
gt=np.loadtxt(f'{SH}/{SEQ}/visible.txt',delimiter=',')
xt=getattr(params.cfg.DATA,'XTYPE','rgbrgb')
def fr(i,z=False):
    img=np.array(get_x_frame(f'{SH}/{SEQ}/visible/{vis[i]}',f'{SH}/{SEQ}/infrared/{inf[i]}',dtype=xt))
    if z: img=img.copy(); img[:,:,3:]=0
    return img
tk.initialize(fr(0),{'init_bbox':gt[0].tolist()})
caps.clear(); tk.track(fr(5),[1.0,1.0]); Vf=caps[-1][:196]
caps.clear(); tk.track(fr(5,z=True),[1.0,0.0]); Vm=caps[-1][:196]
print("Vf shape",tuple(Vf.shape),"dtype",Vf.dtype)
print("Vf norm mean", Vf.float().norm(dim=-1).mean().item(), "Vm norm mean", Vm.float().norm(dim=-1).mean().item())
print("Vf[0,:5]", Vf[0,:5].float().cpu().numpy())
print("Vm[0,:5]", Vm[0,:5].float().cpu().numpy())
cs=F.cosine_similarity(Vf.float(),Vm.float(),dim=-1)
print("cosine per-token: mean",cs.mean().item(),"min",cs.min().item(),"max",cs.max().item())
print("num nan", torch.isnan(cs).sum().item())
