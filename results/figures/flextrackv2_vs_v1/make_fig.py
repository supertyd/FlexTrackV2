import sys, os, glob
sys.path.insert(0,"/mnt/tmp/claude-0/-mnt-task-runtime/3ebbad2c-2577-437f-9b2e-1dc89c08c5c7/scratchpad")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from PIL import Image
from loaders import load_pred_generic, load_pred_vot, load_gt, iou

OUT="/mnt/task_runtime/figures/flextrackv2_vs_v1"
os.makedirs(OUT, exist_ok=True)
LASHER="/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset"
VE="/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test"
DT="Depthtrack_workspace/sequences"
FR="final_result/raw_predictions"
V1="data_missing_modality/Missing_data_annotation/FlexTrack"

def imglist(d, exts):
    fs=[]
    for e in exts: fs+=glob.glob(os.path.join(d,e))
    return sorted(fs)

# each config: name, seq, images dir + exts, gt path, v2 pred path, v1 loader
CONFIGS={
 "LasHeR_full":   dict(seq="drillmaster1117", imgs=f"{LASHER}/drillmaster1117/visible", exts=["*.jpg"],
                       gt=f"{LASHER}/drillmaster1117/visible.txt",
                       v2=f"{FR}/LasHeR/drillmaster1117.txt", v1=("gen",f"{V1}/LasHER/drillmaster1117.txt")),
 "LasHeR_miss":   dict(seq="4men", imgs=f"{LASHER}/4men/visible", exts=["*.jpg"],
                       gt=f"{LASHER}/4men/visible.txt",
                       v2=f"{FR}/LasHeR_miss/4men.txt", v1=("gen",f"{V1}/LasHeR_miss/4men.txt")),
 "DepthTrack_full":dict(seq="notebook01_indoor", imgs=f"{DT}/notebook01_indoor/color", exts=["*.jpg"],
                       gt=f"{DT}/notebook01_indoor/groundtruth.txt",
                       v2=f"{FR}/DepthTrack/notebook01_indoor.txt", v1=("vot",f"{V1}/depthtrack","notebook01_indoor")),
 "DepthTrack_miss":dict(seq="mobilephone03_indoor", imgs=f"{DT}/mobilephone03_indoor/color", exts=["*.jpg"],
                       gt=f"{DT}/mobilephone03_indoor/groundtruth.txt",
                       v2=f"{FR}/DepthTrack_miss/mobilephone03_indoor.txt", v1=("vot",f"{V1}/depthtrack_miss","mobilephone03_indoor")),
 "VisEvent_full": dict(seq="dvSave-2021_02_14_16_37_15_car5", imgs=f"{VE}/dvSave-2021_02_14_16_37_15_car5/vis_imgs", exts=["*.bmp","*.jpg","*.png"],
                       gt=f"{VE}/dvSave-2021_02_14_16_37_15_car5/groundtruth.txt",
                       v2=f"{FR}/VisEvent/dvSave-2021_02_14_16_37_15_car5.txt", v1=("gen",f"{V1}/VisEvent/dvSave-2021_02_14_16_37_15_car5.txt")),
 "VisEvent_miss": dict(seq="dvSave-2021_02_14_16_46_34_car8", imgs=f"{VE}/dvSave-2021_02_14_16_46_34_car8/vis_imgs", exts=["*.bmp","*.jpg","*.png"],
                       gt=f"{VE}/dvSave-2021_02_14_16_46_34_car8/groundtruth.txt",
                       v2=f"{FR}/VisEvent_miss/dvSave-2021_02_14_16_46_34_car8.txt", v1=("gen",f"{V1}/VisEvent_miss/dvSave-2021_02_14_16_46_34_car8.txt")),
}

BORDER=["#1f77b4","#000000","#d62728","#2ca02c"]  # blue black red green markers
C_GT="#00e000"; C_V2="#d62728"; C_V1="#1f77b4"   # box colors
FRACS=[0.13,0.40,0.63,0.88]

def build(name,cfg):
    imgs=imglist(cfg["imgs"],cfg["exts"])
    gt=load_gt(cfg["gt"])
    a2=load_pred_generic(cfg["v2"])
    if cfg["v1"][0]=="gen": a1=load_pred_generic(cfg["v1"][1])
    else: a1=load_pred_vot(cfg["v1"][1],cfg["v1"][2])
    m=min(len(imgs),len(gt),len(a2),len(a1))
    imgs=imgs[:m]; gt=gt[:m]; a2=a2[:m]; a1=a1[:m]
    # V1 depthtrack init placeholder -> gt
    if cfg["v1"][0]=="vot":
        nanrows=np.isnan(a1).any(1); a1[nanrows]=gt[nanrows]
    valid=~np.isnan(gt).any(1)&(gt[:,2]>0)&(gt[:,3]>0)
    io2=iou(a2,gt); io1=iou(a1,gt)
    io2=np.where(valid,io2,np.nan); io1=np.where(valid,io1,np.nan)
    # choose sample frames at fractions, snapped to valid present frames
    valididx=np.where(valid)[0]
    picks=[int(valididx[int(f*(len(valididx)-1))]) for f in FRACS]

    fig=plt.figure(figsize=(15,6.2))
    gs=GridSpec(2,4,height_ratios=[1.05,1.0],hspace=0.16,wspace=0.06,
                left=0.06,right=0.985,top=0.965,bottom=0.11)
    for k,fi in enumerate(picks):
        ax=fig.add_subplot(gs[0,k])
        im=Image.open(imgs[fi]).convert("RGB"); ax.imshow(im)
        for arr,col,lw in [(gt,C_GT,2.4),(a1,C_V1,2.0),(a2,C_V2,2.0)]:
            b=arr[fi]
            if np.isnan(b).any(): continue
            ax.add_patch(Rectangle((b[0],b[1]),b[2],b[3],fill=False,edgecolor=col,linewidth=lw))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(BORDER[k]); s.set_linewidth(4)
        ax.set_title(f"#{fi}",fontsize=11,color=BORDER[k],pad=3)
        if k==0: ax.set_ylabel("Images",fontsize=13,fontweight='bold')
    axp=fig.add_subplot(gs[1,:])
    x=np.arange(m)
    axp.plot(x,io1,color=C_V1,lw=1.3,alpha=0.9,label="FlexTrack (V1)")
    axp.plot(x,io2,color=C_V2,lw=1.6,alpha=0.95,label="FlexTrackV2 (Ours)")
    for k,fi in enumerate(picks):
        axp.axvline(fi,color=BORDER[k],ls="--",lw=1.6)
    axp.set_xlim(0,m-1); axp.set_ylim(0,1.02)
    axp.set_xlabel("Frame",fontsize=13); axp.set_ylabel("Overlap",fontsize=13,fontweight='bold')
    axp.grid(True,alpha=0.25)
    axp.legend(loc="lower left",fontsize=11,ncol=2,framealpha=0.9)
    m2=np.nanmean(io2)*100; m1=np.nanmean(io1)*100
    axp.set_title(f"{name}   |   {cfg['seq']}   |   mean Overlap:  FlexTrackV2 {m2:.1f}%  vs  V1 {m1:.1f}%",
                  fontsize=12,fontweight='bold',pad=6)
    p=os.path.join(OUT,name+".png")
    fig.savefig(p,dpi=140); plt.close(fig)
    print(f"[OK] {name}: {p}  (V2={m2:.1f} V1={m1:.1f}, frames={m})")

which=sys.argv[1] if len(sys.argv)>1 else "all"
for name,cfg in CONFIGS.items():
    if which!="all" and name!=which: continue
    try: build(name,cfg)
    except Exception as e:
        import traceback; print(f"[ERR] {name}: {e}"); traceback.print_exc()
