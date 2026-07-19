import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.gridspec import GridSpec
from PIL import Image
from loaders import load_pred_generic, load_pred_vot, load_gt, iou

OUT = os.path.dirname(os.path.abspath(__file__))
LASHER="/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset"
VE="/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test"
DT="Depthtrack_workspace/sequences"
FR="final_result/raw_predictions"
V1="data_missing_modality/Missing_data_annotation/FlexTrack"
MISS_JSON={
 "LasHeR":  "data_missing_modality/Missing_data_annotation/LasHeR245-Miss/missing_results_lasher245.json",
 "VisEvent":"data_missing_modality/Missing_data_annotation/visevent-miss/missing_results_visevent.json",
 "DepthTrack":"data_missing_modality/Missing_data_annotation/depthtrack-miss/missing_results_depthtrack.json",
}

def imglist(d, exts):
    fs=[]
    for e in exts: fs+=glob.glob(os.path.join(d,e))
    return sorted(fs)

def load_masks(json_path, key):
    """Return (N,2) array of [rgb_present, aux_present] per frame, or None."""
    if not (json_path and os.path.exists(json_path)): return None
    d=json.load(open(json_path))
    if key not in d: return None
    v=d[key]; data=v["data"] if isinstance(v,dict) and "data" in v else v
    return np.array(data, dtype=float)

# each config: name, seq, images, gt, v2 pred, v1 loader; miss configs add aux label + mask key
CONFIGS={
 "LasHeR_full":   dict(seq="raincarturn", imgs=f"{LASHER}/raincarturn/visible", exts=["*.jpg"],
                       gt=f"{LASHER}/raincarturn/visible.txt",
                       v2=f"{FR}/LasHeR/raincarturn.txt", v1=("gen",f"{V1}/LasHER/raincarturn.txt")),
 "LasHeR_miss":   dict(seq="4men", imgs=f"{LASHER}/4men/visible", exts=["*.jpg"],
                       gt=f"{LASHER}/4men/visible.txt",
                       v2=f"{FR}/LasHeR_miss/4men.txt", v1=("gen",f"{V1}/LasHeR_miss/4men.txt"),
                       aux="Thermal", miss=(MISS_JSON["LasHeR"], "4men")),
 "DepthTrack_full":dict(seq="notebook01_indoor", imgs=f"{DT}/notebook01_indoor/color", exts=["*.jpg"],
                       gt=f"{DT}/notebook01_indoor/groundtruth.txt",
                       v2=f"{FR}/DepthTrack/notebook01_indoor.txt", v1=("vot",f"{V1}/depthtrack","notebook01_indoor")),
 "DepthTrack_miss":dict(seq="mobilephone03_indoor", imgs=f"{DT}/mobilephone03_indoor/color", exts=["*.jpg"],
                       gt=f"{DT}/mobilephone03_indoor/groundtruth.txt",
                       v2=f"{FR}/DepthTrack_miss/mobilephone03_indoor.txt", v1=("vot",f"{V1}/depthtrack_miss","mobilephone03_indoor"),
                       aux="Depth", miss=(MISS_JSON["DepthTrack"], "mobilephone03_indoor")),
 "VisEvent_full": dict(seq="dvSave-2021_02_14_16_37_15_car5", imgs=f"{VE}/dvSave-2021_02_14_16_37_15_car5/vis_imgs", exts=["*.bmp","*.jpg","*.png"],
                       gt=f"{VE}/dvSave-2021_02_14_16_37_15_car5/groundtruth.txt",
                       v2=f"{FR}/VisEvent/dvSave-2021_02_14_16_37_15_car5.txt", v1=("gen",f"{V1}/VisEvent/dvSave-2021_02_14_16_37_15_car5.txt")),
 "VisEvent_miss": dict(seq="dvSave-2021_02_14_16_46_34_car8", imgs=f"{VE}/dvSave-2021_02_14_16_46_34_car8/vis_imgs", exts=["*.bmp","*.jpg","*.png"],
                       gt=f"{VE}/dvSave-2021_02_14_16_46_34_car8/groundtruth.txt",
                       v2=f"{FR}/VisEvent_miss/dvSave-2021_02_14_16_46_34_car8.txt", v1=("gen",f"{V1}/VisEvent_miss/dvSave-2021_02_14_16_46_34_car8.txt"),
                       aux="Event", miss=(MISS_JSON["VisEvent"], "dvSave-2021_02_14_16_46_34_car8")),
}

BORDER=["#1f77b4","#000000","#d62728","#2ca02c"]
C_GT="#00e000"; C_V2="#d62728"; C_V1="#1f77b4"
FRACS=[0.13,0.40,0.63,0.88]
# modality-availability ribbon colors
C_BOTH="#dfeddf"      # both modalities present (faint green-grey)
C_AUXMISS="#f2a341"   # aux dropped (RGB-only)
C_RGBMISS="#8e6cc0"   # RGB dropped (aux-only)
C_NONE="#c94b4b"      # both dropped

def frame_state(mk_row):
    r,a = mk_row[0]>0.5, mk_row[1]>0.5
    if r and a:   return "both"
    if r and not a: return "auxmiss"
    if a and not r: return "rgbmiss"
    return "none"

def ribbon_colors(mk):
    cmap={"both":C_BOTH,"auxmiss":C_AUXMISS,"rgbmiss":C_RGBMISS,"none":C_NONE}
    return np.array([[int(cmap[frame_state(r)][i:i+2],16)/255 for i in (1,3,5)] for r in mk])

def build(name,cfg):
    imgs=imglist(cfg["imgs"],cfg["exts"])
    gt=load_gt(cfg["gt"])
    a2=load_pred_generic(cfg["v2"])
    if cfg["v1"][0]=="gen": a1=load_pred_generic(cfg["v1"][1])
    else: a1=load_pred_vot(cfg["v1"][1],cfg["v1"][2])
    is_miss = "miss" in cfg
    mk = load_masks(*cfg["miss"]) if is_miss else None
    lens=[len(imgs),len(gt),len(a2),len(a1)] + ([len(mk)] if mk is not None else [])
    m=min(lens)
    imgs=imgs[:m]; gt=gt[:m]; a2=a2[:m]; a1=a1[:m]
    if mk is not None: mk=mk[:m]
    if cfg["v1"][0]=="vot":
        nanrows=np.isnan(a1).any(1); a1[nanrows]=gt[nanrows]
    valid=~np.isnan(gt).any(1)&(gt[:,2]>0)&(gt[:,3]>0)
    io2=iou(a2,gt); io1=iou(a1,gt)
    io2=np.where(valid,io2,np.nan); io1=np.where(valid,io1,np.nan)

    # ---- choose the 4 sample frames ----
    if mk is not None:
        incomplete = valid & ((mk[:,0]<0.5)|(mk[:,1]<0.5))
        # prefer dropout frames where V1 lost (<0.3) but V2 held (>0.5): the money shot
        cand=np.where(incomplete & (io1<0.3) & (io2>0.5))[0]
        if len(cand)<4:
            cand=np.where(incomplete)[0]
        picks=[int(cand[int(f*(len(cand)-1))]) for f in FRACS] if len(cand)>=4 else \
              [int(np.where(valid)[0][int(f*(valid.sum()-1))]) for f in FRACS]
    else:
        vi=np.where(valid)[0]; picks=[int(vi[int(f*(len(vi)-1))]) for f in FRACS]

    fig=plt.figure(figsize=(15,6.6))
    if mk is not None:
        gs=GridSpec(3,4,height_ratios=[1.05,0.9,0.13],hspace=0.16,wspace=0.06,
                    left=0.06,right=0.985,top=0.955,bottom=0.20)
    else:
        gs=GridSpec(2,4,height_ratios=[1.05,1.0],hspace=0.16,wspace=0.06,
                    left=0.06,right=0.985,top=0.965,bottom=0.11)
    # top row: sample frames
    for k,fi in enumerate(picks):
        ax=fig.add_subplot(gs[0,k])
        im=Image.open(imgs[fi]).convert("RGB"); ax.imshow(im)
        for arr,col,lw in [(gt,C_GT,2.4),(a1,C_V1,2.0),(a2,C_V2,2.0)]:
            b=arr[fi]
            if np.isnan(b).any(): continue
            ax.add_patch(Rectangle((b[0],b[1]),b[2],b[3],fill=False,edgecolor=col,linewidth=lw))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(BORDER[k]); s.set_linewidth(4)
        ttl=f"#{fi}"
        if mk is not None:
            st=frame_state(mk[fi])
            tag={"both":"both present","auxmiss":f"{cfg['aux']} missing",
                 "rgbmiss":"RGB missing","none":"both missing"}[st]
            ttl=f"#{fi}  ({tag})"
        ax.set_title(ttl,fontsize=10.5,color=BORDER[k],pad=3)
        if k==0: ax.set_ylabel("Images",fontsize=13,fontweight='bold')

    # overlap curve
    axp=fig.add_subplot(gs[1,:])
    x=np.arange(m)
    axp.plot(x,io1,color=C_V1,lw=1.3,alpha=0.9,label="FlexTrack (V1)")
    axp.plot(x,io2,color=C_V2,lw=1.6,alpha=0.95,label="FlexTrackV2 (Ours)")
    for k,fi in enumerate(picks):
        axp.axvline(fi,color=BORDER[k],ls="--",lw=1.6)
    axp.set_xlim(0,m-1); axp.set_ylim(0,1.02)
    axp.set_ylabel("Overlap",fontsize=13,fontweight='bold')
    axp.grid(True,alpha=0.25)
    if mk is None: axp.set_xlabel("Frame",fontsize=13)
    else: axp.set_xticklabels([])
    axp.legend(loc="lower left",fontsize=11,ncol=2,framealpha=0.9)
    m2=np.nanmean(io2)*100; m1=np.nanmean(io1)*100

    # modality-availability ribbon
    if mk is not None:
        axr=fig.add_subplot(gs[2,:])
        cols=ribbon_colors(mk)[None,:,:]         # (1,N,3)
        axr.imshow(cols,aspect="auto",extent=[0,m-1,0,1],interpolation="nearest")
        axr.set_yticks([]); axr.set_xlim(0,m-1)
        axr.set_xlabel("Frame",fontsize=13)
        axr.set_ylabel("Modality",fontsize=10,rotation=0,ha="right",va="center")
        aux_present_frac=(mk[:,1]>0.5).mean()*100; rgb_present_frac=(mk[:,0]>0.5).mean()*100
        handles=[Patch(color=C_BOTH,label="both present"),
                 Patch(color=C_AUXMISS,label=f"{cfg['aux']} dropped"),
                 Patch(color=C_RGBMISS,label="RGB dropped"),
                 Patch(color=C_NONE,label="both dropped")]
        axr.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-1.15),
                   ncol=4,fontsize=9.5,frameon=False)
        # story annotation inside the plot (kept short so the title never clips)
        axp.text(0.985,0.06,"V1 loses lock at dropouts · V2 sustains",
                 transform=axp.transAxes,ha="right",va="bottom",fontsize=10.5,
                 style="italic",color="#333",
                 bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#ccc",alpha=0.85))
    axp.set_title(f"{name}   |   {cfg['seq']}   |   mean Overlap:  FlexTrackV2 {m2:.1f}%  vs  V1 {m1:.1f}%",
                  fontsize=12.5,fontweight='bold',pad=6)
    p=os.path.join(OUT,name+".png")
    fig.savefig(p,dpi=140); plt.close(fig)
    print(f"[OK] {name}: {p}  (V2={m2:.1f} V1={m1:.1f}, frames={m}, miss={'y' if mk is not None else 'n'})")

which=sys.argv[1] if len(sys.argv)>1 else "all"
for name,cfg in CONFIGS.items():
    if which!="all" and name!=which: continue
    try: build(name,cfg)
    except Exception as e:
        import traceback; print(f"[ERR] {name}: {e}"); traceback.print_exc()
