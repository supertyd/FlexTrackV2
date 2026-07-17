import numpy as np, os, glob
def _load(p):
    txt=open(p).read().strip()
    first=txt.split('\n')[0]
    d=',' if ',' in first else None
    return np.atleast_2d(np.genfromtxt(p,delimiter=d))
def load_pred_generic(p):  # space/comma sep box files (V2 all, V1 lasher/visevent)
    return _load(p)
def load_pred_vot(seqdir,seq):  # V1 depthtrack: subfolder with _001.txt, first line "1"
    p=os.path.join(seqdir,seq,seq+"_001.txt")
    lines=[l.strip() for l in open(p) if l.strip()]
    rows=[]
    for l in lines:
        parts=l.replace(',',' ').split()
        if len(parts)==1:  # "1" init marker -> placeholder, fill later with gt
            rows.append([np.nan]*4)
        else:
            rows.append([float(x) for x in parts[:4]])
    return np.array(rows)
def load_gt(p):
    return _load(p)
def iou(pred,gt):
    px,py,pw,ph=pred.T[:4]; gx,gy,gw,gh=gt.T[:4]
    ax1=np.maximum(px,gx); ay1=np.maximum(py,gy)
    ax2=np.minimum(px+pw,gx+gw); ay2=np.minimum(py+ph,gy+gh)
    iw=np.clip(ax2-ax1,0,None); ih=np.clip(ay2-ay1,0,None)
    inter=iw*ih; union=pw*ph+gw*gh-inter
    with np.errstate(invalid='ignore'):
        r=np.where(union>0,inter/union,0.0)
    return r
