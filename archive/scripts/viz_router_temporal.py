"""Temporal router behaviour, tied to the actual video content.

For one sequence, track full-modality and capture, per frame:
  * the 8-way MoE gate distribution  (which experts fire)
  * the predicted box + confidence

Figure:
  row 1  filmstrip of sampled RGB frames with the predicted box
  row 2  gate heatmap  [8 experts x T frames]  -- when does each expert fire
  row 3  router entropy + tracker confidence over time (full vs aux-missing)

This shows whether routing is CONTENT-DEPENDENT (switches with scene events)
rather than a static average. rung0 only. Inference-only.
"""
import os, sys, json
sys.path.insert(0, '/mnt/task_runtime'); os.chdir('/mnt/task_runtime')
import numpy as np, torch, cv2, copy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from viz_experiment import build, frames, sync_state, SEQ_HOME, OUT
from lib.train.dataset.depth_utils import get_x_frame

CFG = 'flextrackv2_b224_56_abl_rung0'


def frame_cond(seq, vis, inf, i, xtype, cond):
    img = np.array(get_x_frame(f'{SEQ_HOME}/{seq}/visible/{vis[i]}',
                               f'{SEQ_HOME}/{seq}/infrared/{inf[i]}', dtype=xtype)).copy()
    if cond == 'aux-miss':
        img[:, :, 3:] = 0
    return img


def entropy(p):
    p = np.clip(p / (p.sum() + 1e-9), 1e-9, 1)
    return float(-(p * np.log(p)).sum())


def run(seq, nf):
    tkD, params, moeD = build(CFG)     # full driver
    tkM, _, moeM = build(CFG)          # aux-missing probe (synced)
    xtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
    vis, inf, gt = frames(seq)
    nf = min(nf, len(vis) - 1)
    tkD.initialize(frame_cond(seq, vis, inf, 0, xtype, 'full'), {'init_bbox': gt[0].tolist()})
    tkM.initialize(frame_cond(seq, vis, inf, 0, xtype, 'full'), {'init_bbox': gt[0].tolist()})

    gates_full, gates_miss, boxes, conf_full, conf_miss = [], [], [], [], []
    for i in range(1, nf + 1):
        with torch.no_grad():
            sync_state(tkM, tkD)
            oF = tkD.track(frame_cond(seq, vis, inf, i, xtype, 'full'), [1.0, 1.0])
            oM = tkM.track(frame_cond(seq, vis, inf, i, xtype, 'aux-miss'), [1.0, 0.0])
        gates_full.append(moeD._cap['gates'].float().cpu().numpy().reshape(-1))
        gates_miss.append(moeM._cap['gates'].float().cpu().numpy().reshape(-1))
        boxes.append(list(tkD.state))
        conf_full.append(float(oF['best_score'])); conf_miss.append(float(oM['best_score']))
    return dict(seq=seq, vis=vis, inf=inf, nf=nf,
                gates_full=np.stack(gates_full), gates_miss=np.stack(gates_miss),
                boxes=boxes, conf_full=conf_full, conf_miss=conf_miss,
                gt=np.array(gt[:nf + 1]))


def draw(seq, vis, box, i):
    im = cv2.imread(f'{SEQ_HOME}/{seq}/visible/{vis[i]}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # crop a padded window around the target so the (often small) object is visible
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * 3.0
    x0 = int(max(0, cx - s)); y0 = int(max(0, cy - s))
    x1 = int(min(W, cx + s)); y1 = int(min(H, cy + s))
    crop = im[y0:y1, x0:x1]
    return crop if crop.size else im


def figure(r):
    seq, nf = r['seq'], r['nf']
    ne = r['gates_full'].shape[1]
    n_strip = 8
    idxs = np.linspace(0, nf - 1, n_strip).astype(int)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, n_strip, height_ratios=[2.2, 2.0, 2.0, 1.6], hspace=0.35, wspace=0.08)
    # row1 filmstrip
    for k, fi in enumerate(idxs):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(draw(seq, r['vis'], r['boxes'][fi], fi + 1))
        ax.set_title(f"f{fi+1}", fontsize=8); ax.axis('off')
    # row2 gate heatmap full
    axh = fig.add_subplot(gs[1, :])
    im = axh.imshow(r['gates_full'].T, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                    extent=[1, nf, ne - 0.5, -0.5])
    axh.set_ylabel('expert'); axh.set_yticks(range(ne))
    axh.set_title(f'{seq}: MoE gate weights over time (FULL modality)')
    for fi in idxs:
        axh.axvline(fi + 1, color='k', lw=0.5, alpha=0.3)
    plt.colorbar(im, ax=axh, fraction=0.02, pad=0.01)
    # row3 gate heatmap aux-miss
    axh2 = fig.add_subplot(gs[2, :])
    im2 = axh2.imshow(r['gates_miss'].T, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                      extent=[1, nf, ne - 0.5, -0.5])
    axh2.set_ylabel('expert'); axh2.set_yticks(range(ne))
    axh2.set_title(f'{seq}: MoE gate weights over time (AUX MISSING)')
    plt.colorbar(im2, ax=axh2, fraction=0.02, pad=0.01)
    # row4 routing divergence (full vs aux-missing) overlaid with GT-DERIVED
    # per-frame events. LasHeR gives only sequence-level attributes, so we
    # derive frame-level motion/scale from the ground-truth box to align router
    # response with objective events (honestly labelled GT-derived).
    axe = fig.add_subplot(gs[3, :])
    gf, gm = r['gates_full'], r['gates_miss']
    div = 1 - (gf * gm).sum(1) / (np.linalg.norm(gf, axis=1) * np.linalg.norm(gm, axis=1) + 1e-9)
    xs = np.arange(1, nf + 1)
    # GT-derived motion: center displacement / sqrt(area); scale: |log area ratio|
    gt = r['gt']; cen = gt[:, :2] + gt[:, 2:] / 2.0
    area = np.clip(gt[:, 2] * gt[:, 3], 1, None)
    disp = np.linalg.norm(np.diff(cen, axis=0), axis=1) / np.sqrt(area[1:])   # per frame 1..nf
    scale = np.abs(np.log(np.clip(area[1:] / area[:-1], 1e-3, 1e3)))
    def nrm(a):
        a = np.asarray(a, float); return (a - a.min()) / (a.ptp() + 1e-9)
    axe.plot(xs, div, color='#7b3294', lw=1.6, label='routing divergence (full vs aux-miss)')
    axe.fill_between(xs, 0, div, color='#7b3294', alpha=0.12)
    axe.plot(xs, nrm(disp[:nf]), color='#e8912a', lw=1.1, alpha=0.85, label='GT motion (norm.)')
    axe.plot(xs, nrm(scale[:nf]), color='#2a9d5c', lw=1.1, alpha=0.7, label='GT scale-change (norm.)')
    # mark the top motion frames
    top = np.argsort(disp[:nf])[-5:]
    for t in top:
        axe.axvline(t + 1, color='#e8912a', lw=0.8, ls=':', alpha=0.6)
    axe.set_ylabel('normalised'); axe.set_xlabel('frame')
    axe.set_ylim(0, 1.02); axe.legend(loc='upper left', fontsize=8, ncol=3)
    # correlation annotation
    from numpy import corrcoef
    c_mot = corrcoef(div, disp[:nf])[0, 1]
    axe.set_title(f'routing divergence vs GT-derived events   (corr with motion = {c_mot:.2f})', fontsize=9)
    plt.suptitle('Router behaviour along the video (rung0)', fontsize=14)
    out = f'{OUT}/viz_router_temporal_{seq}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"saved {out}", flush=True)
    return out


if __name__ == '__main__':
    seqs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['2runseven']
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    for s in seqs:
        r = run(s, nf)
        figure(r)
        print(f"{s}: mean entropy full={np.mean([entropy(g) for g in r['gates_full']]):.3f} "
              f"aux-miss={np.mean([entropy(g) for g in r['gates_miss']]):.3f}", flush=True)
    print("done", flush=True)
