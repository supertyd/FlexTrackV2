"""MoE router-focused visualization.

Does the heterogeneous MoE router SPECIALIZE by which modality is present?
For each frame we run the SAME crop under 3 conditions and read the router's
8-way expert distribution:
    full      = [1,1]   both modalities
    aux-miss  = [1,0]   thermal/aux zeroed
    rgb-miss  = [0,1]   RGB zeroed

Processing:
  * expert-usage matrix  [3 conditions x 8 experts]  (mean gate)
  * router entropy per condition  (H = -sum p log p ; low = concentrated/specialised)
  * routing shift  = 1 - cosine(full, missing)   (how much routing moves)

Compared rung0 (real HMoE) vs no_hallucinate (zero-fill) -- both BMR_HMoE.
Inference-only, existing checkpoints. Reuses viz_experiment's patched forward.
"""
import os, sys, json
sys.path.insert(0, '/mnt/task_runtime')
os.chdir('/mnt/task_runtime')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
# importing applies the MoEFusion.forward monkeypatch + gives helpers
from viz_experiment import build, frames, make_frame, sync_state, SEQ_HOME, OUT
from lib.train.dataset.depth_utils import get_x_frame

CONFIGS = ['flextrackv2_b224_56_abl_rung0', 'flextrackv2_b224_56_abl_no_hallucinate']
LABELS = {'flextrackv2_b224_56_abl_rung0': 'rung0 (HMoE, hallucinate-fill)',
          'flextrackv2_b224_56_abl_no_hallucinate': 'no_hallucinate (zero-fill)'}
CONDS = ['full', 'aux-miss', 'rgb-miss']


def frame_cond(seq, vis, inf, i, xtype, cond):
    img = np.array(get_x_frame(f'{SEQ_HOME}/{seq}/visible/{vis[i]}',
                               f'{SEQ_HOME}/{seq}/infrared/{inf[i]}', dtype=xtype))
    img = img.copy()
    if cond == 'aux-miss':
        img[:, :, 3:] = 0
    elif cond == 'rgb-miss':
        img[:, :, :3] = 0
    return img


MISS = {'full': [1.0, 1.0], 'aux-miss': [1.0, 0.0], 'rgb-miss': [0.0, 1.0]}


def entropy(p):
    p = np.clip(p / (p.sum() + 1e-9), 1e-9, 1)
    return float(-(p * np.log(p)).sum())


def run_router(cfg, seqs, nf):
    # one driver tracker (full) + one probe tracker re-synced each frame/condition
    tkD, params, moeD = build(cfg)
    tkP, _, moeP = build(cfg)
    xtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
    usage = {c: [] for c in CONDS}         # list of [8] gate vectors
    for seq in seqs:
        vis, inf, gt = frames(seq)
        n = min(nf, len(vis) - 1)
        tkD.initialize(frame_cond(seq, vis, inf, 0, xtype, 'full'), {'init_bbox': gt[0].tolist()})
        for i in range(1, n + 1):
            with torch.no_grad():
                # advance driver on full modality
                prev_state = list(tkD.state)
                tkD.track(frame_cond(seq, vis, inf, i, xtype, 'full'), MISS['full'])
                gfull = moeD._cap['gates'].float().cpu().numpy().reshape(-1)
                usage['full'].append(gfull)
                # probe the SAME crop (prev_state) under the two missing conditions
                for c in ['aux-miss', 'rgb-miss']:
                    tkP.state = list(prev_state)
                    for a in ['template_list', 'template_anno_list', 'missing_list',
                              'memory_template_list', 'memory_template_anno_list', 'memory_missing_list']:
                        if hasattr(tkD, a):
                            import copy; setattr(tkP, a, copy.copy(getattr(tkD, a)))
                    tkP.track(frame_cond(seq, vis, inf, i, xtype, c), MISS[c])
                    usage[c].append(moeP._cap['gates'].float().cpu().numpy().reshape(-1))
        print(f"  [{cfg.split('abl_')[-1]:14s}] {seq:22s} done", flush=True)
    mat = np.stack([np.stack(usage[c]).mean(0) for c in CONDS])   # [3, 8]
    ent = {c: entropy(np.stack(usage[c]).mean(0)) for c in CONDS}
    # routing shift: mean over frames of 1 - cos(full_t, miss_t)
    shift = {}
    for c in ['aux-miss', 'rgb-miss']:
        F = np.stack(usage['full']); M = np.stack(usage[c])
        cs = (F * M).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(M, axis=1) + 1e-9)
        shift[c] = float((1 - cs).mean())
    return dict(mat=mat, ent=ent, shift=shift, n=len(usage['full']))


if __name__ == '__main__':
    N_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    NF = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    ALL = sorted([d for d in os.listdir(SEQ_HOME) if os.path.isdir(f'{SEQ_HOME}/{d}')])
    SEQS = ALL[:N_SEQ]
    print(f"router analysis on {len(SEQS)} seqs x {NF} frames", flush=True)

    res = {}
    for cfg in CONFIGS:
        res[cfg] = run_router(cfg, SEQS, NF)
        r = res[cfg]
        print(f"=== {cfg}: entropy {[(c, round(r['ent'][c],3)) for c in CONDS]}  "
              f"shift {[(c, round(r['shift'][c],3)) for c in ['aux-miss','rgb-miss']]} ===", flush=True)

    # ---------- figure ----------
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(16, 4.4 * len(CONFIGS)))
    for row, cfg in enumerate(CONFIGS):
        r = res[cfg]
        # (a) expert-usage heatmap: conditions x experts
        ax = axes[row, 0]
        im = ax.imshow(r['mat'], cmap='rocket' if False else 'YlOrRd', aspect='auto')
        ax.set_yticks(range(3)); ax.set_yticklabels(CONDS)
        ax.set_xticks(range(r['mat'].shape[1])); ax.set_xlabel('expert')
        ax.set_title(f"{LABELS[cfg]}\nexpert usage by condition")
        for (yy, xx), v in np.ndenumerate(r['mat']):
            ax.text(xx, yy, f"{v:.2f}", ha='center', va='center', fontsize=7,
                    color='black' if v < r['mat'].max() * 0.6 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.046)
        # (b) grouped bar of usage
        ax = axes[row, 1]
        ne = r['mat'].shape[1]; xx = np.arange(ne)
        for j, c in enumerate(CONDS):
            ax.bar(xx + (j - 1) * 0.27, r['mat'][j], 0.27, label=c)
        ax.set_xlabel('expert'); ax.set_ylabel('mean gate'); ax.legend(fontsize=8)
        ax.set_title(f"{LABELS[cfg]}\nexpert distribution")
        # (c) entropy + shift text
        ax = axes[row, 2]; ax.axis('off')
        t = (f"router entropy (lower = more specialised):\n"
             f"   full     = {r['ent']['full']:.3f}\n"
             f"   aux-miss = {r['ent']['aux-miss']:.3f}\n"
             f"   rgb-miss = {r['ent']['rgb-miss']:.3f}\n\n"
             f"routing shift (1 - cos vs full):\n"
             f"   aux-miss = {r['shift']['aux-miss']:.3f}\n"
             f"   rgb-miss = {r['shift']['rgb-miss']:.3f}\n\n"
             f"(n = {r['n']} frames)")
        ax.text(0.0, 0.5, t, fontsize=11, va='center', family='monospace')
    plt.suptitle('MoE router specialisation by present modality', fontsize=14)
    plt.tight_layout()
    out = f'{OUT}/viz_router.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"saved {out}", flush=True)
    json.dump({cfg: {'mat': res[cfg]['mat'].tolist(), 'ent': res[cfg]['ent'],
                     'shift': res[cfg]['shift'], 'n': res[cfg]['n']} for cfg in CONFIGS},
              open(f'{OUT}/viz_router.json', 'w'))
    print("done", flush=True)
