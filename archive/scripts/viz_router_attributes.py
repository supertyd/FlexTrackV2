"""Sequence-level: does the router respond to labelled challenges?

LasHeR ships per-SEQUENCE attribute labels (19 challenges; no per-frame timing).
For each sequence we measure how much zeroing the aux (thermal) modality
disturbs the MoE routing (routing divergence = mean_t 1-cos(gate_full, gate_miss)),
then group sequences by official attribute and compare WITH vs WITHOUT.

This tests, with official labels, whether the router's reliance on thermal is
challenge-dependent (e.g. lower disturbance in thermal-crossover sequences where
thermal is already unreliable). rung0 only. Inference-only.
"""
import os, sys, json, re
sys.path.insert(0, '/mnt/task_runtime'); os.chdir('/mnt/task_runtime')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from viz_experiment import build, frames, sync_state, SEQ_HOME, OUT
from lib.train.dataset.depth_utils import get_x_frame

CFG = 'flextrackv2_b224_56_abl_rung0'
LA = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher'
ATTR = f'{LA}/Attributes&Order/AttriSeqsTxt'
ORDER = "NO PO TO HO MB LI HI AIV LR DEF BC SA CM TC FL OV FM SV ARV".split()
DESC = {'PO':'partial-occ','TO':'total-occ','TC':'thermal-crossover','FM':'fast-motion',
        'SV':'scale-var','LI':'low-illum','BC':'bg-clutter','DEF':'deformation','MB':'motion-blur'}


def attrs(seq):
    p = f'{ATTR}/{seq}.txt'
    if not os.path.exists(p):
        return None
    v = [int(x) for x in re.split(r'[,\s]+', open(p).read().strip()) if x != '']
    return v[:19] if len(v) >= 19 else None


def frame_cond(seq, vis, inf, i, xtype, miss):
    img = np.array(get_x_frame(f'{SEQ_HOME}/{seq}/visible/{vis[i]}',
                               f'{SEQ_HOME}/{seq}/infrared/{inf[i]}', dtype=xtype)).copy()
    if miss:
        img[:, :, 3:] = 0
    return img


def seq_divergence(tkD, moeD, tkM, moeM, params, seq, nf):
    xtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
    vis, inf, gt = frames(seq)
    nf = min(nf, len(vis) - 1)
    tkD.initialize(frame_cond(seq, vis, inf, 0, xtype, False), {'init_bbox': gt[0].tolist()})
    tkM.initialize(frame_cond(seq, vis, inf, 0, xtype, False), {'init_bbox': gt[0].tolist()})
    divs = []
    for i in range(1, nf + 1):
        with torch.no_grad():
            sync_state(tkM, tkD)
            tkD.track(frame_cond(seq, vis, inf, i, xtype, False), [1.0, 1.0])
            tkM.track(frame_cond(seq, vis, inf, i, xtype, True), [1.0, 0.0])
        gf = moeD._cap['gates'].float().cpu().numpy().reshape(-1)
        gm = moeM._cap['gates'].float().cpu().numpy().reshape(-1)
        divs.append(1 - float(np.dot(gf, gm) / (np.linalg.norm(gf) * np.linalg.norm(gm) + 1e-9)))
    return float(np.mean(divs))


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    NF = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    allseq = sorted([d for d in os.listdir(f'{LA}/testingset') if os.path.isdir(f'{SEQ_HOME}/{d}')])
    seqs = allseq[:N]

    tkD, params, moeD = build(CFG)
    tkM, _, moeM = build(CFG)
    per = {}
    for s in seqs:
        a = attrs(s)
        if a is None:
            continue
        try:
            d = seq_divergence(tkD, moeD, tkM, moeM, params, s, NF)
            per[s] = dict(div=d, attr=a)
            print(f"  {s:26s} routing_div={d:.3f}", flush=True)
        except Exception as e:
            print(f"  {s} FAILED {e}", flush=True)

    # group by attribute: mean divergence with vs without
    rows = []
    for attr in ['TC', 'TO', 'PO', 'FM', 'SV', 'LI', 'BC', 'DEF']:
        idx = ORDER.index(attr)
        withv = [r['div'] for r in per.values() if r['attr'][idx] == 1]
        without = [r['div'] for r in per.values() if r['attr'][idx] == 0]
        if len(withv) >= 3 and len(without) >= 3:
            rows.append((attr, np.mean(withv), np.std(withv), len(withv),
                         np.mean(without), np.std(without), len(without)))
            print(f"=== {attr}({DESC.get(attr,attr)}): with={np.mean(withv):.3f}±{np.std(withv):.3f}(n{len(withv)})  "
                  f"without={np.mean(without):.3f}±{np.std(without):.3f}(n{len(without)}) ===", flush=True)

    # figure: grouped bars
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w/2, [r[1] for r in rows], w, yerr=[r[2] for r in rows], capsize=3,
           label='sequences WITH attribute', color='#e45756')
    ax.bar(x + w/2, [r[4] for r in rows], w, yerr=[r[5] for r in rows], capsize=3,
           label='sequences WITHOUT', color='#4c78a8')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\n{DESC.get(r[0],r[0])}\n(n={r[3]}/{r[6]})" for r in rows], fontsize=8)
    ax.set_ylabel('routing divergence (full vs aux-missing)')
    ax.set_title(f'Router sensitivity to missing thermal, grouped by official LasHeR attribute\n({len(per)} sequences, {NF} frames each) — rung0')
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f'{OUT}/viz_router_attributes.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"saved {out}", flush=True)
    json.dump({'per_seq': per, 'grouped': [{'attr': r[0], 'with_mean': r[1], 'with_std': r[2], 'with_n': r[3],
               'without_mean': r[4], 'without_std': r[5], 'without_n': r[6]} for r in rows]},
              open(f'{OUT}/viz_router_attributes.json', 'w'), default=float)
    print("done", flush=True)
