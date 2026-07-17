"""Experiment #2 (+#1 for free): mechanism visualization.

For the SAME video, run the tracker twice per frame -- full modality vs aux
zeroed (missing) -- and measure how close the FUSED internal representation
(MoEFusion output V, search-region tokens) stays. If the hallucination /
curriculum works, the missing-input representation should remain close to the
full-input one.

  #2 feature_sim   = cosine( V_full , V_missing )  per search token
  #1 hal_fidelity  = cosine( hallucinated_aux , real_aux ) per search token
                     (measured on the FULL pass, where real aux exists)

Compared across rung0 (curriculum ON, P_MAX=0.35) vs pmax_000 (curriculum OFF).
Prediction: rung0 keeps high similarity under missing; pmax_000 collapses,
visually explaining the pmax_000 result.

Inference-only, existing checkpoints. Search tokens = first 196 = 14x14.
"""
import os, sys, json
sys.path.insert(0, '/mnt/task_runtime')
os.chdir('/mnt/task_runtime')
import numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.train.dataset.depth_utils import get_x_frame
import lib.test.parameter.flextrackv2 as P
from lib.test.tracker.flextrackv2 import FlexTrackV2
from lib.models.flextrackv2.moe_fusion import MoEFusion

SEQ_HOME = '/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset'
GRID = 14  # sqrt(196)
OUT = '/mnt/tmp/claude-0/-mnt-task-runtime/d59af684-d511-4336-8b76-0e1cdbd03aee/scratchpad'


# Monkeypatch MoEFusion.forward once: it stashes the TRUE internal fused
# representation on the module (self._cap), computed with the REAL missing_mask
# the model passes (the model signals missing via missing_mask=[1,0], NOT via
# zeroed features, so an input-only hook can't tell). Behaviour is unchanged.
_ORIG_FWD = MoEFusion.forward
def _patched_forward(self, x, loss_coef=1e-2, missing_mask=None):
    out = _ORIG_FWD(self, x, loss_coef, missing_mask)
    if getattr(self, 'moe_type', None) == 'BMR_HMoE':
        xr = x[:, :, :self.input_size]; xa = x[:, :, self.input_size:]
        hal_aux = self.rgb_to_aux_hallucinater(xr)
        if missing_mask is not None:
            aux_missing = (missing_mask[:, 1:2] == 0.0).unsqueeze(-1)
        else:
            aux_missing = (xa.abs().mean(dim=(1, 2)) < 1e-4).unsqueeze(-1).unsqueeze(-1)
        fill_aux = torch.zeros_like(xa) if self.substitute_mode == 'zero' else hal_aux
        r_aux = torch.where(aux_missing, fill_aux, xa)
        x_recon = torch.cat((xr, r_aux), dim=-1)
        # MoE routing that this fused rep produces (which experts fire per token)
        gates = self.noisy_top_k_gating(x_recon, False)[0]   # [B, L, num_experts]
        self._cap = dict(x_recon=x_recon.detach()[0],
                         x_aux=xa.detach()[0], hal_aux=hal_aux.detach()[0],
                         gates=gates.detach()[0])
    return out
MoEFusion.forward = _patched_forward


def build(cfg):
    params = P.parameters(cfg, 40)
    tk = FlexTrackV2(params, 'LasHeR')
    moe = [m for _, m in tk.network.named_modules() if isinstance(m, MoEFusion)]
    return tk, params, moe[0]


def frames(seq):
    vis = sorted(os.listdir(f'{SEQ_HOME}/{seq}/visible'))
    inf = sorted(os.listdir(f'{SEQ_HOME}/{seq}/infrared'))
    gt = np.loadtxt(f'{SEQ_HOME}/{seq}/visible.txt', delimiter=',')
    n = min(len(vis), len(inf), len(gt))
    return vis[:n], inf[:n], gt[:n]


def make_frame(seq, vis, inf, i, xtype, zero_aux=False):
    img = np.array(get_x_frame(f'{SEQ_HOME}/{seq}/visible/{vis[i]}',
                               f'{SEQ_HOME}/{seq}/infrared/{inf[i]}', dtype=xtype))
    if zero_aux:
        img = img.copy(); img[:, :, 3:] = 0
    return img


def cos_tokens(a, b):
    # a,b: [L, C] -> [L] cosine per token
    return F.cosine_similarity(a.float(), b.float(), dim=-1).cpu().numpy()


def sync_state(dst, src):
    """Make tracker `dst` an exact clone of `src`'s current state, so the only
    difference between the two passes is this frame's zeroed aux -- NOT a
    drifted search crop or a different template memory."""
    import copy
    dst.state = list(src.state)
    for attr in ['template_list', 'template_anno_list', 'missing_list',
                 'memory_template_list', 'memory_template_anno_list', 'memory_missing_list']:
        if hasattr(src, attr):
            setattr(dst, attr, copy.copy(getattr(src, attr)))


def run_config(cfg, seq, n_frames):
    tkF, params, moeF = build(cfg)
    tkM, _, moeM = build(cfg)
    xtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
    vis, inf, gt = frames(seq)
    n_frames = min(n_frames, len(vis) - 1)

    f0 = make_frame(seq, vis, inf, 0, xtype)
    tkF.initialize(f0, {'init_bbox': gt[0].tolist()})
    tkM.initialize(f0, {'init_bbox': gt[0].tolist()})

    sim_maps, hal_maps, sim_scal, hal_scal, gate_sims = [], [], [], [], []
    gate_full_acc, gate_miss_acc = [], []
    for i in range(1, n_frames + 1):
        with torch.no_grad():
            # sync BEFORE tracking: crop uses the PREVIOUS-frame box, so both
            # must share it now (tkF.track will then overwrite tkF.state).
            sync_state(tkM, tkF)
            tkF.track(make_frame(seq, vis, inf, i, xtype, zero_aux=False), [1.0, 1.0])
            tkM.track(make_frame(seq, vis, inf, i, xtype, zero_aux=True), [1.0, 0.0])
        capF, capM = moeF._cap, moeM._cap
        # #2 fused-representation stability: x_reconstructed full vs missing.
        Xf = capF['x_recon'][:196]; Xm = capM['x_recon'][:196]
        sim = cos_tokens(Xf, Xm)                                   # [196]
        # #1 hallucination fidelity: hallucinated aux vs real aux (full pass)
        hal = cos_tokens(capF['hal_aux'][:196], capF['x_aux'][:196])
        # #3 MoE expert routing (per-frame 8-way distribution) full vs missing
        gf = capF['gates'].float().cpu().numpy().reshape(-1)      # [num_experts]
        gm = capM['gates'].float().cpu().numpy().reshape(-1)
        gate_sim = float(np.dot(gf, gm) / (np.linalg.norm(gf) * np.linalg.norm(gm) + 1e-9))
        sim_maps.append(sim); hal_maps.append(hal)
        sim_scal.append(float(sim.mean())); hal_scal.append(float(hal.mean()))
        gate_sims.append(gate_sim); gate_full_acc.append(gf); gate_miss_acc.append(gm)
    return dict(cfg=cfg, seq=seq, n=len(sim_scal),
                sim_map=np.stack(sim_maps).mean(0).reshape(GRID, GRID),
                hal_map=np.stack(hal_maps).mean(0).reshape(GRID, GRID),
                sim_overall=float(np.mean(sim_scal)), hal_overall=float(np.mean(hal_scal)),
                gate_sim=float(np.mean(gate_sims)),
                gate_full=np.stack(gate_full_acc).mean(0), gate_miss=np.stack(gate_miss_acc).mean(0))


LABELS = {'flextrackv2_b224_56_abl_rung0': 'rung0 (hallucinate-fill)',
          'flextrackv2_b224_56_abl_no_hallucinate': 'no_hallucinate (zero-fill)'}


def aggregate(per_seq):
    """Combine per-sequence dicts into means + stds."""
    a = {}
    a['n_seq'] = len(per_seq)
    a['sim_map'] = np.stack([r['sim_map'] for r in per_seq]).mean(0)
    a['hal_map'] = np.stack([r['hal_map'] for r in per_seq]).mean(0)
    a['gate_full'] = np.stack([r['gate_full'] for r in per_seq]).mean(0)
    a['gate_miss'] = np.stack([r['gate_miss'] for r in per_seq]).mean(0)
    for k in ['sim_overall', 'hal_overall', 'gate_sim']:
        v = np.array([r[k] for r in per_seq])
        a[k + '_mean'] = float(v.mean()); a[k + '_std'] = float(v.std())
    return a


if __name__ == '__main__':
    N_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    NF = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    CONFIGS = ['flextrackv2_b224_56_abl_rung0', 'flextrackv2_b224_56_abl_no_hallucinate']
    ALL_SEQS = sorted([d for d in os.listdir(SEQ_HOME) if os.path.isdir(f'{SEQ_HOME}/{d}')])
    SEQS = ALL_SEQS[:N_SEQ]
    print(f"sequences ({len(SEQS)}): {SEQS}", flush=True)

    agg = {}
    for cfg in CONFIGS:
        per_seq = []
        for s in SEQS:
            try:
                r = run_config(cfg, s, NF); per_seq.append(r)
                print(f"  [{cfg.split('abl_')[-1]:14s}] {s:24s} sim={r['sim_overall']:.3f} hal={r['hal_overall']:.3f} gate={r['gate_sim']:.3f}", flush=True)
            except Exception as e:
                print(f"  [{cfg}] {s} FAILED: {e}", flush=True)
        agg[cfg] = aggregate(per_seq)
        a = agg[cfg]
        print(f"=== {cfg}: sim={a['sim_overall_mean']:.3f}±{a['sim_overall_std']:.3f}  "
              f"hal={a['hal_overall_mean']:.3f}±{a['hal_overall_std']:.3f}  "
              f"gate={a['gate_sim_mean']:.3f}±{a['gate_sim_std']:.3f}  (n={a['n_seq']} seqs)", flush=True)

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, cfg in enumerate(CONFIGS):
        a = agg[cfg]
        im0 = axes[row, 0].imshow(a['sim_map'], cmap='viridis', vmin=0.5, vmax=1)
        axes[row, 0].set_title(f"{LABELS[cfg]}\nfused-rep stability (full vs missing)\nmean={a['sim_overall_mean']:.3f}")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046); axes[row, 0].axis('off')
        # expert usage full vs missing
        ax = axes[row, 1]
        ne = len(a['gate_full']); xx = np.arange(ne)
        ax.bar(xx - 0.2, a['gate_full'], 0.4, label='full', color='#4c78a8')
        ax.bar(xx + 0.2, a['gate_miss'], 0.4, label='aux missing', color='#e45756')
        ax.set_title(f"{LABELS[cfg]}\nMoE expert usage (routing sim={a['gate_sim_mean']:.3f})")
        ax.set_xlabel('expert'); ax.set_ylabel('mean gate'); ax.legend(fontsize=8)
    # summary bars (feature sim + routing sim) with std
    ax = axes[0, 2]
    x = np.arange(2); w = 0.35
    for j, cfg in enumerate(CONFIGS):
        a = agg[cfg]
        ax.bar(x + (j - 0.5) * w, [a['sim_overall_mean'], a['gate_sim_mean']], w,
               yerr=[a['sim_overall_std'], a['gate_sim_std']], capsize=4,
               label=LABELS[cfg], color=['#4c78a8', '#e45756'][j])
    ax.set_xticks(x); ax.set_xticklabels(['fused-rep\nstability', 'routing\nconsistency'])
    ax.set_ylabel('cosine (full vs missing)'); ax.set_ylim(0, 1.05)
    ax.set_title(f'aggregate over {agg[CONFIGS[0]]["n_seq"]} LasHeR sequences'); ax.legend(fontsize=8)
    # text summary
    axes[1, 2].axis('off')
    txt = f"aggregate over {agg[CONFIGS[0]]['n_seq']} sequences ({NF} frames each)\n\n"
    for cfg in CONFIGS:
        a = agg[cfg]
        txt += (f"{LABELS[cfg]}:\n"
                f"  fused-rep stability = {a['sim_overall_mean']:.3f} ± {a['sim_overall_std']:.3f}\n"
                f"  hallucination fidel = {a['hal_overall_mean']:.3f} ± {a['hal_overall_std']:.3f}\n"
                f"  routing consistency = {a['gate_sim_mean']:.3f} ± {a['gate_sim_std']:.3f}\n\n")
    axes[1, 2].text(0.0, 0.5, txt, fontsize=10, va='center', family='monospace')
    plt.tight_layout()
    out_png = f'{OUT}/viz_mechanism_aggregate.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f"\nsaved {out_png}", flush=True)
    json.dump({cfg: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in agg[cfg].items()}
               for cfg in CONFIGS}, open(f'{OUT}/viz_mechanism_aggregate.json', 'w'))
    print("done", flush=True)
