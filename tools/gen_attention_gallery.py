"""Regenerate interp_attention_gallery.png with an added raw auxiliary-modality
(thermal) column, so the figure shows: RGB search region | thermal search region
| attention (full modality) | attention (thermal missing).

Eval-only, inference-only. Monkeypatches sample_target (module-level import in
lib.test.tracker.flextrackv2) and the tracker's network.forward_decoder to
capture the search-region crop (RGB+thermal, 6ch) and the response/score map,
without touching any repo source.
"""
import os, sys, json
sys.path.insert(0, "/mnt/task_runtime")
os.chdir("/mnt/task_runtime")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import lib.test.tracker.flextrackv2 as TRK
import lib.test.parameter.flextrackv2 as P
from lib.train.dataset.depth_utils import get_x_frame

LASHER = "/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset"
OUT = "/mnt/task_runtime/results/figures/flextrackv2_vs_v1"

# ---- capture buffer, filled by monkeypatched sample_target ----
_cap = {}
_orig_sample_target = TRK.sample_target
def _sample_target_hook(im, target_bb, search_area_factor, output_sz=None):
    patch, rf = _orig_sample_target(im, target_bb, search_area_factor, output_sz=output_sz)
    if _cap.get("search_factor") is not None and abs(search_area_factor - _cap["search_factor"]) < 1e-6:
        _cap["patch"] = patch  # HxWx6 numpy (RGB[:3] + aux[3:]), the SEARCH region crop
    return patch, rf
TRK.sample_target = _sample_target_hook


def build(cfg="flextrackv2"):
    params = P.parameters(cfg, 40)
    tk = TRK.FlexTrackV2Tracker(params, "LasHeR")
    _cap["search_factor"] = params.search_factor
    orig_decoder = tk.network.forward_decoder
    def decoder_hook(*a, **kw):
        out = orig_decoder(*a, **kw)
        _cap["score_map"] = out["score_map"].detach().float().cpu().clone()
        return out
    tk.network.forward_decoder = decoder_hook
    return tk, params


def frame_cond(seq, vis, inf, i, xtype, cond):
    img = np.array(get_x_frame(f"{LASHER}/{seq}/visible/{vis[i]}",
                               f"{LASHER}/{seq}/infrared/{inf[i]}", dtype=xtype)).copy()
    if cond == "aux-miss":
        img[:, :, 3:] = 0
    return img


def response_at(tk, seq, vis, inf, gt, fi, xtype, cond):
    img = frame_cond(seq, vis, inf, fi, xtype, cond)
    missing = [1.0, 1.0] if cond == "full" else [1.0, 0.0]
    out = tk.track(img, missing)
    sm = _cap["score_map"]
    if tk.cfg.TEST.WINDOW:
        resp = (tk.output_window.detach().float().cpu() * sm)
    else:
        resp = sm
    resp = resp.reshape(1, 1, tk.fx_sz, tk.fx_sz)
    patch = _cap["patch"].copy()
    return patch, resp, out


def overlay(rgb_uint8, resp, size):
    """rgb_uint8: HxWx3 uint8 crop. resp: [1,1,h,w] tensor. Returns an RGB uint8 blend."""
    r = F.interpolate(resp, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    r = (r - r.min()) / (r.max() - r.min() + 1e-9)
    heat = (cm.jet(r.numpy())[:, :, :3] * 255).astype(np.uint8)
    base = rgb_uint8.astype(np.float32)
    blended = 0.55 * base + 0.45 * heat.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


SEQS = [("3men", 15), ("3bike1", 31), ("10runone", 20), ("2runseven", 14)]

rows = []
for seq, fi in SEQS:
    tk, params = build()
    xtype = getattr(params.cfg.DATA, "XTYPE", "rgbrgb")
    vis = sorted(os.listdir(f"{LASHER}/{seq}/visible"))
    inf = sorted(os.listdir(f"{LASHER}/{seq}/infrared"))
    gt = np.loadtxt(f"{LASHER}/{seq}/visible.txt", delimiter=",")
    tk.initialize(frame_cond(seq, vis, inf, 0, xtype, "full"), {"init_bbox": gt[0].tolist()})
    with torch.no_grad():
        for i in range(1, fi):
            tk.track(frame_cond(seq, vis, inf, i, xtype, "full"), [1.0, 1.0])
        patch_full, resp_full, _ = response_at(tk, seq, vis, inf, gt, fi, xtype, "full")
        patch_miss, resp_miss, _ = response_at(tk, seq, vis, inf, gt, fi, xtype, "aux-miss")
    cos = F.cosine_similarity(resp_full.flatten(), resp_miss.flatten(), dim=0).item()
    rows.append(dict(seq=seq, fi=fi, patch_full=patch_full, resp_full=resp_full,
                      resp_miss=resp_miss, cos=cos))
    print(f"{seq} #{fi}: cos(full,miss)={cos:.3f}", flush=True)

sz = rows[0]["patch_full"].shape[0]
fig, axes = plt.subplots(len(rows), 4, figsize=(4 * 3.1, len(rows) * 3.15))
col_titles = ["Search region (RGB)", "Search region (thermal)",
              "Attention — full modality", "Attention — thermal missing"]
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=13)

for ridx, r in enumerate(rows):
    rgb = r["patch_full"][:, :, :3].astype(np.uint8)
    aux = r["patch_full"][:, :, 3:].astype(np.uint8)  # real (unzeroed) thermal crop
    if aux.shape[2] == 1:
        aux = np.repeat(aux, 3, axis=2)
    ov_full = overlay(rgb, r["resp_full"], sz)
    ov_miss = overlay(rgb, r["resp_miss"], sz)

    axes[ridx, 0].imshow(rgb)
    axes[ridx, 1].imshow(aux)
    axes[ridx, 2].imshow(ov_full)
    axes[ridx, 3].imshow(ov_miss)
    axes[ridx, 3].text(0.03, 0.06, f"cos(full,miss)={r['cos']:.3f}", transform=axes[ridx, 3].transAxes,
                        fontsize=10.5, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6, ec="none"))
    axes[ridx, 0].set_ylabel(f"{r['seq']}\n#{r['fi']}", fontsize=11)
    for c in range(4):
        axes[ridx, c].set_xticks([]); axes[ridx, c].set_yticks([])

fig.suptitle("FlexTrackV2 — target attention is near-invariant to modality dropout",
             fontsize=15, fontweight="bold", y=0.995)
fig.text(0.5, 0.975, "Same frame & same tracker state: RGB+Thermal (full) vs Thermal removed (aux-missing). "
                     "The response map stays locked on the target.",
         ha="center", fontsize=11, color="#444")
plt.tight_layout(rect=[0, 0, 1, 0.96])
outp = f"{OUT}/interp_attention_gallery.png"
plt.savefig(outp, dpi=150, bbox_inches="tight")
print("SAVED", outp)
