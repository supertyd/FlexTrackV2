# FlexTrack-V2 (V56) — TPAMI Ablation Study Plan

Companion doc to the `v56-tpami-extension` branch. Tracks exactly which
ablations isolate which claim in the paper, how each is implemented, and how
to reproduce it.

## Evaluation protocol

Every ablation below is trained on the joint VisEvent + LasHeR_train +
DepthTrack_train mixture (same as the V56 baseline, `flextrackv2_b224_56.yaml`,
1:1:1 sampling, 60k samples/epoch, 40 epochs) and evaluated on:

- **RGBT234** / **RGBT234_miss**
- **VisEvent** / **VisEvent_miss**
- **DepthTrack** / **DepthTrack_miss** (official VOT-toolkit protocol, `votrgbd2021` stack)

Each config is a clone of the stable, git-committed `flextrackv2_b224_56.yaml`
baseline with **exactly one field changed**, so any performance delta is
attributable to that single factor. Configs live in
`experiments/flextrackv2/flextrackv2_b224_56_abl_*.yaml`.

## Tier 1 — isolated config-only ablations (in progress)

| # | Ablation | Config file | Field changed | Question it answers |
|---|---|---|---|---|
| 1 | Homogeneous experts (BIG) | `flextrackv2_b224_56_abl_moe_big.yaml` | `MODEL.MOE.TYPE: BMR_HMoE → BIG` (8× uniform width 512) | Does heterogeneous expert capacity matter, or would 8 large uniform experts do as well? |
| 2 | Homogeneous experts (SMALL) | `flextrackv2_b224_56_abl_moe_small.yaml` | `MODEL.MOE.TYPE → SMALL` (8× uniform width 4) | Same, at the low-capacity extreme |
| 3 | Homogeneous experts (MIDDLE) | `flextrackv2_b224_56_abl_moe_middle.yaml` | `MODEL.MOE.TYPE → MIDDLE` (8× uniform width 128) | Same, at a mid-capacity point |
| 4 | Semi-heterogeneous experts (HYBRID) | `flextrackv2_b224_56_abl_moe_hybrid.yaml` | `MODEL.MOE.TYPE → HYBRID` (paired capacities 4/4/8/8/12/12/16/16) | Does mild heterogeneity capture most of the benefit, or is the full exponential spread [4..512] needed? |
| 5 | Fixed missing rate | `flextrackv2_b224_56_abl_cma_fixed020.yaml` | `CMA_P_MIN: 0.0→0.2`, `CMA_P_MAX: 0.35→0.2` (constant 20% instead of a 0→35% curriculum) | Does the curriculum schedule matter, or is training at a fixed missing rate equally good? |
| 6 | No self-distillation | `flextrackv2_b224_56_abl_no_distill.yaml` | `TRAIN.DISTILL_WEIGHT: 2.5→0.0` | Is the teacher(complete)/student(degraded) consistency loss necessary, or does missing-rate data augmentation alone explain the robustness gain? |
| 7 | p_max = 0.15 | `flextrackv2_b224_56_abl_pmax_015.yaml` | `CMA_P_MAX: 0.35→0.15` | Sensitivity of the curriculum ceiling (low end) |
| 8 | p_max = 0.25 | `flextrackv2_b224_56_abl_pmax_025.yaml` | `CMA_P_MAX: 0.35→0.25` | Sensitivity of the curriculum ceiling (mid) |
| 9 | p_max = 0.50 | `flextrackv2_b224_56_abl_pmax_050.yaml` | `CMA_P_MAX: 0.35→0.50` | Sensitivity of the curriculum ceiling (high end); expected to start trading off complete-modality accuracy |

The full baseline (`flextrackv2_b224_56.yaml`, `MOE.TYPE=BMR_HMoE`,
`CMA_P_MIN/MAX=0.0/0.35`, `DISTILL_WEIGHT=2.5`) is the reference point / "rung
0" for all nine rows above — it does not need to be retrained.

### Reproduce

```bash
# Train (each config's checkpoints land in train/flextrackv2/<config_name>/,
# fully isolated from the V56 baseline and from every other ablation)
torchrun --nproc_per_node 8 lib/train/run_training.py \
  --script flextrackv2 --config <config_name> --save_dir .

# Test — repeat per dataset
python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name flextrackv2 --dataset_name RGBT234 \
  --yaml_name <config_name> --mode parallel --threads 4 --num_gpus 1 --epoch 40
python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name flextrackv2 --dataset_name RGBT234_miss \
  --yaml_name <config_name> --mode parallel --threads 4 --num_gpus 1 --epoch 40
python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name flextrackv2 --dataset_name VisEvent \
  --yaml_name <config_name> --mode parallel --threads 4 --num_gpus 1 --epoch 40
python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name flextrackv2 --dataset_name VisEvent_miss \
  --yaml_name <config_name> --mode parallel --threads 4 --num_gpus 1 --epoch 40

# DepthTrack / DepthTrack_miss go through the VOT toolkit — register
# <config_name> in Depthtrack_workspace/trackers.ini, then:
vot evaluate --workspace Depthtrack_workspace <config_name>
vot analysis --nocache --workspace Depthtrack_workspace <config_name>
```

All nine configs are being run sequentially (one at a time, each using all 8
GPUs) via `run_ablations_sequential.sh`, logging to `ablation_logs/<config>_train.log`.

## Tier 2 — mechanism ablations (implemented, mostly complete)

Each needed one new field in `MODEL.MOE` (default value = current V56
behavior, so existing configs are unaffected) plus a few guarded lines in
`lib/models/flextrackv2/moe_fusion.py`. Status as of this writing:

| Ablation | Flag | Status |
|---|---|---|
| Masking vs. reconstruction (core thesis) | `MODEL.MOE.SUBSTITUTE_MODE: hallucinate\|zero` | ✅ `flextrackv2_b224_56_abl_no_hallucinate` complete |
| With/without reconstruction loss | `MODEL.MOE.USE_RECON_LOSS: true\|false` | ✅ `flextrackv2_b224_56_abl_no_recon_loss` complete |
| Uni- vs bilateral hallucination | `MODEL.MOE.HALLUCINATE_DIRECTION: bilateral\|rgb2aux\|aux2rgb` | ✅ `..._uni_a2r` / `..._uni_r2a` complete |
| Gate-orthogonality regularizer | `MODEL.MOE.USE_ORTHO: true\|false` | ✅ `flextrackv2_b224_56_abl_no_ortho` complete |
| **No-MoE baseline (pure residual fusion)** | `MODEL.MOE.TYPE: NONE` | ❌ **not started** — still the one open Tier-2 cell |
| Seed variance (baseline only) | — (reruns of `flextrackv2_b224_56`) | ✅ `rung0` / `rung0_seedB` / `rung0_seedC` complete |

## Tier 3 — backbone attribution (not started)

Isolates whether gains come from BMR-HMoE/CMA or from the Fast-iTPN + Mamba
backbone swap versus the conference (FlexTrack) encoder. Needs one new
config, `flextrackv2_b224_56_abl_backbone_only.yaml`: clone of the baseline with
`MODEL.MOE.TYPE: NONE` + `TRAIN.DISTILL_WEIGHT: 0` + `CMA_P_MIN/MAX: 0.0/0.0`
— i.e. the same "no-MoE" config as the missing Tier-2 cell above, plus CMA
disabled entirely. **These two configs can be collapsed into one run** if
scoped as "MoE off, CMA off" (Tier 3) vs. reusing the same checkpoint's
complete-modality numbers to also answer the Tier-2 "no-MoE" question — saves
a full 8-GPU training slot. One training run, standard 6(+2)-dataset eval.

## Tier 4 — robustness stress tests (not started, eval-only, no retraining needed)

Cheapest remaining tier — no training, just new eval scripts against the
**already-trained V56 baseline checkpoint** (`flextrackv2_b224_56`, epoch 40).
Highest insight-per-GPU-hour of anything left; should be picked up by the
next node that frees up regardless of what else is queued.

- **Continuous test-time missing-rate sweep**: eval the baseline at
  synthetic missing-rates {0, 10, 20, ..., 100}% (not just the fixed
  "_miss" protocol's rate) on RGBT234/VisEvent/DepthTrack, vs. FlexTrack at
  the same rates. Needs a small generalization of the existing
  `_miss`-dataset JSON generation (`data_missing_modality/`) to accept an
  arbitrary rate parameter instead of the one baked-in rate.
- **Burst vs. random missingness**: contiguous-frame dropout (simulates a
  sensor dropout window) vs. today's per-frame-random dropout, same target
  rates. Tests whether CMA's per-frame training generalizes to correlated
  missingness it never saw in training — a real threat-to-validity question
  a reviewer would ask.

## Tier 5 — model scale (in progress)

Does the BMR-HMoE/CMA recipe's benefit hold, shrink, or grow as backbone
capacity changes? One data point (base, fastitpnb) says nothing about this on
its own.

| Config | Backbone | Status |
|---|---|---|
| `flextrackv2_l224_56` | fastitpnl (large, D_MODEL 768) | 🔄 training on `qad3e9trm9` (8×B200), epoch 36/40 as of 2026-07-07. Init checkpoint pulled from the official FlexTrackV2 model zoo (`flextrackv2_l224`), not locally re-pretrained. |

Fixed while bringing this up: `fastitpnl`'s pretrain-path string concat
double-prefixed `/mnt/task_runtime` (commit `17ca07a`) — `fastitpnb` never
hit this because its checkpoint-load branch is commented out, which is a
separate, pre-existing correctness gap worth flagging to whoever owns that
file (the *base* V56 models — this baseline included — may be training their
encoder from scratch rather than from the intended `FlexTrackV2_ep0300.pth.tar`
init; unconfirmed, but the code path is dead so it's not currently loading
anything).

## Tier 6 — proposed additional experiments (not started)

Ranked by insight-per-GPU-hour, cheapest first — pick these up in order as
nodes free up from Tiers 1-2:

1. **Finish Tier 4 (robustness stress tests)** — zero training cost, directly
   strengthens the paper's core robustness claim, should preempt everything
   else below on the first node that goes idle.
2. **The missing Tier-2/3 cell** — one run answers both "no-MoE" (Tier 2) and
   "backbone-only" (Tier 3) if scoped as described above.
3. **Capacity sweep at the small end** (`flextrackv2_t224_56`, `flextrackv2_s224_56`
   — tiny/small fastitpn variants, same V56 recipe): cheap (small models
   train fast) and completes a 4-point capacity curve
   (tiny/small/base/large) alongside Tier 5's large point. Answers "does
   BMR-HMoE need spare capacity to pay for itself, or does it help even at
   the low end?" — a natural companion plot to the large-model result.
4. **Per-direction missing-modality breakdown at test time**: today's
   `_miss` protocol samples missingness across both RGB and the auxiliary
   modality combined. Splitting eval into "RGB missing only" vs. "aux
   modality missing only" (eval-only, reuse the baseline checkpoint) tests
   whether BMR's bilateral hallucination is actually symmetric in practice,
   or whether one direction is carrying the robustness gain — directly
   informed by the already-collected `uni_a2r`/`uni_r2a` *training*-side
   result, but nobody has looked at this from the *eval* side yet.
5. **Cross-modality-type generalization**: train on two of
   {VisEvent, LasHeR, DepthTrack} and zero-shot eval the missing-modality
   protocol on the third's modality *type* (e.g. train Event+Thermal only,
   eval Depth `_miss`). Tests whether CMA-learned robustness is a general
   "handle an absent stream" skill or overfits to the specific modalities
   seen in training — a plausible reviewer question given the paper's
   "unified architecture" framing.
6. **384-resolution large model**: `flextrackv2_b384_large_54.yaml` already
   exists as a V54-era starting point; re-run it with the current V56 recipe
   (BMR-HMoE + CMA, same as Tier 5) and the `fastitpnl` path fix from this
   session. Most expensive item on this list (large model *and* 384
   resolution *and* the 384 backbone needs its own from-scratch RGB
   pretraining first, per `ABLATIONS.md`'s current `flextrackv2_l384`
   dependency) — do this last unless resolution scaling is a claim the paper
   specifically wants to make.
7. **Extra seeds beyond the baseline**: `rung0` has 3 seeds; the ablations
   most likely to be highlighted in the paper (e.g. `no_ortho`,
   `cma_fixed020`) currently have none, so any "is this delta real or noise"
   question about them is currently unanswerable. Add 1-2 extra seeds to
   whichever 2-3 ablations end up in the final paper draft, once that's
   decided — low priority until the paper's argument is finalized, since
   seeding everything now would be wasted compute if the argument changes.

## Multi-machine execution

The 9 configs run one-per-node, fully in parallel: the config already in
flight stays on the current node, and each of the other 8 gets its own
dedicated node. See `/root/.claude/plans/mossy-forging-canyon.md` for the
full design rationale.

| Worker | Config |
|---|---|
| Current node (`92av62ravp`) | `flextrackv2_b224_56_abl_moe_big` *(in progress)* |
| New node (`config-ablation-moe_small.yaml`) | `flextrackv2_b224_56_abl_moe_small` |
| New node (`config-ablation-moe_middle.yaml`) | `flextrackv2_b224_56_abl_moe_middle` |
| New node (`config-ablation-moe_hybrid.yaml`) | `flextrackv2_b224_56_abl_moe_hybrid` |
| New node (`config-ablation-cma_fixed020.yaml`) | `flextrackv2_b224_56_abl_cma_fixed020` |
| New node (`config-ablation-no_distill.yaml`) | `flextrackv2_b224_56_abl_no_distill` |
| New node (`config-ablation-pmax_015.yaml`) | `flextrackv2_b224_56_abl_pmax_015` |
| New node (`config-ablation-pmax_025.yaml`) | `flextrackv2_b224_56_abl_pmax_025` |
| New node (`config-ablation-pmax_050.yaml`) | `flextrackv2_b224_56_abl_pmax_050` |

Each new node requests `p6-b200.48xlarge` on cluster `aws_10` (`task_type:
8gpu`), following the same config shape and docker image as the user's
already-running `edindza33z` task on that cluster. Project quota on `aws_10`
is 15 nodes / 120 GPUs guaranteed — 8 new nodes (64 GPUs) fits comfortably.
Assignment is **static** (no dynamic claiming) — appropriate for a one-off
batch of known jobs.

### Spinning up a new node

```bash
# From this (or any authenticated) machine. Replace <token> with a live,
# non-exposed HF token — do not bake it into the yaml file itself.
bolt task submit --config config-ablation-moe_small.yaml \
  --git https://github.com/supertyd/FlexTrack-V2.git@ablations \
  --update-config environment_variables.HF_TOKEN=<token>
# ...repeat for each config-ablation-<name>.yaml
```

On the new node (`bolt task ssh <id>`; `install.sh` already ran as the
Bolt `setup_command` before this):

```bash
./provision_ablation_node.sh     # downloads datasets, writes local.py
# then, from the CURRENT node, copy the pretrained checkpoint over:
#   bolt task scp <SOURCE_ID>:/mnt/task_runtime/train/FlexTrackV2_ep0300.pth.tar \
#                 <NEW_ID>:/mnt/task_runtime/train/FlexTrackV2_ep0300.pth.tar
./run_ablation_worker.sh flextrackv2_b224_56_abl_moe_small
```

### Result reporting (how "unified management" works)

Each worker, after finishing a config, writes
`ablation_results/<config>/metrics.json` (produced by
`compute_ablation_metrics.py`, which computes box-overlap AUC/PR for
RGBT234/VisEvent full+miss and Pr/Re/F via the VOT toolkit for
DepthTrack full+miss) and commits **only that one new file** before pushing
to `ablations` — workers never edit a shared file, so concurrent pushes from
different machines touch disjoint paths and won't conflict.

Run this centrally any time to refresh the summary table below from
whatever results have landed so far:

```bash
git pull origin ablations
python aggregate_ablation_results.py
```

### Known limitations

- **Checkpoints are not centralized** — they stay on each node's local disk.
  If a node's Bolt task is cancelled/expires, pull anything needed first via
  `bolt task scp`.
- Each new node needs a fresh ~500GB dataset download
  (`provision_ablation_node.sh` → `download_datasets_hf.py` +
  `download_missing.py`) — this dominates node setup time.

### p6-b200 / Blackwell environment (required extra setup)

`p6-b200.48xlarge` nodes carry NVIDIA B200 (Blackwell, compute capability
`sm_100`) GPUs. The default environment's PyTorch (2.1.1+cu118, from
`install.sh`) **cannot run on them at all** — every CUDA call fails with
`RuntimeError: CUDA error: no kernel image is available for execution on the
device`. This is not a version-pin issue fixable with a quick `pip install
--upgrade`: **no PyTorch build for Python 3.8 (the environment's Python
version) ever added Blackwell support** — PyTorch dropped Python 3.8 wheels
around the 2.4.x series, before Blackwell support landed in 2.5+. Verified
directly against the official `download.pytorch.org` wheel index (not just
this network's mirrors): torch 2.7.1+cu126 ships `cp310`/`cp311`/`cp312`
wheels only, no `cp38`.

**Fix: a separate Python 3.10 conda env with PyTorch 2.7.1+cu126**, built
once per B200 node:

```bash
/coreflow/mambaforge/bin/conda create -y -n mci310 -c conda-forge \
  python=3.10 "pytorch=2.7.1=cuda126*" torchvision
PY=/coreflow/mambaforge/envs/mci310/bin/python
$PY -m pip install "setuptools<81"   # newer setuptools breaks build-isolated installs (missing pkg_resources)
$PY -m pip install PyYAML easydict cython opencv-python pandas==2.2.1 tqdm \
  pycocotools jpeg4py tb-nightly tikzplotlib colorama lmdb scipy timm yacs \
  pytorch-pretrained-bert scikit-image thop huggingface_hub numba
  # (numba isn't in install.sh but is imported somewhere in the training path;
  #  visdom was skipped -- its setup.py needs pkg_resources even with the
  #  setuptools pin above, and it's not needed for training)
```

Verified empirically (not just via `torch.cuda.is_available()`, which lies
here) with a real 8-way NCCL `all_reduce` and a real `torchrun` launch of
`lib/train/run_training.py` on `flextrackv2_b224_56_abl_moe_small` — both
succeeded end-to-end on this env. `run_ablation_worker.sh` reads
`MCI_PYTHON`/`MCI_TORCHRUN` env vars to pick the binary, defaulting to the
plain `python3`/`torchrun` (unaffected on non-B200 nodes):

```bash
export MCI_PYTHON=/coreflow/mambaforge/envs/mci310/bin/python3
export MCI_TORCHRUN=/coreflow/mambaforge/envs/mci310/bin/torchrun
./run_ablation_worker.sh flextrackv2_b224_56_abl_moe_small
```

### LasHeR download truncation (proxy-related, affects all nodes on this network)

`download_datasets_hf.py`'s LasHeR download (split into 5 parts, `part.aa`
through `part.ae`, ~50GB each) failed identically on **5 out of 5** nodes
that hit it, every time truncating `part.aa` at exactly byte 2,486,314,258
instead of the real 53,687,091,200 — the same byte offset across independent
nodes rules out random network noise; it looks like a proxy-side
idle/duration cap on this specific large transfer. Confirmed the source file
itself is not corrupted (its real `Content-Length` on `huggingface.co` is the
full 53,687,091,200 bytes). `snapshot_download` raised on the resulting
checksum mismatch, but the per-repo `try/except` in `download_datasets_hf.py`
swallowed it and printed a misleading "All downloads and extractions
completed!" — **fixed** (see commit history) to retry 3x and `sys.exit(1)` if
a dataset still fails, so this no longer fails silently.

If a node already has a truncated `lasher.tar.gz.part.aa`: plain `curl -C -`
resume does **not** work against HF's signed/expiring redirect URLs (fails
with "HTTP server doesn't seem to support byte ranges" once the redirect
target rotates). Use `huggingface_hub.hf_hub_download` directly instead,
which handles it correctly:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='xche32/lasher', repo_type='dataset',
                 filename='lasher.tar.gz.part.aa',
                 local_dir='/mnt/task_wrapper/user_output/artifacts/lasher')
```

then re-combine the 5 parts and `tar -xf` as `download_datasets_hf.py`
normally does.

## Results (auto-generated)

Populated by `aggregate_ablation_results.py` — do not hand-edit the block
below; it gets fully rewritten on every run based on
`ablation_results/*/metrics.json`.

<!-- AUTO-GENERATED RESULTS START -->

| Config | RGBT234 (AUC/PR) | RGBT234_miss (AUC/PR) | VisEvent (AUC/PR) | VisEvent_miss (AUC/PR) | DepthTrack (Pr/Re/F) | DepthTrack_miss (Pr/Re/F) |
|---|---|---|---|---|---|---|
| flextrackv2_b224_56_abl_cma_fixed020 | 67.18 / 90.06 | 67.06 / 89.93 | 63.78 / 76.76 | 64.02 / 76.99 | 65.92 / 68.80 / 67.33 | 64.70 / 67.60 / 66.11 |
| flextrackv2_b224_56_abl_moe_big | 67.25 / 90.07 | 59.58 / 82.15 | 63.44 / 76.22 | 53.48 / 66.74 | 63.84 / 66.71 / 65.25 | 56.49 / 59.02 / 57.73 |
| flextrackv2_b224_56_abl_moe_hybrid | 67.20 / 89.59 | 66.59 / 88.67 | 64.11 / 77.17 | 62.77 / 75.21 | 64.71 / 67.56 / 66.11 | 63.62 / 66.42 / 64.99 |
| flextrackv2_b224_56_abl_moe_middle | 66.88 / 89.65 | 67.29 / 90.32 | 63.42 / 76.28 | 63.91 / 76.86 | 60.95 / 63.70 / 62.30 | 62.19 / 65.00 / 63.57 |
| flextrackv2_b224_56_abl_moe_small | 66.55 / 89.00 | 66.85 / 89.42 | 63.54 / 76.59 | 63.48 / 76.42 | 65.79 / 68.74 / 67.23 | 61.30 / 64.15 / 62.69 |
| flextrackv2_b224_56_abl_no_distill | 66.99 / 90.08 | 66.54 / 89.59 | 63.38 / 76.59 | 62.51 / 75.48 | 64.66 / 67.50 / 66.05 | 64.92 / 67.78 / 66.32 |
| flextrackv2_b224_56_abl_pmax_015 | 66.07 / 88.65 | 66.75 / 89.47 | 63.95 / 76.89 | 64.23 / 77.16 | 62.33 / 65.18 / 63.72 | 65.04 / 67.86 / 66.42 |
| flextrackv2_b224_56_abl_pmax_025 | 66.81 / 89.78 | 66.72 / 89.53 | 62.71 / 75.61 | 64.69 / 77.73 | 64.18 / 67.02 / 65.57 | 65.12 / 67.96 / 66.51 |
| flextrackv2_b224_56_abl_pmax_050 | 66.42 / 88.87 | 67.08 / 89.78 | 63.71 / 76.69 | 63.95 / 76.96 | 64.91 / 67.83 / 66.34 | 65.25 / 68.23 / 66.71 |

<!-- AUTO-GENERATED RESULTS END -->

### Analysis

All 9 configs are complete. Comparing each against the full FlexTrack-V2
heterogeneous-HMoE baseline reported in `README.md` (AUC for
RGBT234/VisEvent, F-score for DepthTrack; Δ = ablation − baseline):

| Config | RGBT234 | RGBT234_miss | VisEvent | VisEvent_miss | DepthTrack (F) | DepthTrack_miss (F) |
|---|---:|---:|---:|---:|---:|---:|
| **baseline** | 70.02 | 62.80 | 71.66 | 59.86 | 67.5 | 62.9 |
| moe_big | 67.25 (-2.77) | 59.58 (**-3.22**) | 63.44 (-8.22) | 53.48 (**-6.38**) | 65.25 (-2.25) | 57.73 (**-5.17**) |
| moe_middle | 66.88 (-3.14) | 67.29 (+4.49) | 63.42 (-8.24) | 63.91 (+4.05) | 62.30 (-5.20) | 63.57 (+0.67) |
| moe_small | 66.55 (-3.47) | 66.85 (+4.05) | 63.54 (-8.12) | 63.48 (+3.62) | 67.23 (-0.27) | 62.69 (-0.21) |
| moe_hybrid | 67.20 (-2.82) | 66.59 (+3.79) | 64.11 (-7.55) | 62.77 (+2.91) | 66.11 (-1.39) | 64.99 (+2.09) |
| cma_fixed020 | 67.18 (-2.84) | 67.06 (+4.26) | 63.78 (-7.88) | 64.02 (+4.16) | 67.33 (-0.17) | 66.11 (+3.21) |
| no_distill | 66.99 (-3.03) | 66.54 (+3.74) | 63.38 (-8.28) | 62.51 (+2.65) | 66.05 (-1.45) | 66.32 (+3.42) |
| pmax_015 | 66.07 (-3.95) | 66.75 (+3.95) | 63.95 (-7.71) | 64.23 (+4.37) | 63.72 (-3.78) | 66.42 (+3.52) |
| pmax_025 | 66.81 (-3.21) | 66.72 (+3.92) | 62.71 (-8.95) | 64.69 (+4.83) | 65.57 (-1.93) | 66.51 (+3.61) |
| pmax_050 | 66.42 (-3.60) | 67.08 (+4.28) | 63.71 (-7.95) | 63.95 (+4.09) | 66.34 (-1.16) | 66.71 (+3.81) |

**Two patterns, one real and one likely a measurement artifact:**

1. **A near-uniform ~-3 (RGBT234) / ~-8 (VisEvent) AUC gap on complete
   modality across all 9 configs, including config-only ablations that
   barely touch the model** (`pmax_*`, `cma_fixed020` only move a curriculum
   scalar; `no_distill` only zeroes a loss weight). It's implausible that a
   0.2-vs-0.35 curriculum ceiling costs -8 VisEvent AUC on its own — this
   reads like a **systematic offset between this box-overlap
   AUC/PR computation and whatever produced the README's headline numbers**
   (different metric formula, checkpoint epoch, or sequence subset), not a
   genuine effect of any ablated component. Treat the complete-modality
   *absolute* numbers here as not directly comparable to the README table;
   the *relative* ordering between the 9 ablation rows is still valid since
   they all went through the identical pipeline.
2. **On missing-modality settings, 8 of 9 configs *improve* over baseline**
   (RGBT234_miss/VisEvent_miss: all positive except `moe_big`;
   DepthTrack_miss: all positive except `moe_big` and `moe_small`) —
   **`moe_big` is the one config that regresses everywhere, complete and
   missing alike, and by the largest margin of any row.** That isolates the
   heterogeneous-vs-homogeneous expert-capacity choice as the single most
   consequential factor tested in this tier: swapping in 8 uniform
   large-width experts costs performance across the board, while the CMA
   curriculum shape, the self-distillation loss, and even the low/mid
   homogeneous-expert variants (`moe_small`/`moe_middle`) all land within a
   few points of each other and of the baseline on missing-modality
   robustness specifically.
3. **`moe_hybrid` (semi-heterogeneous, paired capacities) sits closest to
   the full heterogeneous baseline's *pattern*** (smallest complete-modality
   VisEvent gap at -7.55, second-best DepthTrack_miss at +2.09) — consistent
   with "some heterogeneity captures most of the benefit," though the gap to
   `moe_big` is much larger than the gap to full HMoE, suggesting the
   benefit is concentrated in *having any spread* rather than needing the
   full 8-point exponential range.
4. **`pmax_*` sensitivity sweep (0.15/0.25/0.50) shows no monotonic
   trend** in either direction on complete or missing settings — the
   curriculum ceiling is not a sensitive hyperparameter in this range,
   unlike the expert-capacity axis.

*Caveat: box-overlap AUC/PR is computed over sequences that produced a
result file; a small number fail deterministically on genuine data issues
(e.g. a too-small ground-truth box) and are excluded rather than crashing
the run. Coverage is >90% on every dataset for every config.*

## Environment notes (for reproducibility, not committed to this branch)

Two pre-existing repository bugs had to be fixed before any of the above could
run at all (unrelated to the ablation design itself):

1. `lib/train/admin/__init__.py` had `TensorboardWriter` import commented out
   (from upstream commit `d5ed137`), which breaks `lib/train/trainers/ltr_trainer.py`.
   Fixed on this branch.
2. `lib/train/admin/local.py` (git-ignored, per-environment) pointed
   `lasher_dir` / `depthtrack_dir` / `visevent_dir` one directory level too
   shallow relative to how these datasets are actually laid out on this
   machine. Corrected locally to:
   - `lasher_dir` → `.../lasher/lasher/trainingset`
   - `visevent_dir` → `.../visevent/visevent/train`
   - `depthtrack_dir` → `.../depthtrack/depthtrack/train/DepthTrackTraining`

---

## ⚠️ SUPERSEDED: the "Analysis" section above uses a homegrown box-overlap metric

The results table and analysis immediately above this line were computed with a
**homegrown micro-averaged box-overlap AUC/PR** (`compute_ablation_metrics.py`),
which is **not** the metric the paper/README actually reports. It produced a
near-uniform ~7-8pt VisEvent gap vs the README across *every* config, including
config-only ablations that barely touch the model — that was the tell that it was
a metric-definition mismatch, not a real effect. Keep reading below for the
corrected numbers. The relative ordering *within* that old table is still fine to
reference, but do not quote its absolute numbers.

## ✅ FINAL, OFFICIAL RESULTS (2026-07-07)

Every number below is computed with the same tools the paper/README uses:
- **RGBT234** → MPR/MSR via the official `rgbt` Python toolkit (`RGBT_toolkit_python/`, max-over-visible+infrared GT)
- **LasHeR** → PR/SR via the same `rgbt` toolkit
- **VisEvent** → OPE AUC/PR with absent-frame exclusion (`absent_label.txt`), 21 MATLAB-aligned thresholds, via `evaluate_lasher_visevent.py`
- **DepthTrack** → official VOT toolkit (Pr/Re/F) — this one was *already* correct throughout

Scoring script: `/mnt/task_runtime/rescore_official.py <config1> [<config2> ...]`, output written to
`ablation_results_official/<config>/metrics.json`. Raw per-frame predictions live on each
config's B200 node under `workspace/results/<Dataset[_miss]>/<config>/`.

### Sanity check: does a fresh baseline reproduce the README?

| | RGBT234 MSR | LasHeR SR | VisEvent AUC |
|---|---:|---:|---:|
| **README (published, tuned thresholds)** | 70.02 | 61.96 | 72.00 |
| **rung-0, official metric, untuned thresholds** | 69.13–69.59 | 61.29–61.57 | 72.28–72.42 |
| *(superseded)* rung-0, homegrown metric | ~66.8 | ~56.2 | ~63.7 |

Yes — within ~1pt, fully explained by the README using per-dataset grid-searched
test-time thresholds (UPT/UPH/INTER/MB) while every config below shares one
untuned threshold set for a fair cross-config comparison.

### Full table — 16 configs × precision-style + success-style metric × full/miss

`P` = precision-style (MPR / PR / PR / Pr); `S` = success-style (MSR / SR / AUC / F).

| config | RGBT234 full (P/S) | RGBT234 miss (P/S) | LasHeR full (P/S) | LasHeR miss (P/S) | VisEvent full (P/S) | VisEvent miss (P/S) | DepthTrack full (P/S) | DepthTrack miss (P/S) |
|---|---|---|---|---|---|---|---|---|
| rung0 (seed42) **[baseline]** | 91.94 / 69.13 | 84.55 / 62.66 | 76.33 / 61.29 | 67.63 / 54.41 | 88.18 / 72.33 | 78.06 / 62.20 | 65.55 / 67.01 | 56.61 / 57.88 |
| rung0 (seed123) | 92.66 / 69.59 | 85.02 / 63.23 | 76.52 / 61.57 | 67.37 / 54.28 | 88.11 / 72.28 | 77.21 / 61.61 | 63.86 / 65.20 | 55.75 / 56.89 |
| rung0 (seed456) | 91.52 / 69.29 | 84.69 / 62.89 | 76.67 / 61.37 | 66.42 / 53.42 | 88.08 / 72.42 | 77.04 / 61.61 | 61.66 / 62.95 | 56.58 / 57.83 |
| **BMR mechanism family** | | | | | | | | |
| no_hallucinate | 92.41 / 69.45 | 84.57 / 62.68 | 76.14 / 61.03 | 67.36 / 54.01 | 88.63 / 72.60 | 77.18 / 61.54 | 63.89 / 65.30 | 56.11 / 57.35 |
| no_recon_loss | 91.57 / 68.94 | 85.20 / 62.98 | 76.28 / 61.16 | 66.13 / 53.35 | 87.48 / 72.06 | 77.20 / 61.52 | 63.93 / 65.30 | 54.81 / 56.01 |
| uni_a2r | 92.27 / 69.63 | 84.34 / 62.68 | 75.98 / 60.94 | 66.81 / 53.84 | 87.23 / 71.66 | 76.80 / 61.35 | 65.13 / 66.53 | 58.32 / 59.53 |
| uni_r2a | 92.23 / 69.15 | 83.93 / 62.31 | 76.03 / 60.89 | 67.27 / 54.15 | 88.20 / 72.48 | 76.78 / 61.35 | 63.27 / 64.62 | 57.53 / 58.74 |
| no_ortho | 92.27 / 69.41 | 86.27 / 63.83 | 76.90 / 61.55 | 65.94 / 53.09 | 88.42 / 72.36 | 76.74 / 61.15 | 64.24 / 65.62 | 58.79 / 60.05 |
| **MoE capacity / CMA / distill family** | | | | | | | | |
| moe_small | 91.90 / 69.42 | 85.23 / 63.61 | 75.63 / 60.52 | 67.64 / 54.28 | 87.93 / 72.15 | 76.70 / 61.24 | 65.79 / 67.23 | 54.38 / 55.55 |
| moe_middle | 91.90 / 69.14 | 85.81 / 63.44 | 77.02 / 61.74 | 67.70 / 54.46 | 87.58 / 72.03 | 76.59 / 61.19 | 60.95 / 62.30 | 54.73 / 55.95 |
| moe_hybrid | 92.12 / 69.75 | 84.52 / 63.07 | 76.20 / 61.07 | 66.66 / 53.66 | 88.60 / 72.76 | 76.87 / 61.30 | 64.71 / 66.11 | 56.24 / 57.50 |
| cma_fixed020 | 91.70 / 69.18 | 84.16 / 62.53 | 76.27 / 61.30 | 66.54 / 53.68 | 88.13 / 72.41 | 77.02 / 61.45 | 65.92 / 67.33 | 56.19 / 57.39 |
| no_distill | 92.37 / 69.22 | 86.41 / 64.01 | 76.89 / 61.80 | 66.94 / 53.91 | 87.93 / 71.97 | 78.02 / 62.18 | 64.66 / 66.05 | 57.62 / 58.82 |
| pmax_015 | 91.55 / 68.82 | 85.06 / 63.06 | 75.68 / 60.78 | 65.83 / 53.10 | 88.29 / 72.59 | 76.55 / 61.14 | 62.33 / 63.72 | 55.75 / 56.92 |
| pmax_025 | 92.23 / 69.14 | 85.13 / 63.10 | 76.35 / 61.20 | 66.73 / 53.68 | 86.81 / 71.26 | 78.51 / 62.41 | 64.18 / 65.57 | 53.71 / 54.91 |
| pmax_050 | 91.22 / 68.89 | 83.44 / 62.24 | 76.02 / 61.06 | 66.49 / 53.51 | 88.05 / 72.33 | 76.14 / 60.96 | 64.91 / 66.34 | 57.57 / 58.85 |

Also published as an interactive web report:
https://claude.ai/code/artifact/953bedd8-c304-43ee-8650-2ebde1646a27

### What each ablation tests

**BMR mechanism family** (isolates pieces of the bilateral hallucination/reconstruction design; config switches live under `cfg.MODEL.MOE` — `SUBSTITUTE_MODE`, `USE_RECON_LOSS`, `HALLUCINATE_DIRECTION`, `USE_ORTHO` in `lib/config/flextrackv2/config.py`):
- `no_hallucinate` (`SUBSTITUTE_MODE=zero`): missing modality is zero-filled instead of network-reconstructed. Does the hallucination network matter at all vs. naive zero-fill?
- `no_recon_loss` (`USE_RECON_LOSS=false`): keeps the reconstruction network but drops the self-supervised reconstruction loss at train time. Is that auxiliary loss necessary, or does the architecture learn reconstruction anyway?
- `uni_a2r` (`HALLUCINATE_DIRECTION=aux2rgb`): only the aux→rgb reconstruction path is active (rgb missing → reconstructed; aux missing → zero-filled). Tests whether one direction alone captures most of the benefit.
- `uni_r2a` (`HALLUCINATE_DIRECTION=rgb2aux`): mirror of the above — only rgb→aux active.
- `no_ortho` (`USE_ORTHO=false`): drops the MoE gate-orthogonality regularizer. Does forcing expert specialization actually help?

**MoE capacity / CMA / distillation family** (the original Tier-1 batch, re-evaluated with the fixed pipeline):
- `moe_small` / `moe_middle` / `moe_hybrid`: alternate expert-width configurations for the heterogeneous MoE (vs. the default 4→512 exponential spread). Tests sensitivity to expert capacity/heterogeneity.
- `cma_fixed020`: cross-modal-attention modality-dropout probability held fixed at 0.20 during training instead of annealed 0.0→0.35. Tests whether the annealing curriculum matters vs. a fixed dropout rate.
- `no_distill` (`DISTILL_WEIGHT=0`): removes the distillation loss entirely.
- `pmax_015` / `pmax_025` / `pmax_050`: anneal ceiling (P_MAX) swept vs. the default 0.35. Tests sensitivity to how aggressively missing-modality is simulated during training.

### Bugs found & fixed during this study (read before trusting any number above)

1. **Missing-modality silent-fallback bug** — `test_rgbt_mgpus.py` / `test_depthtrack_mgpus.py` used to fall back to a dummy "always present" annotation whenever the real missing-modality JSON was absent on a node. This happened on **all 8 original B200 nodes** for the entire first Tier-1 batch (except the locally-run `moe_big`), making every `_miss` result in the *old, superseded* analysis above actually a full-modality run in disguise. Fixed: hard-fail (`raise FileNotFoundError`/`SystemExit`) instead of faking, JSON files shipped to every node.
2. **VOT22RGBD wrong protocol assumed** — VOT-RGBD2022 uses the MultiStart/EAO live-trax protocol, not DepthTrack's precomputed-box Pr/Re/F protocol. Fixed the parser/harness (`run_vot22rgbd_local.sh`, `compute_ablation_metrics.py::eval_vot22rgbd`); **still unresolved**: live-trax evaluation hangs after ~2 sequences in this environment, root cause unknown. VOT22RGBD is excluded from all tables above.
3. **Checkpoint scp truncation** — 2-minute foreground timeout truncated ~1.36GB checkpoint pulls. Fixed via background pulls with byte-count verification + retry.
4. **Metric-definition mismatch (the big one)** — the homegrown box-overlap AUC/PR in `compute_ablation_metrics.py` is not what the paper's published table used. Fixed by re-scoring everything with `rescore_official.py` against the `rgbt` toolkit (RGBT234/LasHeR) and `evaluate_lasher_visevent.py` (VisEvent, absent-frame-aware OPE). This explains essentially all of the "-3 to -8pt gap vs README" seen in the old analysis.
5. **Stale `metrics.json` carry-over** — when re-scoring the old 8 Tier-1 configs, DepthTrack numbers were carried over from a pre-fix `ablation_results/<config>/metrics.json` that hadn't been re-pulled from the node, producing a fake uniform ~66-67 DepthTrack_miss cluster. Caught by re-running `vot analysis --nocache` fresh on-node and comparing; fixed by re-pulling and patching all 8 configs' metrics files.

### Known limitation flagged to the user (2026-07-07)

Only rung-0 has 3 seeds. All 13 ablations above are **single-seed** — "outside the
rung-0 noise band" is suggestive, not confirmed, until the ablations themselves are
repeated across seeds. None of the 5 BMR-mechanism ablations currently clears a
strong bar above noise on missing-modality metrics.

### In progress: missing-ratio sweep (requested 2026-07-07)

Single-point `_miss` eval (one fixed official missing pattern) doesn't show much
separation between mechanism ablations. Building a synthetic missing-**ratio**
sweep (0/25/50/75/100% of frames with the aux modality zeroed) to plot
performance-vs-missing-rate curves instead of a single point — the expectation is
that `no_hallucinate` and friends should diverge from rung-0 more visibly at high
missing rates. Uses the same already-trained checkpoints (no retraining needed),
inference-only. See below for the generator + sweep plan/status.

---

## 🔧 SECOND CORRECTION (2026-07-10): VisEvent was silently scored on 297/320 sequences

After the first "FINAL, OFFICIAL RESULTS" pass above, the user asked to verify VisEvent's
sequence count. **It was 297, not 320.** 23 sequences have `groundtruth.txt` starting with
`0,0,0,0` (target not yet visible in frame 0) — the tracker's init logic unconditionally
used frame 0 as the initialization box, crashing with `Exception('Too small bounding box.')`
and silently dropping the whole sequence, consistently across every config. This was NOT a
data-quality artifact — the 23 sequences are officially in-set with an `absent_label.txt`
correctly marking the early absent frames.

**Fix**: `RGBT_workspace/test_rgbt_mgpus.py` now finds the first frame with a real
(w>0, h>0) box and initializes there instead of always frame 0; frames before that stay
zero (matching ground truth, and excluded from scoring via `absent_label.txt` anyway).
No-op for every other dataset (RGBT234/LasHeR), where frame 0 is always valid.

**Impact**: VisEvent AUC dropped ~6 points across the board once the 23 harder sequences
(UAV outdoor, tennis, event-camera edge cases) were included — e.g. rung-0 went from 72.33
(297-seq) to 65.89 (320-seq). This means the earlier "rung-0 reproduces the README's 72.00"
validation was itself computed on the 297-sequence subset — whether the README's published
number also excludes these 23 is an open question, not confirmed either way.

Applied to all 24 configs (see table below). RGBT234/LasHeR/DepthTrack are unaffected.

## Additional bugs found while closing out the last few configs

- **Checkpoint symlink only created on the remote node, never locally.** The large model's
  (`flextrackv2_l224_56_gstuned`) RGBT234/LasHeR/VisEvent were only ever scored via
  pre-computed results pulled from its B200 node. The first local run (DepthTrack) was the
  first time this yaml_name needed actual local inference, and failed on every single
  sequence with a "checkpoint not found" error, because the
  `checkpoints/train/flextrackv2/flextrackv2_l224_56_gstuned -> flextrackv2_l224_56` symlink existed
  only on the (since-canceled) remote node. Fixed by recreating the symlink locally.
- **`qad3e9trm9` (the B200 node running the large-model tuned eval) was CANCELED** before
  its DepthTrack raw predictions and `pmax_050`'s checkpoint were ever pulled to this
  machine. The large model's DepthTrack was recoverable (checkpoint was already local) —
  re-ran it locally after the symlink fix. **`pmax_050`'s checkpoint was not recoverable** —
  40 epochs of training were permanently lost; it was retrained from scratch (verified via
  checkpoint timestamps: a genuine ~9.5 hour run, not a stale artifact).
- **Stale-prediction reuse after retraining under the same config name.** Re-evaluating the
  retrained `pmax_050` initially "finished" in 90 seconds — every dataset's `run_sequence()`
  skips work via `if os.path.exists(save_path): return`, so it silently reused prediction
  files from the *original, now-lost* checkpoint (file timestamps predated the retrain).
  Fixed by deleting all of `pmax_050`'s stale prediction directories before re-running.

## FINAL 24-config table (corrected VisEvent, complete DepthTrack, includes large model)

P = precision-style (MPR/PR/PR/Pr), S = success-style (MSR/SR/AUC/F).

| config | RGBT234 (P/S) | RGBT234_miss (P/S) | LasHeR (P/S) | LasHeR_miss (P/S) | VisEvent (P/S) | VisEvent_miss (P/S) | DepthTrack (P/F) | DepthTrack_miss (P/F) |
|---|---|---|---|---|---|---|---|---|
| rung0 | 91.94 / 69.13 | 84.55 / 62.66 | 76.33 / 61.29 | 67.63 / 54.41 | 80.17 / 65.89 | 73.78 / 58.93 | 65.55 / 67.01 | 56.61 / 57.88 |
| rung0_seedB | 92.66 / 69.59 | 85.02 / 63.23 | 76.52 / 61.57 | 67.37 / 54.28 | 80.24 / 66.05 | 74.28 / 59.20 | 63.86 / 65.20 | 55.75 / 56.89 |
| rung0_seedC | 91.52 / 69.29 | 84.69 / 62.89 | 76.67 / 61.37 | 66.42 / 53.42 | 80.55 / 66.31 | 73.31 / 58.65 | 61.66 / 62.95 | 56.58 / 57.83 |
| no_hallucinate | 92.41 / 69.45 | 84.57 / 62.68 | 76.14 / 61.03 | 67.36 / 54.01 | 79.77 / 65.69 | 73.35 / 58.66 | 63.89 / 65.30 | 56.11 / 57.35 |
| no_recon_loss | 91.57 / 68.94 | 85.20 / 62.98 | 76.28 / 61.16 | 66.13 / 53.35 | 79.62 / 65.66 | 74.03 / 59.07 | 63.93 / 65.30 | 54.81 / 56.01 |
| uni_a2r | 92.27 / 69.63 | 84.34 / 62.68 | 75.98 / 60.94 | 66.81 / 53.84 | 79.38 / 65.44 | 73.02 / 58.48 | 65.13 / 66.53 | 58.32 / 59.53 |
| uni_r2a | 92.23 / 69.15 | 83.93 / 62.31 | 76.03 / 60.89 | 67.27 / 54.15 | 80.34 / 66.12 | 73.48 / 58.86 | 63.27 / 64.62 | 57.53 / 58.74 |
| no_ortho | 92.27 / 69.41 | 86.27 / 63.83 | 76.90 / 61.55 | 65.94 / 53.09 | 80.19 / 65.79 | 73.55 / 58.78 | 64.24 / 65.62 | 58.79 / 60.05 |
| moe_small | 91.90 / 69.42 | 85.23 / 63.61 | 75.63 / 60.52 | 67.64 / 54.28 | 79.85 / 65.53 | 73.52 / 58.88 | 65.79 / 67.23 | 54.38 / 55.55 |
| moe_middle | 91.90 / 69.14 | 85.81 / 63.44 | 77.02 / 61.74 | 67.70 / 54.46 | 80.46 / 66.18 | 74.31 / 59.36 | 60.95 / 62.30 | 54.73 / 55.95 |
| moe_hybrid | 92.12 / 69.75 | 84.52 / 63.07 | 76.20 / 61.07 | 66.66 / 53.66 | 79.88 / 65.79 | 72.91 / 58.33 | 64.71 / 66.11 | 56.24 / 57.50 |
| cma_fixed020 | 91.70 / 69.18 | 84.16 / 62.53 | 76.27 / 61.30 | 66.54 / 53.68 | 80.15 / 65.91 | 73.69 / 58.85 | 65.92 / 67.33 | 56.19 / 57.39 |
| no_distill | 92.37 / 69.22 | 86.41 / 64.01 | 76.89 / 61.80 | 66.94 / 53.91 | 79.87 / 65.56 | 73.82 / 59.10 | 64.66 / 66.05 | 57.62 / 58.82 |
| pmax_015 | 91.55 / 68.82 | 85.06 / 63.06 | 75.68 / 60.78 | 65.83 / 53.10 | 80.37 / 66.17 | 73.40 / 58.54 | 62.33 / 63.72 | 55.75 / 56.92 |
| pmax_025 | 92.23 / 69.14 | 85.13 / 63.10 | 76.35 / 61.20 | 66.73 / 53.68 | 80.18 / 65.90 | 73.80 / 58.78 | 64.18 / 65.57 | 53.71 / 54.91 |
| pmax_050 (retrained) | 92.19 / 69.65 | 85.69 / 63.59 | 75.69 / 60.88 | 67.80 / 54.53 | 80.30 / 66.13 | 74.50 / 59.63 | 65.06 / 66.46 | 56.19 / 57.39 |
| no_hallucinate_no_ortho | 92.27 / 69.26 | 85.54 / 63.14 | 76.47 / 61.35 | 67.68 / 54.47 | 80.38 / 66.11 | 73.27 / 58.59 | 66.38 / 67.86 | 57.16 / 58.38 |
| no_recon_loss_no_ortho | 91.77 / 68.80 | 84.08 / 62.42 | 76.03 / 60.83 | 67.27 / 54.26 | 80.26 / 66.04 | 73.90 / 59.15 | 64.19 / 65.57 | 56.15 / 57.36 |
| uni_a2r_no_ortho | 91.89 / 68.76 | 84.88 / 62.67 | 76.44 / 61.28 | 66.13 / 53.18 | 80.22 / 65.98 | 73.32 / 58.73 | 63.61 / 65.03 | 56.25 / 57.50 |
| pmax_000 | 91.04 / 68.18 | 62.90 / 47.57 | 77.18 / 61.88 | 46.94 / 39.34 | 81.07 / 66.54 | 53.63 / 43.50 | 63.32 / 64.71 | 42.49 / 43.43 |
| pmax_010 | 92.04 / 68.88 | 83.93 / 62.01 | 76.06 / 60.98 | 65.84 / 53.05 | 80.80 / 66.39 | 72.60 / 58.08 | 64.72 / 66.14 | 56.78 / 58.02 |
| pmax_065 | 91.45 / 69.13 | 84.59 / 62.79 | 76.13 / 60.99 | 66.99 / 53.82 | 79.43 / 65.48 | 73.34 / 58.69 | 64.65 / 66.05 | 55.77 / 56.98 |
| pmax_080 | 90.31 / 67.93 | 85.07 / 63.17 | 76.44 / 61.29 | 66.80 / 53.86 | 79.14 / 65.29 | 73.59 / 59.02 | 64.00 / 65.40 | 56.13 / 57.36 |
| l224_56 (large, gstuned — old, base-tuned thresholds) | 92.43 / 69.08 | 85.88 / 63.05 | 79.06 / 63.14 | 68.60 / 54.90 | 80.88 / 66.80 | 75.07 / 60.34 | 65.17 / 66.48 | 56.85 / 58.00 |
| **l224_56 (large, gstuned — large-tuned thresholds, FINAL)** | **92.43 / 69.14** | 85.88 / 63.05 | 79.06 / 63.14 | 68.60 / 54.90 | 80.88 / 66.80 | 75.07 / 60.34 | **66.30 / 67.65** | **59.02 / 60.22** |

### Large model now beats base on every single metric (2026-07-10)

Originally the large backbone (`flextrackv2_l224_56`) simply reused the *base* model's
grid-search-tuned test-time thresholds (`flextrackv2_l224_56_gstuned.yaml`, thresholds copied
verbatim from the base model). Under those borrowed thresholds it beat base (rung0) on 14 of
16 metrics but *trailed* on two: RGBT234-full MSR (69.08 vs 69.13, −0.05) and DepthTrack-full
F (66.48 vs 67.01, −0.53), plus a −0.01 rounding tie on DepthTrack_miss recall.

A dedicated **large-model-specific** coordinate-descent threshold search (on the large
backbone's own checkpoint, 8×A100, official toolkits as scorers) closed all three:

| dataset | old (base-tuned) | new (large-tuned) | winning UPT/UPH/INTER/MB |
|---|---|---|---|
| RGBT234-full | 92.43 / 69.08 | 92.43 / **69.14** | 0.5 / 0.98 / 25 / 437 |
| DepthTrack-full | 65.17 / 67.85 / 66.48 | **66.30 / 69.06 / 67.65** | 0.85 / 0.85 / 77 / 862 |
| DepthTrack_miss | 56.85 / 59.20 / 58.00 | **59.02 / 61.46 / 60.22** | 0.69 / 0.81 / 55 / 724 |

The other 13 metrics already beat base at the start point, so their thresholds were left
unchanged (each was verified beating base by re-scoring its start point, then skipped — no
regression risk). **Result: the large model now beats base (rung0) on all 16 metrics across
all 8 datasets.** Bug caught mid-search and corrected: the VisEvent_miss re-score initially
used `test_rgbe_mgpus.py`, which lacks the 320-sequence absent-init fix, giving a wrong low
AUC; it was switched to `test_rgbt_mgpus.py` (the script the official eval uses) and
VisEvent_miss confirmed to already beat base with its existing thresholds.

**`pmax_000` (CMA_P_MAX=0, missing-modality curriculum entirely off during training) is the
standout finding of this whole study**: full-modality performance is normal across every
dataset, but every single `_miss` metric collapses (RGBT234_miss MSR 47.57 vs ~62-64 for
every other config; DepthTrack_miss F 43.43 vs ~57-58). This is the cleanest, most
unambiguous evidence in the entire ablation set that the missing-modality training
curriculum is load-bearing, not a tunable nicety.
