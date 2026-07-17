# Missing-Rate Degradation Experiment — Reproduction Spec

Reproduce the auxiliary-modality **missing-rate degradation curves** for FlexTrack-V2
on another machine. One trained model, evaluated at 5 inference-time missing rates
per benchmark, scored with each benchmark's official metric. **R0 (0% missing) must
equal the reported SOTA numbers** — that's the built-in sanity check.

---

## 0. Model (what "FlexTrack-V2" is)

- **Weights:** the **V54-trained** checkpoint
  `output/checkpoints/train/flextrackv2/flextrackv2_b224_54/FlexTrackV2_ep0040.pth.tar`
- **Config name to pass:** `flextrackv2_b224_56`
  (the loader special-cases `flextrackv2_b224_56 → checkpoint flextrackv2_b224_54`;
  it looks under `output/checkpoints/...` first, then `checkpoints/...`).
- **Test-time thresholds** = V56 grid-search winners, already in
  `experiments/flextrackv2/flextrackv2_b224_56.yaml`:

  | benchmark  | UPT  | UPH  | INTER | MB  |
  |------------|------|------|-------|-----|
  | RGBT234    | 0.5  | 0.93 | 25    | 312 |
  | LasHeR     | 0.6  | 0.98 | 10    | 100 |
  | VisEvent   | 0.8  | 0.95 | 70    | 500 |
  | DepthTrack | 0.77 | 0.85 | 55    | 862 |

> Do NOT use `flextrackv2_b224_56_abl_rung0` — that is a *separately-trained* ablation
> checkpoint (≠ V54) and will not reproduce the SOTA R0.

---

## 1. REQUIRED code patch (the bug that caused misalignment)

`lib/test/tracker/flextrackv2.py`, in `__init__`, right after `DATASET_NAME = dataset_name.upper()`.
Without this, dataset names like `RGBT234_missR050` don't match any threshold key and
**silently fall back to DEFAULT (UPT=1, INTER=999999 → online update OFF)**, giving a
different operating point than the main results.

```python
DATASET_NAME = dataset_name.upper()
# Ratio-sweep names ("<BASE>_MISSR###") are a missing-RATE variant of the full
# benchmark: resolve their online-update thresholds to the BASE dataset's, so the
# sweep runs at the SAME operating point as the reported full-modality results.
import re as _re
_mr = _re.search(r'_MISSR\d{3}$', DATASET_NAME)
if _mr:
    DATASET_NAME = DATASET_NAME[:_mr.start()]
# ... existing hasattr(cfg.TEST.UPT, DATASET_NAME) lookups follow unchanged ...
```

---

## 2. Missing-rate annotations (5 ratios)

**Prebuilt masks** (official `_miss` splits + all synthetic-ratio sweeps) are released on
HuggingFace — extract to `data_missing_modality/` at the repo root:

```bash
huggingface-cli download taryya/FlexTrackV2 missing_modality_annotations.tar.gz --local-dir .
tar -xzf missing_modality_annotations.tar.gz
```

Or regenerate the synthetic per-frame masks at ratios **0 / 25 / 50 / 75 / 100 %** with
`generate_missing_ratio_json.py` → `data_missing_modality/synthetic_ratio/<base>_missR<ratio>.json`
for `<base>` ∈ {`rgbt234`,`lasher`,`visevent`,`depthtrack`}.

**Convention (identical for all datasets):**
- format per frame `[rgb_present, aux_present]`
- **RGB always present** (`1.0`); the **auxiliary** modality (thermal / event / depth)
  is zeroed per-frame with probability `ratio/100`
- **frame 0 is never dropped** (always `[1,1]`) — init on the full modality
- reproducible seed `hash((name, ratio)) & 0xFFFFFFFF`

Regenerate on the new machine: `python3 generate_missing_ratio_json.py`

---

## 3. Datasets (full test sets)

| benchmark | seqs | path |
|---|---|---|
| RGBT234    | 234 | `/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234` |
| LasHeR     | 245 | `/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset` |
| VisEvent   | 320 | `/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test` |
| DepthTrack | 50  | `Depthtrack_workspace/sequences` (GT), VOT `votrgbd2021` stack |

---

## 4. Run the sweep (predictions)

Environment: **pin BLAS/OpenMP threads** or the worker pool thrashes CPU and GPUs
sit idle (observed util 5–30% unpinned → 80–88% pinned):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /mnt/task_runtime
CFG=flextrackv2_b224_56

# RGBT234 / LasHeR / VisEvent  (RGBT toolkit-style eval script)
for ds in RGBT234 LasHeR VisEvent; do
  for r in 000 025 050 075 100; do
    python3 RGBT_workspace/test_rgbt_mgpus.py \
      --yaml_name $CFG --dataset_name ${ds}_missR${r} \
      --threads 16 --num_gpus 8 --epoch 40
  done
done

# DepthTrack (separate script, same convention)
for r in 000 025 050 075 100; do
  python3 RGBT_workspace/test_depthtrack_mgpus.py \
    --yaml_name $CFG --dataset_name depthtrack_missR${r} \
    --threads 16 --num_gpus 8 --epoch 40
done
```

Predictions land in `workspace/results/<Base>_missR<ratio>/flextrackv2_b224_56/<seq>.txt`
(bbox `x,y,w,h`). The scripts skip a sequence whose result already exists (safe to re-run).
Driver used here: `run_missrate_fixed.sh` (adds OMP-pinning + skip-if-complete).

---

## 5. Score with OFFICIAL metrics (NOT box-overlap)

Toolkit at `RGBT_toolkit_python/src`. Example for RGBT234 (MPR/MSR); LasHeR is analogous
with `LasHeR` + `.PR()/.SR()`:

```python
import sys, os; sys.path.insert(0, "RGBT_toolkit_python/src")
from rgbt.dataset.rgbt234_dataset import RGBT234
GT = "/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234"
# each seq needs a <seq>.txt GT next to its folder: copy <seq>/init.txt -> <seq>.txt
r = RGBT234(gt_path=GT, seq_name_path=os.path.join(GT, "attr_txt/SequencesName.txt"))
for rt in ["000","025","050","075","100"]:
    r(tracker_name=f"R{rt}",
      result_path=f"workspace/results/RGBT234_missR{rt}/flextrackv2_b224_56",
      bbox_type="ltwh")
mpr, msr = r.MPR(), r.MSR()   # value = mpr["R000"][0]*100
```

- **RGBT234** → `RGBT234.MPR()/MSR()`
- **LasHeR**  → `LasHeR(gt_path=..., seq_name_path=.../lashertest.txt).PR()/SR()`
- **VisEvent** → OPE AUC/PR with absent-frame exclusion (21 overlap thresholds, PR@20px);
  see `VisEvent_SOT_Benchmark/eval_ours.py` / `rescore_official.py`
- **DepthTrack** → official VOT F-score via the VOT toolkit (`vot analysis` on the
  `votrgbd2021` stack), same as the main-table DepthTrack number

Helper that does all four at once: `rescore_official.py`.

---

## 6. Sanity check — R0 MUST equal SOTA

If the setup is correct, the **0%-missing** point reproduces the reported main numbers:

| benchmark | R0 expected (SOTA) | verified here |
|---|---|---|
| RGBT234 MPR/MSR   | 93.4 / 69.9 | **93.37 / 70.02** ✓ |
| LasHeR PR/SR      | 77.3 / 61.9 | **77.32 / 61.96** ✓ |
| VisEvent PR/SR    | 81.2 / 64.0 | (pending sweep) |
| DepthTrack F      | 67.8        | (pending sweep) |

If R0 ≠ SOTA → wrong checkpoint (must be V54), missing patch (§1, DEFAULT-threshold
fallback), or wrong metric (box-overlap instead of official).

---

## 7. Result so far (this machine, official metric, R0-aligned)

RGBT234 (thermal drop):  0%→100% = 93.37/70.02 → 89.15/66.70  (−4.22 MPR / −3.32 MSR)
LasHeR  (thermal drop):  0%→100% = 77.32/61.96 → 70.13/56.28  (−7.19 PR / −5.68 SR)

Graceful, monotonic degradation; thermal is more load-bearing on LasHeR than RGBT234.
Raw per-ratio numbers: `rgbt234_degradation_official.json`, `lasher_degradation_official.json`.
