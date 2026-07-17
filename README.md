# FlexTrack-V2

### Adaptive and Robust Multimodal Tracking under Incomplete Modalities

> **Journal (IEEE TPAMI) extension of** *"What You Have is What You Track: Adaptive and Robust Multimodal Tracking"* (ICCV 2025)
> [[ICCV paper](https://arxiv.org/abs/2507.05899)] · [[conference code (FlexTrack)](https://github.com/supertyd/FlexTrack)]

FlexTrack-V2 is a unified multimodal tracker that maintains state-of-the-art accuracy when **all** modalities are available, and degrades gracefully — often *improving over the complete-modality baseline of prior work* — when a modality is **partially or fully missing** (sensor dropouts, de-synchronization). A single model handles RGB-Thermal, RGB-Depth, and RGB-Event tracking under both complete and missing-modality protocols.

<p align="center"><i>complete-modality parity · missing-modality superiority</i></p>

---

## Highlights

**Bilateral Modality-specific feature Reconstruction & Hallucination (BMR).**
Rather than only *masking* an absent modality, FlexTrack-V2 actively *reconstructs* it: bilateral cross-modal projection modules hallucinate the missing modality's features from the available stream, trained with a self-supervised reconstruction loss applied only where the target modality is genuinely present.

**Heterogeneous Mixture-of-Experts fusion (HMoE).**
Eight experts of exponentially growing capacity `[4, 8, 16, 32, 64, 128, 256, 512]` with noisy top-k routing and a gate-orthogonality regularizer, so the fusion cost adapts to modality availability and scene difficulty.

**Curriculum Missing Augmentation with self-distillation (CMA).**
The per-sample missing probability grows over training on a curriculum; a complete-modality *teacher* pass distills a degraded *student* pass of the same live model, aligning representations across completeness levels.

**Unified architecture.**
A Fast-iTPN encoder, a Mamba (selective state-space) interaction neck, and a center-based prediction head.

---

## Results

FlexTrack-V2 vs. the ICCV baseline (FlexTrack). Higher is better; **bold** marks the winner. DepthTrack follows the official VOT-toolkit protocol (Precision / Recall / F-score); other datasets use Success (AUC / SR / MSR) and Precision (PR / MPR).

### 🟢 Complete modality

| Dataset | Metric | FlexTrack | **FlexTrack-V2** | Δ |
| :--- | :--- | :---: | :---: | :---: |
| LasHeR   | PR / AUC   | 77.28 / **61.97** | **77.32** / 61.96 | +0.04 / −0.01 |
| RGBT234  | MPR / MSR  | 92.72 / 69.96 | **93.37 / 70.02** | +0.65 / +0.06 |
| VisEvent | AUC / PR   | 71.19 / 87.14 | **71.66 / 87.47** | +0.47 / +0.33 |
| DepthTrack | Pr / Re / F | **67.1** / 66.9 / 67.0 | 66.1 / **69.0 / 67.5** | −1.0 / +2.1 / +0.5 |

### 🔴 Missing modality

| Dataset | Metric | FlexTrack | **FlexTrack-V2** | Δ |
| :--- | :--- | :---: | :---: | :---: |
| LasHeR\_miss   | PR / AUC   | 65.11 / 52.34 | **67.64 / 54.24** | **+2.53 / +1.90** |
| RGBT234\_miss  | MPR / MSR  | 84.07 / 62.73 | **85.71 / 62.80** | +1.64 / +0.07 |
| VisEvent\_miss | AUC / PR   | 58.29 / 73.74 | **59.86 / 75.20** | +1.57 / +1.46 |
| DepthTrack\_miss | Pr / Re / F | 59.6 / 56.1 / 57.8 | **61.6 / 64.3 / 62.9** | **+2.0 / +8.2 / +5.1** |

**Takeaways.** On complete modality FlexTrack-V2 is on par with the SOTA conference model (no accuracy sacrificed). On missing modality it wins across the board, with the largest gains on DepthTrack\_miss (**+5.1 F**) and LasHeR\_miss (**+1.90 AUC / +2.53 PR**) — exactly where active reconstruction (BMR) and completeness-curriculum distillation (CMA) are designed to help.

---

## Installation

```bash
git clone https://github.com/supertyd/FlexTrackV2.git
cd FlexTrackV2
conda create -n flextrack python=3.11
conda activate flextrack
bash install.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Model weights

Two released variants (each a single self-describing model — the config carries its own
`TEST.CHECKPOINT`, no code changes needed), from
**[huggingface.co/taryya/FlexTrackV2](https://huggingface.co/taryya/FlexTrackV2)**:

| Variant | Backbone | Config | Checkpoint |
|---|---|---|---|
| **FlexTrack-V2** (base) | Fast-iTPN-B | `experiments/flextrackv2/flextrackv2.yaml` | `FlexTrackV2.pth.tar` (1.4 GB) |
| **FlexTrack-V2 Large** | Fast-iTPN-L | `experiments/flextrackv2/flextrackv2_large.yaml` | `FlexTrackV2_large.pth.tar` (4.5 GB) |

```bash
mkdir -p checkpoints
# base
huggingface-cli download taryya/FlexTrackV2 FlexTrackV2.pth.tar       --local-dir checkpoints
# large (optional)
huggingface-cli download taryya/FlexTrackV2 FlexTrackV2_large.pth.tar --local-dir checkpoints
sha256sum -c checkpoints/*.sha256    # base e7cbfb20…  ·  large fc5e4193…
```

Run the large variant by passing `--yaml_name flextrackv2_large` to any eval driver, e.g.:

```bash
python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 \
       --yaml_name flextrackv2_large --dataset_name RGBT234 --threads 8
```

## Data preparation

Place the multimodal tracking datasets under `./data`:

```
${ROOT}
 -- data
     -- LasHeR          # RGB-Thermal
     -- VisEvent        # RGB-Event
     -- DepthTrack      # RGB-Depth
     -- RGBT234         # RGB-Thermal (test)
     -- VOT22RGBD       # RGB-Depth (test)
```

Set project paths (edit the generated files afterwards if needed):

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
# lib/train/admin/local.py       # training data roots
# lib/test/evaluation/local.py   # evaluation data roots
```

## Training

Download the backbone pre-train and place it at the path given by `MODEL.ENCODER.PRETRAIN_TYPE`
in the config (default `pretrained/FlexTrackV2_backbone_pretrain.pth.tar`), then:

```bash
bash scripts/train.sh            # 8-GPU DDP; equivalently:
# torchrun --nproc_per_node 8 lib/train/run_training.py \
#   --script flextrackv2 --config flextrackv2 --save_dir .
```

The production configuration (`experiments/flextrackv2/flextrackv2.yaml`) trains jointly on VisEvent + LasHeR + DepthTrack with the BMR-HMoE fusion, CMA curriculum, and self-distillation objective. Checkpoints are written to `output/checkpoints/train/flextrackv2/`.

## Testing

The multi-GPU eval harnesses live in `RGBT_workspace/` (RGB-T / RGB-D) and `RGBE_workspace/`
(RGB-E). Convenience wrappers run each benchmark and write boxes to
`workspace/results/<DS>/flextrackv2/`:

```bash
bash scripts/eval/rgbt234.sh      # RGBT234
bash scripts/eval/lasher.sh       # LasHeR
bash scripts/eval/visevent.sh     # VisEvent
bash scripts/eval/depthtrack.sh   # DepthTrack
```

Missing modality — first download the missing-modality annotation masks (official `_miss`
splits for all four benchmarks + the synthetic missing-rate sweeps) from HuggingFace and extract
them to `data_missing_modality/` at the repo root:

```bash
huggingface-cli download taryya/FlexTrackV2 missing_modality_annotations.tar.gz --local-dir .
tar -xzf missing_modality_annotations.tar.gz          # -> data_missing_modality/
```

Then pass the `_miss` dataset variant to the same driver, e.g.:

```bash
python RGBT_workspace/test_rgbt_mgpus.py --script_name flextrackv2 --yaml_name flextrackv2 \
       --dataset_name LasHeR_miss --threads 8
```

The masks are per-frame `[rgb_present, aux_present]` flags; the drivers resolve them relative to
the repo root, so no path editing is needed. For the synthetic missing-rate degradation sweep
(`_missR000…_missR100`, `_missRGB###`) see [docs/REPRODUCE_missrate.md](docs/REPRODUCE_missrate.md).

Score the produced boxes with the tools in `scripts/tools/`
(`evaluate_lasher.py`, `evaluate_lasher_visevent.py`, `evaluate_depthtrack.py`).
DepthTrack / VOT-RGBD also support the official VOT toolkit; see `lib/test/vot/`.

> Checkpoints, workspaces, datasets, and raw result archives are **not** tracked in this repository
> (see `.gitignore`). The model checkpoint is released on
> [HuggingFace](https://huggingface.co/taryya/FlexTrackV2).

## TPAMI ablation study

The ablations backing the journal extension (see [docs/ABLATIONS.md](docs/ABLATIONS.md) and the
interactive [results/ablation_dashboard.html](results/ablation_dashboard.html)) run across multiple
machines in parallel — see [docs/MULTI_NODE_SETUP.md](docs/MULTI_NODE_SETUP.md) for the multi-node
setup log if reproducing on new hardware. The missing-modality degradation protocol is documented in
[docs/REPRODUCE_missrate.md](docs/REPRODUCE_missrate.md).

## Repository layout

```
lib/                         core model, tracker, training code
  models/flextrackv2/        FlexTrackV2 network + BMR-HMoE fusion
  test/tracker/flextrackv2.py       inference tracker
  test/parameter/flextrackv2.py     checkpoint / param loader
experiments/flextrackv2/
  flextrackv2.yaml           production config (weights path + operating point)
scripts/                     train.sh · eval/ per-benchmark drivers · tools/ scoring utils
RGBT_workspace/, RGBE_workspace/   multi-GPU eval harnesses
docs/                        reproduction & analysis docs
results/                     published metrics, figures, ablation dashboard
archive/                     historical configs, one-off scripts, dev notes (not needed to use the model)
```

---

## Acknowledgments

The tracking framework builds on [MCITrack](https://github.com/kangben258/MCITrack) (AAAI 2025) and the conference release [FlexTrack](https://github.com/supertyd/FlexTrack) (ICCV 2025).

## Citation

```bibtex
@inproceedings{tan2025flextrack,
  title     = {What You Have is What You Track: Adaptive and Robust Multimodal Tracking},
  author    = {Tan, Yuedong and Shao, Jiawei and Zamfir, Eduard and Li, Ruanjun and
               An, Zhaochong and Ma, Chao and Paudel, Danda and Van Gool, Luc and
               Timofte, Radu and Wu, Zongwei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
