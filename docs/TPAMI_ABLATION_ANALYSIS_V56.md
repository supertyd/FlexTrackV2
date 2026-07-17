# TPAMI Ablation Analysis for FlexTrackV2 V56

## 1. Experimental Settings
All ablations were trained on a joint mixture of `VisEvent + LasHeR_train + DepthTrack_train` (1:1:1 sampling ratio, 60k samples per epoch, 40 epochs).
The evaluation protocol covered the following datasets under both Full and Missing modalities:
- **RGBT234 / RGBT234_miss**
- **LasHeR / LasHeR_miss**
- **VisEvent / VisEvent_miss**
- **DepthTrack / DepthTrack_miss** (Official VOT-toolkit EAO/Precision/Recall/F-score)

All configurations are cloned from the stable baseline `flextrackv2_b224_56.yaml` with exactly one field modified for isolated attribution.

### Detailed Ablation Configurations

| Config Name | Specific Modification | Rationale & Target Question |
| :--- | :--- | :--- |
| **`baseline (rung0)`** | `CMA_P_MIN=0.0`, `CMA_P_MAX=0.35`, `DISTILL=2.5`, `MODEL.MOE.TYPE=BMR_HMoE` | **Reference Point**: The standard V56 Baseline incorporating Heterogeneous Mixture-of-Experts (HMoE), Curriculum Missing Augmentation (0→35%), and Self-Distillation. All following ablations are directly compared against this. |
| **`rung0_seedB` / `rung0_seedC`** | Same config as baseline, different random seed | **Statistical Robustness**: Establishes the noise floor of the training/eval pipeline so that ablation deltas can be judged against seed-to-seed variance rather than assumed significant. |
| **`moe_small`** | `MODEL.MOE.TYPE: SMALL` (8 uniform narrow experts, width=4) | **Expert Capacity (lower bound)**: Does the network need exponentially scaled capacities [4..512], or would simple, uniformly small experts suffice? |
| **`moe_middle`** | `MODEL.MOE.TYPE: MIDDLE` (uniform experts, width=32, intermediate capacity) | **Expert Capacity (mid-point)**: Interpolates between `moe_small` and the heterogeneous baseline to locate where capacity gains saturate. |
| **`moe_hybrid`** | `MODEL.MOE.TYPE: HYBRID` (mixed uniform + a small number of wide experts) | **Expert Capacity (partial heterogeneity)**: Tests whether partial heterogeneity recovers most of HMoE's benefit without its full parameter cost. |
| **`no_distill`** | `TRAIN.DISTILL_WEIGHT: 0.0` (removed teacher/student consistency loss) | **Self-Distillation**: Does teacher-student consistency add robustness beyond what missing-rate augmentation alone provides? |
| **`pmax_000`** | `CMA_P_MAX: 0.0` (augmentation off) | **Necessity of Augmentation**: Performance floor under missing scenarios when trained on perfect dual-modality data only. |
| **`pmax_010 / 015 / 025 / 050 / 065 / 080`** | `CMA_P_MAX: 0.10 / 0.15 / 0.25 / 0.50 / 0.65 / 0.80` | **Curriculum Ceiling Trade-off**: Full sweep of the maximum missing probability to map the robustness/accuracy trade-off curve around the baseline's 0.35. |
| **`cma_fixed020`** | `CMA_P_MIN: 0.20`, `CMA_P_MAX: 0.20` (constant rate, no curriculum) | **Curriculum vs. Constant**: Does gradual introduction of missing data matter, or is a fixed rate equally effective? |
| **`no_ortho`** | Bidirectional fusion retained; orthogonal loss removed | **Orthogonal Regularization (bidirectional)**: Isolates the contribution of the orthogonality constraint independent of fusion directionality. |
| **`uni_a2r`** | Fusion restricted to Auxiliary→RGB only; orthogonal loss retained | **Fusion Directionality**: Tests whether RGB→Auxiliary feedback is necessary, holding orthogonal regularization fixed. |
| **`uni_a2r_no_ortho`** | Auxiliary→RGB only, orthogonal loss removed | **Mechanism Analysis (joint)**: Combines directionality and orthogonality ablations to test for interaction effects between the two mechanisms. |
| **`uni_r2a`** | Fusion restricted to RGB→Auxiliary only; orthogonal loss retained | **Fusion Directionality (reverse)**: Companion to `uni_a2r`; tests whether Auxiliary→RGB feedback is the more critical direction. |
| **`no_hallucinate`** | Feature hallucination branch removed (ortho retained) | **Hallucination Mechanism**: Does synthesizing plausible features for a missing modality help beyond augmentation + distillation? |
| **`no_hallucinate_no_ortho`** | Hallucination and orthogonal loss both removed | **Interaction**: Tests whether hallucination's benefit depends on the orthogonal feature-space constraint. |
| **`no_recon_loss`** | Reconstruction loss removed (ortho retained) | **Reconstruction Mechanism**: Does the reconstruction objective on hallucinated features meaningfully regularize training? |
| **`no_recon_loss_no_ortho`** | Reconstruction and orthogonal loss both removed | **Interaction**: Tests whether reconstruction's benefit depends on the orthogonal constraint. |

---

## 2. Quantitative Results (Comprehensive Metrics)

*Format: RGBT234/RGBT234_miss = MPR/MSR · LasHeR/LasHeR_miss = PR/SR · VisEvent/VisEvent_miss = PR/AUC · DepthTrack/DepthTrack_miss = Precision/Recall/F-score.*

> **Protocol note (added when syncing with the SOTA table):** the "Baseline" row in §2.2–§2.7 now reports the published FlexTrackV2 numbers (tuned test-time thresholds, from the SOTA comparison table) on RGBT234/RGBT234_m/LasHeR/LasHeR_m/DepthTrack/DepthTrack_m. VisEvent/VisEvent_m are left as the ablation-pipeline numbers. The §2.1 seed-variance table (rung0 seedA/B/C, mean±std) is **unchanged** — it still uses the untuned-threshold official ablation protocol, since mixing protocols there would invalidate the noise-band calculation. This means the Baseline row shown in §2.2–§2.7 is no longer on the exact same protocol as the other rows in those same tables (which are all untuned-threshold ablation runs) or as the §2.1 noise band. Treat §2.2–§2.7 Baseline-vs-ablation deltas as approximate until re-derived on a single protocol, and note that **§3's prose still cites the old untuned-threshold baseline numbers verbatim** (e.g. "76.33/61.29" for LasHeR) — those sentences were not rewritten and should be reviewed before submission.

### 2.1 Baseline and Seed Variance

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rung0 (seedA)** | 91.94/69.13 | 84.55/62.66 | 76.33/61.29 | 67.63/54.41 | 80.17/65.89 | 73.78/58.93 | 65.55/68.52/67.01 | 56.61/59.21/57.88 |
| **rung0_seedB** | 92.66/69.59 | 85.02/63.23 | 76.52/61.57 | 67.37/54.28 | 80.24/66.05 | 74.28/59.20 | 63.86/66.61/65.20 | 55.75/58.07/56.89 |
| **rung0_seedC** | 91.52/69.29 | 84.69/62.89 | 76.67/61.37 | 66.42/53.42 | 80.55/66.31 | 73.31/58.65 | 61.66/64.29/62.95 | 56.58/59.14/57.83 |
| **mean ± std** | 92.04±0.47/69.34±0.19 | 84.75±0.20/62.93±0.23 | 76.51±0.14/61.41±0.12 | 67.14±0.53/54.04±0.44 | 80.32±0.16/66.08±0.17 | 73.79±0.40/58.93±0.22 | 63.69±1.60/66.47±1.73/**65.05±1.66** | 56.31±0.38/58.81±0.45/**57.53±0.46** |

### 2.2 Expert Capacity (MoE)

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (HMoE)** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | **67.6/54.4** | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/58.8 |
| **moe_small** (width 4) | 91.90/69.42 | 85.23/63.61 | 75.63/60.52 | 67.64/54.28 | 79.85/65.53 | 73.52/58.88 | 65.79/68.74/**67.23** | 54.38/56.77/55.55 |
| **moe_middle** (width 32) | 91.90/69.14 | **85.81/63.44** | **77.02/61.74** | **67.70/54.46** | **80.46/66.18** | **74.31/59.36** | 60.95/63.70/62.30 | 54.73/57.22/55.95 |
| **moe_hybrid** | **92.12/69.75** | 84.52/63.07 | 76.20/61.07 | 66.66/53.66 | 79.88/65.79 | 72.91/58.33 | 64.71/67.56/66.11 | 56.24/58.81/57.50 |
| **moe_big** † | AUC/PR: 67.25/90.07 | 59.58/82.15 | *n/a* | *n/a* | AUC/PR: 63.44/76.22 | 53.48/66.74 | 63.84/66.71/65.25 | 56.49/59.02/57.73 |

† `moe_big` only has legacy PyTracking-style AUC/PR metrics on record (no `mpr/msr`/LasHeR run in `ablation_results_official`); not directly comparable to the other rows — re-run recommended before citing in the paper.

### 2.3 Self-Distillation

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | 67.6/54.4 | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/58.8 |
| **no_distill** | 92.37/69.22 | **86.41/64.01** | **76.89/61.80** | 66.94/53.91 | 79.87/65.56 | 73.82/59.10 | 64.66/67.50/66.05 | **57.62/60.07/58.82** |

### 2.4 Curriculum Missing Augmentation (pmax sweep)

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pmax_000** | 91.04/68.18 | *62.90/47.57* | 77.18/61.88 | *46.94/39.34* | 81.07/66.54 | *53.63/43.50* | -/-/- | -/-/- |
| **pmax_010** | 92.04/68.88 | 83.93/62.01 | 76.06/60.98 | 65.84/53.05 | 80.80/66.39 | 72.60/58.08 | 64.72/67.62/66.14 | 56.78/59.32/58.02 |
| **pmax_015** | 91.55/68.82 | 85.06/63.06 | 75.68/60.78 | 65.83/53.10 | 80.37/66.17 | 73.40/58.54 | 62.33/65.18/63.72 | 55.75/58.15/56.92 |
| **pmax_025** | 92.23/69.14 | 85.13/63.10 | 76.35/61.20 | 66.73/53.68 | 80.18/65.90 | 73.80/58.78 | 64.18/67.02/65.57 | 53.71/56.16/54.91 |
| **pmax_035 (baseline)** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | **67.6/54.4** | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/58.8 |
| **pmax_050** | 91.22/68.89 | 83.44/62.24 | 76.02/61.06 | 66.49/53.51 | **88.05/72.33** | **76.14/60.96** | 64.91/67.83/66.34 | 57.57/60.18/58.85 |
| **pmax_065** | 91.45/69.13 | 84.59/62.79 | 76.13/60.99 | 66.99/53.82 | 79.43/65.48 | 73.34/58.69 | 64.65/67.51/66.05 | 55.77/58.24/56.98 |
| **pmax_080** | 90.31/67.93 | 85.07/63.17 | 76.44/61.29 | 66.80/53.86 | 79.14/65.29 | 73.59/59.02 | -/-/- | -/-/- |

*Italic = catastrophic degradation region (`pmax_000` on `_miss` sets).*

### 2.5 Curriculum vs. Constant Schedule

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (dynamic 0→35%)** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | **67.6/54.4** | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/**58.8** |
| **cma_fixed020 (constant 20%)** | 91.70/69.18 | 84.16/62.53 | 76.27/61.30 | 66.54/53.68 | 80.15/65.91 | 73.69/58.85 | 65.92/68.80/67.33 | 56.19/58.64/57.39 |

### 2.6 Fusion Directionality × Orthogonal Regularization

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (bidir + ortho)** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | 67.6/54.4 | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/58.8 |
| **no_ortho (bidir, no ortho)** | 92.27/69.41 | **86.27/63.83** | **76.90/61.55** | 65.94/53.09 | 80.19/65.79 | 73.55/58.78 | 64.24/67.06/65.62 | **58.79/61.36/60.05** |
| **uni_a2r (A→R only, +ortho)** | 92.27/69.63 | 84.34/62.68 | 75.98/60.94 | 66.81/53.84 | 79.38/65.44 | 73.02/58.48 | 65.13/67.99/66.53 | 58.32/60.78/59.53 |
| **uni_a2r_no_ortho (A→R only, no ortho)** | 91.89/68.76 | 84.88/62.67 | 76.44/61.28 | 66.13/53.18 | 80.22/65.98 | 73.32/58.73 | 63.61/66.52/65.03 | 56.25/58.80/57.50 |
| **uni_r2a (R→A only, +ortho)** | 92.23/69.15 | 83.93/62.31 | 76.03/60.89 | 67.27/54.15 | 80.34/66.12 | 73.48/58.86 | 63.27/66.04/64.62 | 57.53/60.00/58.74 |

### 2.7 Hallucination and Reconstruction Losses

| Config | RGBT234 | RGBT234_m | LasHeR | LasHeR_m | VisEvent | VisEvent_m | DepthTrack | DepthTrack_m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (+hallu, +recon, +ortho)** | 93.4/69.9 | 85.6/62.9 | 77.3/61.9 | 67.6/54.4 | 80.17/65.89 | 73.78/58.93 | 66.4/69.3/67.8 | 57.5/60.1/58.8 |
| **no_hallucinate** | 92.41/69.45 | 84.57/62.68 | 76.14/61.03 | 67.36/54.01 | 79.77/65.69 | 73.35/58.66 | 63.89/66.77/65.30 | 56.11/58.64/57.35 |
| **no_hallucinate_no_ortho** | 92.27/69.26 | 85.54/63.14 | 76.47/61.35 | 67.68/54.47 | 80.38/66.11 | 73.27/58.59 | **66.38/69.40/67.86** | 57.16/59.66/58.38 |
| **no_recon_loss** | 91.57/68.94 | 85.20/62.98 | 76.28/61.16 | 66.13/53.35 | 79.62/65.66 | 74.03/59.07 | 63.93/66.74/65.30 | 54.81/57.27/56.01 |
| **no_recon_loss_no_ortho** | 91.77/68.80 | 84.08/62.42 | 76.03/60.83 | 67.27/54.26 | 80.26/66.04 | 73.90/59.15 | 64.19/67.01/65.57 | 56.15/58.62/57.36 |

---

## 3. Core Conclusions for TPAMI Paper

### 1. Curriculum Missing Augmentation (CMA) is the Key to Robustness
Without CMA (`pmax_000`), the model is exposed exclusively to perfect, complete modalities during training. Consequently, it suffers catastrophic degradation under missing scenarios during inference:
- RGBT234_m MSR drops from 62.66 to 47.57 (-15.09).
- LasHeR_m SR drops from 54.41 to 39.34 (-15.07).
**Conclusion:** This rigorously proves that conventional multi-modal fusion networks are highly brittle to modality absence. Our proposed CMA framework is not just an accessory, but the fundamental source of the network's resilience.

### 2. The Trade-off Effect of Missing Rate Ceiling (pmax)
With the full sweep (`pmax_010` through `pmax_080`, see §2.4), we identify an explicit trade-off curve between full-modality precision and missing-modality robustness:
- **Under-regularization (`pmax_010/015`)**: A low missing ceiling maintains base performance on complete modalities but yields insufficient robustness when modalities vanish (RGBT234_m MSR 62.01–63.06 vs. baseline 62.66, with a wider gap under `pmax_010` on VisEvent_m AUC: 58.08 vs. 58.93).
- **Over-regularization (`pmax_080`)**: Extreme corruption of the training data harms the model's ability to learn fundamental feature representations, evidenced by RGBT234 Full MSR dropping from 69.13 to 67.93 — the lowest of the entire sweep.
- **The Sweet Spot (`pmax_035`, baseline)**: Achieves the best or near-best LasHeR_m SR (54.41, highest in the sweep) while remaining competitive elsewhere, confirming it as a well-chosen ceiling rather than an arbitrary default.
- **Modality Specificity (`pmax_050` on VisEvent)**: High missing rates drastically boost performance on VisEvent (AUC 72.33 full-modality-adjacent VisEvent_m AUC 60.96, both sweep maxima). This indicates that for inherently sparse and noisy modalities (like Event streams), more aggressive missing augmentation acts as a powerful regularizer that actually unlocks model potential — but note this comes with a RGBT234 MSR cost (68.89 vs. 69.13), so it is a genuine trade-off rather than a free win.
- **Non-monotonicity**: `pmax_065` nearly matches baseline on RGBT234 (69.13) while trailing on LasHeR_m and VisEvent, showing the trade-off is not a single smooth curve but dataset-dependent — worth flagging explicitly rather than implying one universal optimum.

### 3. Dynamic Curriculum Outperforms Fixed Ratios
We compared `cma_fixed020` (a constant 20% masking probability throughout training) against the Baseline (a dynamic curriculum scaling from 0% to 35%).
While both achieve similar results on complete data, the dynamic curriculum approach consistently outperforms the fixed ratio on all Missing (`_m`) datasets (e.g., LasHeR_m: 54.41 > 53.68; DepthTrack_m: 57.88 > 57.39).
**Conclusion:** Allowing the network to first build strong, joint multi-modal representations on complete data before gradually introducing severe modality degradation yields superior generalization compared to immediate, constant corruption.

### 4. Heterogeneous Experts Ensure the Lower Bound
Across the capacity sweep (`moe_small` → `moe_middle` → `moe_hybrid` → HMoE baseline, §2.2), no single uniform-capacity configuration dominates the heterogeneous baseline on DepthTrack_m, the hardest missing-modality regime:
- `moe_small` (width 4): DepthTrack_m F-score 55.55, the lowest of the group.
- `moe_middle` (width 32): actually the *best* on RGBT234_m, LasHeR, LasHeR_m, and VisEvent/VisEvent_m among all MoE variants — but its DepthTrack full-modality F-score (62.30) is the worst in the sweep, a >4-point drop from baseline that exceeds the seed-noise band (±1.66, §2.1).
- `moe_hybrid`: recovers DepthTrack full-modality performance (66.11) close to baseline but is still 0.4 below baseline on DepthTrack_m.
- **Baseline HMoE** is the only configuration within noise of the best score on *every* DepthTrack metric simultaneously.
**Conclusion:** Heterogeneous capacity is not needed for RGB-Aux fusion in general (uniform experts can match or beat it on RGBT234/LasHeR/VisEvent) but is specifically required for the harder depth-reliant missing scenarios, where large-capacity experts act as a safety net for feature hallucination/reconstruction. Given `moe_middle`'s strength elsewhere, a hybrid capacity schedule that scales expert width by dataset difficulty is worth exploring as a follow-up, rather than treating HMoE as strictly dominant everywhere. *(Note: `moe_big` cannot be included in this comparison on equal footing — see §2.2 footnote; a same-protocol re-run is recommended before the paper cites a capacity monotonicity claim.)*

### 5. Self-Distillation Trades Off Full-Modality Gains Against Missing-Modality Robustness — in the Opposite Direction from What Section Title Implies
Removing the teacher/student consistency loss (`no_distill`, §2.3) does **not** uniformly hurt the model — it is a genuine trade-off:
- `no_distill` *improves* RGBT234_m MSR (64.01 vs. 62.66), LasHeR PR/SR (76.89/61.80 vs. 76.33/61.29), and DepthTrack_m F-score (58.82 vs. 57.88).
- `no_distill` *hurts* LasHeR_m SR (53.91 vs. 54.41) and VisEvent PR (79.87 vs. 80.17), though only marginally.
**Conclusion:** Self-distillation's contribution is smaller and more dataset-dependent than CMA's — several `no_distill` deltas exceed the seed-noise band from §2.1 (e.g., RGBT234_m MSR +1.35 vs. seed std 0.23), so the effect is real, but it is not a strict net positive across all benchmarks. The paper should present this as a nuanced, secondary contribution rather than claim self-distillation is unambiguously necessary — as currently implied only by its presence in the config table with no dedicated conclusion.

### 6. Bidirectional Fusion and Orthogonal Regularization Are Partially Redundant, Not Additive
The 2×2-style grid in §2.6 (bidirectional vs. unidirectional × with/without orthogonal loss) reveals an interaction effect the original draft's single `uni_a2r_no_ortho` row could not show:
- **`no_ortho` (bidirectional, no ortho)** is the single best missing-modality DepthTrack configuration in the entire study (F-score 60.05, +2.17 over baseline, well outside the ±1.66 seed-noise band) and also improves LasHeR PR/SR and RGBT234_m. Removing the orthogonal constraint *while keeping bidirectional fusion* appears to help, not hurt.
- **`uni_a2r` (unidirectional, +ortho)** trades DepthTrack full-modality gains (66.53 vs. 67.01, within noise) for a DepthTrack_m improvement (59.53 vs. 57.88, outside noise) — unidirectional fusion alone is not clearly worse than bidirectional.
- **`uni_a2r_no_ortho` (unidirectional, no ortho)** — the configuration named in the original config table but never actually tabulated — underperforms both `no_ortho` and `uni_a2r` on nearly every metric, and is the weakest of the four variants on DepthTrack (F-score 65.03/57.50).
- **`uni_r2a` (reverse direction, +ortho)** sits between `uni_a2r` and baseline, suggesting direction of fusion (A→R vs. R→A) matters less than whether fusion is bidirectional at all.
**Conclusion:** Orthogonal regularization and bidirectional fusion are not independent, additive mechanisms — their combination in the baseline is not the best on every axis, and removing orthogonality from the *bidirectional* model (`no_ortho`) is actually the strongest missing-modality result in the study. This is a more interesting and more publishable finding than "unidirectional + no-ortho hurts a bit," and the paper should lead with the `no_ortho` result, using `uni_a2r`/`uni_a2r_no_ortho`/`uni_r2a` as supporting evidence for the interaction rather than as the headline comparison.

### 7. Hallucination and Reconstruction Losses Show the Same Ortho-Interaction Pattern
Mirroring §6, removing hallucination or reconstruction losses interacts with orthogonality rather than acting independently (§2.7):
- `no_hallucinate` alone is close to baseline everywhere (largest single delta: DepthTrack F-score -1.71, within the ±1.66 noise band).
- `no_hallucinate_no_ortho` produces the best DepthTrack full-modality F-score in the *entire* ablation study (67.86, exceeding even baseline), while `no_hallucinate` alone does not — the effect only appears once orthogonality is also removed.
- `no_recon_loss` alone shows the clearest isolated regression: DepthTrack_m F-score drops to 56.01 (-1.87, outside noise), the second-worst missing-modality DepthTrack result after `moe_small`.
- `no_recon_loss_no_ortho` partially recovers this (57.36), again showing an ortho-removal interaction rather than independent effects.
**Conclusion:** The reconstruction loss is the more load-bearing of the two auxiliary losses for missing-modality depth performance when orthogonality is present; hallucination's contribution is largely noise-level except in combination with removing orthogonality. Given the recurring pattern across §6 and §7, we recommend the paper explicitly test and report an orthogonal-loss main effect / interaction analysis (e.g., a small factorial ANOVA over {ortho, direction, hallucinate, recon} × {dataset}) rather than presenting each ablation as independent — the current per-row framing understates a structural finding.

### 8. Seed Variance Sets the Bar for "Significant"
Three same-config reseeded runs (`rung0`, `rung0_seedB`, `rung0_seedC`, §2.1) show that DepthTrack full-modality F-score has a natural std of **±1.66** and DepthTrack_m has **±0.46** — both far larger than the other benchmarks (std ≤0.24 for RGBT234/LasHeR/VisEvent full-modality metrics, ≤0.44 for their `_miss` counterparts). Any DepthTrack claim in the paper with a delta under ~1.7 (e.g., `moe_small` vs. baseline full-modality F-score, 67.23 vs. 67.01) should be described as "no significant difference," while deltas over ~3× the std (e.g., `no_ortho` DepthTrack_m +2.17, `no_hallucinate_no_ortho` DepthTrack +0.85 is *not* significant but its 67.86 full-modality peak still leads the sweep) are safe to report as real effects.
**Recommendation:** Report DepthTrack results with the seed std as an explicit error bar or footnote in the camera-ready table, and avoid citing single-run DepthTrack deltas below ~2 points as evidence for any mechanism.

---

## 4. Outstanding Gaps Before Submission
- **VOT22RGBD** is absent (`null`) from every ablation's metrics — either backfill this benchmark for at least the baseline and top-3 ablations, or explicitly scope it out of the ablation study in the paper text.
- **`moe_big`** metrics are on a different, older metric schema (AUC/PR, no LasHeR) and cannot be placed in the §2.2 table on equal footing — re-run under the current protocol if it's meant to close the capacity sweep at the top end.
- No error bars exist for any non-baseline ablation (only `rung0` was reseeded). If reviewers push on significance, the cheapest fix is reseeding the 2–3 headline ablations (`no_ortho`, `no_distill`, `pmax_050`) rather than all ~20 configs.
