# FlexTrackV2 vs FlexTrack (V1) — Qualitative Overlap Comparison

Per-frame overlap (IoU) curves + sampled frames, in the style of the reference figure.
Each figure plots **both** curves in one axis: FlexTrackV2 (red, ours) vs FlexTrack V1 (blue).
Boxes on thumbnails: **green = GT**, **red = FlexTrackV2**, **blue = FlexTrack V1**.
The 4 colored dashed lines (blue/black/red/green) mark the 4 sampled frames shown above.

### Missing-modality panels (`*_miss`)
The `_miss` figures add a **modality-availability ribbon** below the overlap curve, coloured
per frame from the official missing-modality masks (`[rgb_present, aux_present]`):
**both present** · **aux dropped** (Thermal/Event/Depth) · **RGB dropped** · **both dropped**.
For these panels the 4 sampled frames are chosen at dropout frames where V1 has lost the target
(IoU&lt;0.3) while V2 holds it (IoU&gt;0.5) — the thumbnails show the red (V2) box staying on the
target while the blue (V1) box drifts to a distractor, exactly at the annotated dropouts.

## Method mapping
- **FlexTrackV2 (Ours)** = `final_result/raw_predictions/*` (flextrackv2_b224_56 — V54 checkpoint + V56 tuned test-time params; config `final_result/configs/flextrackv2_b224_56_best_full.yaml`)
- **FlexTrack V1** = `data_missing_modality/Missing_data_annotation/FlexTrack/*` (V1 predictions co-located with the missing-modality annotations)

> Note: `RGBT_workspace/results/LasHeR/flextrackv2_b224_sota` is byte-identical to V1 (a copy) — not used.

## Chosen sequences (V2 clearly wins, V1 clearly fails), mean overlap over present frames
| Figure | Sequence | FlexTrackV2 | V1 | gap | frames |
|---|---|---|---|---|---|
| LasHeR_full | drillmaster1117 | 77.1 | 38.2 | +38.9 | 1117 |
| LasHeR_miss | 4men | 61.7 | 17.9 | +43.8 | 414 |
| DepthTrack_full | notebook01_indoor | 62.9 | 16.4 | +46.6 | 2000 |
| DepthTrack_miss | mobilephone03_indoor | 74.3 | 48.6 | +25.7 | 677 |
| VisEvent_full | dvSave-2021_02_14_16_37_15_car5 | 81.6 | 16.4 | +65.1 | 103 |
| VisEvent_miss | dvSave-2021_02_14_16_46_34_car8 | 81.2 | 15.9 | +65.3 | 101 |

Ground truth: LasHeR `visible.txt`, DepthTrack `groundtruth.txt` (absent/NaN frames excluded),
VisEvent `test/<seq>/groundtruth.txt`. DepthTrack V1 is VOT-longterm format (`<seq>_001.txt`,
init line "1" filled with GT).

## Files
- `LasHeR_full.png`, `LasHeR_miss.png`, `DepthTrack_full.png`, `DepthTrack_miss.png`, `VisEvent_full.png`, `VisEvent_miss.png`
- `ALL_six_stacked.png` — all six stacked for quick review

Regenerate: `scratchpad/make_fig.py` (helper `scratchpad/loaders.py`).
