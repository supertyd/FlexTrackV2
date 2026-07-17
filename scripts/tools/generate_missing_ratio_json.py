"""
Generate synthetic missing-modality annotations at controlled global missing
RATIOS (0/25/50/75/100% of frames with the auxiliary modality zeroed), as an
alternative to the single fixed official "_miss" pattern -- for plotting
performance-vs-missing-rate curves instead of one point.

Convention, matching test_rgbt_mgpus.py / test_depthtrack_mgpus.py:
  data format is per-frame [rgb_present, aux_present] (frame 0 always [1,1],
  matching the official pattern -- init frame is never missing).
  RGB stays present throughout; the AUX modality (infrared/event/depth) is
  independently zeroed each frame with probability = ratio. This isolates the
  "reconstruct aux from rgb" direction, which is what the BMR mechanism
  ablations (no_hallucinate, uni_a2r/uni_r2a, no_ortho, no_recon_loss) are
  actually about.

Sequence lists + frame counts are read from each dataset's existing official
missing_results_*.json (frame counts identical across ratios/reused as-is)
so this does not need direct access to each dataset's raw image folders.

Output: /mnt/task_runtime/data_missing_modality/synthetic_ratio/<name>_missR<ratio>.json
  where <name> in {rgbt234, lasher245... no -- must match dataset_name.lower()
  with the "_miss" suffix removed by the caller, e.g. "rgbt234", "lasher",
  "visevent", "depthtrack"} and <ratio> in {000,025,050,075,100}.

Usage: python3 generate_missing_ratio_json.py
"""
import json
import os
import random

RATIOS = [0, 25, 50, 75, 100]
OUT_DIR = "/mnt/task_runtime/data_missing_modality/synthetic_ratio"

SOURCES = {
    "rgbt234": "/mnt/task_runtime/data_missing_modality/Missing_data_annotation/RGBT234-Miss/missing_results_rgbt234.json",
    "lasher": "/mnt/task_runtime/data_missing_modality/Missing_data_annotation/LasHeR245-Miss/missing_results_lasher245.json",
    "visevent": "/mnt/task_runtime/data_missing_modality/Missing_data_annotation/visevent-miss/missing_results_visevent.json",
    "depthtrack": "/mnt/task_runtime/data_missing_modality/Missing_data_annotation/depthtrack-miss/missing_results_depthtrack.json",
}


def build_ratio_annotation(source_data, ratio_pct, seed):
    rng = random.Random(seed)
    out = {}
    for seq, entry in source_data.items():
        n = entry["frames"]
        data = [[1.0, 1.0]]  # frame 0 never missing
        for _ in range(1, n):
            aux_present = 0.0 if rng.random() < (ratio_pct / 100.0) else 1.0
            data.append([1.0, aux_present])
        out[seq] = {"frames": n, "data": data}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            print(f"SKIP {name}: source not found at {src_path}")
            continue
        source_data = json.load(open(src_path))
        for ratio in RATIOS:
            seed = hash((name, ratio)) & 0xFFFFFFFF
            ann = build_ratio_annotation(source_data, ratio, seed)
            out_path = os.path.join(OUT_DIR, f"{name}_missR{ratio:03d}.json")
            json.dump(ann, open(out_path, "w"))
            n_seqs = len(ann)
            total_frames = sum(v["frames"] for v in ann.values())
            missing_frames = sum(
                1 for v in ann.values() for f in v["data"] if f[1] == 0.0
            )
            actual_rate = missing_frames / total_frames * 100 if total_frames else 0
            print(f"{name} R{ratio:03d}: {n_seqs} seqs, {total_frames} frames, "
                  f"actual aux-missing rate={actual_rate:.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
