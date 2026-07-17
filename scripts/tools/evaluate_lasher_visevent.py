import os
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_center_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
    cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

def load_box_file(path):
    with open(path, "r") as f:
        line = f.readline().strip()
    if "," in line:
        return np.loadtxt(path, delimiter=",")
    else:
        return np.loadtxt(path)

def evaluate_tracker_dataset(res_dir, gt_dir, seqs, gt_filename, is_visevent=False):
    # Thresholds match the official VisEvent_SOT_Benchmark toolkit
    # (Evaluate_VisEvent_SOT_benchmark.m / utils/eval_tracker.m):
    # threshold_set_overlap = 0:0.05:1 (21 points), precision @ 20px.
    thresholds = np.linspace(0.0, 1.0, 21)

    seq_success_rates = []  # per-sequence vector of success-rate(threshold), len 21
    seq_precisions = []     # per-sequence precision @ 20px

    for seq in seqs:
        pred_path = os.path.join(res_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, gt_filename)

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print('Skipping', seq, 'pred_path:', pred_path, 'exists:', os.path.exists(pred_path), 'gt_path:', gt_path, 'exists:', os.path.exists(gt_path))
            continue
        try:
            pred_boxes = load_box_file(pred_path)
            gt_boxes = load_box_file(gt_path)
            if len(pred_boxes.shape) == 1:
                pred_boxes = pred_boxes[None, :]
            if len(gt_boxes.shape) == 1:
                gt_boxes = gt_boxes[None, :]

            n_frames = min(len(pred_boxes), len(gt_boxes))
            pred_boxes = pred_boxes[:n_frames].copy()
            gt_boxes = gt_boxes[:n_frames].copy()

            # Robust invalid prediction repair (MATLAB calc_seq_err_robust.m-aligned)
            for idx in range(1, n_frames):
                r = pred_boxes[idx]
                if np.isnan(r).any() or r[2] <= 0 or r[3] <= 0:
                    pred_boxes[idx] = pred_boxes[idx-1]

            # First-Frame Override (MATLAB-aligned)
            pred_boxes[0] = gt_boxes[0]

            # A frame is "target absent" iff the GT box itself is invalid
            # (VisEvent stores absent frames as gt = [0,0,0,0]; this is the
            # authoritative signal used by the official toolkit's absent/*.txt,
            # NOT the local absent_label.txt shipped with the dataset copy,
            # whose 0/1 polarity is inverted relative to the official one).
            gt_valid = (~np.isnan(gt_boxes).any(axis=1)) & (gt_boxes[:, 2] > 0) & (gt_boxes[:, 3] > 0)

            iou_seq = np.zeros(n_frames)
            dist_seq = np.full(n_frames, np.inf)
            for idx in np.where(gt_valid)[0]:
                iou_seq[idx] = compute_iou(pred_boxes[idx], gt_boxes[idx])
                dist_seq[idx] = compute_center_distance(pred_boxes[idx], gt_boxes[idx])

            # Absent frames are NOT dropped from the sequence: they stay in the
            # denominator (len_all) and simply score 0 success / fail precision,
            # exactly like utils/eval_tracker.m does
            # (`ave_success_rate_plot(k,i,:) = success_num_overlap/(len_all+eps)`
            # where len_all = size(anno,1) is the FULL sequence length, while
            # calc_seq_err_robust.m only strips absent rows from the numerator).
            success_rate = np.array([(iou_seq >= t).mean() for t in thresholds])
            precision = (dist_seq <= 20.0).mean() * 100.0

            seq_success_rates.append(success_rate)
            seq_precisions.append(precision)
        except Exception as e:
            print('Error on sequence', seq, ':', e)
            pass

    if len(seq_success_rates) == 0:
        return 0.0, 0.0

    # Official protocol averages per-sequence curves (equal weight per video),
    # not frames pooled across the whole dataset (which over-weights long
    # sequences and was the other source of the discrepancy).
    auc = np.mean([sr.mean() for sr in seq_success_rates]) * 100.0
    pr = np.mean(seq_precisions)

    return auc, pr

def main():
    datasets = {
        "VisEvent": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/visevent/test",
            "gt_filename": "groundtruth.txt",
            "v54_dir": "/mnt/task_runtime/workspace/results/VisEvent/flextrackv2_b224_54",
            "flex_dir": "/mnt/task_runtime/data_missing_modality/FlexTrack/VisEvent",
            "is_visevent": True
        },
        "LasHeR": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/lasher/testingset",
            "gt_filename": "visible.txt",
            "v54_dir": "/mnt/task_runtime/workspace/results/LasHeR/flextrackv2_b224_54",
            "flex_dir": "/mnt/task_runtime/data_missing_modality/FlexTrack/LasHER",
            "is_visevent": False
        },
        "RGBT234": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/rgbt234",
            "gt_filename": "visible.txt",
            "v54_dir": "/mnt/task_runtime/workspace/results/RGBT234/flextrackv2_b224_54",
            "flex_dir": "/mnt/task_runtime/data_missing_modality/FlexTrack/rgbt234",
            "is_visevent": False
        }
    }

    print("==========================================================================")
    print("        FlexTrackV2 V54 vs FlexTrack (ICCV 2025 SOTA) Evaluation             ")
    print("==========================================================================")

    for name, paths in datasets.items():
        gt_dir = paths["gt_dir"]
        gt_filename = paths["gt_filename"]
        is_visevent = paths["is_visevent"]
        
        seqs = []
        if "VisEvent" in name:
            testlist_path = os.path.join(gt_dir, "testlist.txt")
            if os.path.exists(testlist_path):
                with open(testlist_path, "r") as f:
                    seqs = f.read().splitlines()
        else:
            if os.path.exists(gt_dir):
                seqs = [f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))]
                
        seqs = sorted(list(set(seqs)))
        if len(seqs) == 0:
            continue
            
        print(f"\n--- Evaluating {name} on {len(seqs)} sequences ---")
        v54_auc, v54_pr = evaluate_tracker_dataset(paths["v54_dir"], gt_dir, seqs, gt_filename, is_visevent)
        flex_auc, flex_pr = evaluate_tracker_dataset(paths["flex_dir"], gt_dir, seqs, gt_filename, is_visevent)
        
        print(f"🚀 FlexTrackV2 V54       | Success (AUC): {v54_auc:.2f}% | Precision (PR): {v54_pr:.2f}%")
        print(f"🏆 FlexTrack (SOTA)   | Success (AUC): {flex_auc:.2f}% | Precision (PR): {flex_pr:.2f}%")
        print(f"📊 Absolute Gain      | Success: {v54_auc - flex_auc:+.2f}% | Precision: {v54_pr - flex_pr:+.2f}%")

if __name__ == "__main__":
    main()
