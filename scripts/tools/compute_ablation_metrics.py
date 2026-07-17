"""
Computes final metrics for one ablation config across the 6 documented
datasets (RGBT234/_miss, VisEvent/_miss via box-overlap AUC/PR; DepthTrack/_miss
via the official VOT toolkit Pr/Re/F) and writes ablation_results/<config>/metrics.json.

Reuses the same box-overlap AUC/PR formulas already used for the V56 baseline
in evaluate_all_visevent_rgbt_pami_v56.py, generalized by config/tracker name
instead of being hardcoded to "flextrackv2_b224_56".

Usage: python compute_ablation_metrics.py --config <config_name>
"""
import argparse
import json
import os
import subprocess

import numpy as np


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_center_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
    cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def load_box_file(path):
    with open(path, "r") as f:
        line = f.readline().strip()
    return np.loadtxt(path, delimiter=",") if "," in line else np.loadtxt(path)


def evaluate_dataset(results_dir, gt_dir, seqs, gt_filename):
    iou_list, distance_list = [], []
    for seq in seqs:
        pred_path = os.path.join(results_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, gt_filename)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        try:
            pred_boxes = load_box_file(pred_path)
            gt_boxes = load_box_file(gt_path)
            if pred_boxes.ndim == 1:
                pred_boxes = pred_boxes[None, :]
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes[None, :]
            n_frames = min(len(pred_boxes), len(gt_boxes))
            for idx in range(n_frames):
                iou_list.append(compute_iou(pred_boxes[idx], gt_boxes[idx]))
                distance_list.append(compute_center_distance(pred_boxes[idx], gt_boxes[idx]))
        except Exception:
            continue
    if not iou_list:
        return None
    thresholds = np.linspace(0.0, 1.0, 100)
    auc = float(np.mean([(np.array(iou_list) >= t).mean() for t in thresholds]) * 100.0)
    pr = float((np.array(distance_list) <= 20.0).mean() * 100.0)
    return {"auc": round(auc, 2), "pr": round(pr, 2)}


BOX_DATASETS = {
    "VisEvent": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test",
        "gt_filename": "groundtruth.txt",
    },
    "VisEvent_miss": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test",
        "gt_filename": "groundtruth.txt",
    },
    "RGBT234": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234",
        "gt_filename": "visible.txt",
    },
    "RGBT234_miss": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234",
        "gt_filename": "visible.txt",
    },
    "LasHeR": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset",
        "gt_filename": "visible.txt",
    },
    "LasHeR_miss": {
        "gt_dir": "/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset",
        "gt_filename": "visible.txt",
    },
}


def eval_box_dataset(dataset_name, config_name):
    spec = BOX_DATASETS[dataset_name]
    results_dir = f"/mnt/task_runtime/workspace/results/{dataset_name}/{config_name}"
    gt_dir = spec["gt_dir"]
    if not os.path.isdir(gt_dir):
        return None
    seqs = sorted(
        f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))
    )
    if not os.path.isdir(results_dir):
        return None
    return evaluate_dataset(results_dir, gt_dir, seqs, spec["gt_filename"])


def eval_vot(tracker_label, workspace):
    """Run vot analysis in `workspace` for `tracker_label` and return Pr/Re/F.
    Works for both DepthTrack (Depthtrack_workspace, votrgbd2021 stack) and
    VOT22RGBD (VOT22RGBD_workspace, vot2022/rgbd stack)."""
    try:
        subprocess.run(
            ["vot", "analysis", "--nocache", "--format", "json", "--workspace", workspace, tracker_label],
            capture_output=True, text=True, timeout=3600,
        )
    except Exception as e:
        return {"error": str(e)}

    import glob
    candidates = sorted(glob.glob(f"{workspace}/analysis/*/results.json"), key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            data = json.load(open(path))
            if tracker_label in data.get("trackers", {}):
                exp = list(data["results"].values())[0]
                pr = exp["results"][0][0]
                if pr:
                    return {"precision": round(pr[0] * 100, 2), "recall": round(pr[1] * 100, 2), "fscore": round(pr[2] * 100, 2)}
        except Exception:
            continue
    return None


def eval_depthtrack(tracker_label):
    return eval_vot(tracker_label, "/mnt/task_runtime/Depthtrack_workspace")


def eval_vot22rgbd(tracker_label):
    """VOT22RGBD uses the MultiStart protocol (stack vot2022/rgbd): the headline
    metric is EAO, plus Accuracy and Robustness — NOT the Pr/Re/F of the
    DepthTrack unsupervised protocol. Results layout:
      results[0] = [[EAO]]; results[2] = [[accuracy, robustness, ...]]"""
    workspace = "/mnt/task_runtime/VOT22RGBD_workspace"
    try:
        subprocess.run(
            ["vot", "analysis", "--nocache", "--format", "json", "--workspace", workspace, tracker_label],
            capture_output=True, text=True, timeout=3600,
        )
    except Exception as e:
        return {"error": str(e)}
    import glob
    candidates = sorted(glob.glob(f"{workspace}/analysis/*/results.json"), key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            data = json.load(open(path))
            if tracker_label not in data.get("trackers", {}):
                continue
            exp = list(data["results"].values())[0]
            res = exp["results"]
            eao = res[0][0][0]
            ar = res[2][0]
            return {"eao": round(eao * 100, 2),
                    "accuracy": round(ar[0] * 100, 2),
                    "robustness": round(ar[1] * 100, 2)}
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of keys to (re)compute; others preserved. "
                             "e.g. --only VOT22RGBD for the local-only VOT step.")
    args = parser.parse_args()

    out_dir = f"/mnt/task_runtime/ablation_results/{args.config}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")

    # Merge into any existing metrics so a partial re-run (e.g. the local-only
    # VOT22RGBD step) never clobbers fields computed elsewhere.
    metrics = {"config": args.config}
    if os.path.exists(out_path):
        try:
            metrics.update(json.load(open(out_path)))
        except Exception:
            pass
    metrics["config"] = args.config

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    def want(k):
        return only is None or k in only

    for ds in ["RGBT234", "RGBT234_miss", "LasHeR", "LasHeR_miss", "VisEvent", "VisEvent_miss"]:
        if want(ds):
            metrics[ds] = eval_box_dataset(ds, args.config)
    if want("DepthTrack"):
        metrics["DepthTrack"] = eval_depthtrack(f"{args.config}_full")
    if want("DepthTrack_miss"):
        metrics["DepthTrack_miss"] = eval_depthtrack(f"{args.config}_miss")
    if want("VOT22RGBD"):
        metrics["VOT22RGBD"] = eval_vot22rgbd(f"{args.config}_vot22")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
