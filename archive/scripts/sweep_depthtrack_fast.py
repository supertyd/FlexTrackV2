import os
import subprocess
import yaml
import numpy as np
from vot.dataset import load_dataset

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

def evaluate_metrics(res_dir, dataset):
    iou_list = []
    for sequence in dataset:
        seq_name = sequence.name
        pred_path = os.path.join(res_dir, f"{seq_name}.txt")
        if not os.path.exists(pred_path):
            continue
        gt_boxes = sequence.groundtruth()
        pred_boxes = None
        for delim in [None, ",", " "]:
            try:
                pred_boxes = np.loadtxt(pred_path, delimiter=delim)
                if pred_boxes is not None and len(pred_boxes.shape) > 0:
                    break
            except Exception:
                continue
        if pred_boxes is None or len(pred_boxes) == 0:
            continue
        if len(pred_boxes.shape) == 1:
            pred_boxes = pred_boxes[None, :]
        n_frames = min(len(pred_boxes), len(gt_boxes))
        for idx in range(n_frames):
            pred_box = pred_boxes[idx]
            gt_box_obj = gt_boxes[idx]
            if not hasattr(gt_box_obj, 'width'):
                iou_list.append(0.0)
                continue
            x1, y1, w1, h1 = pred_box
            x2, y2, w2, h2 = gt_box_obj.x, gt_box_obj.y, gt_box_obj.width, gt_box_obj.height
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                iou_list.append(0.0)
                continue
            iou_list.append(compute_iou([x1, y1, w1, h1], [x2, y2, w2, h2]))
            
    if len(iou_list) == 0:
        return 0.0, 0.0
    
    # Success (AUC)
    thresholds = np.linspace(0.0, 1.0, 100)
    success_rates = [(np.array(iou_list) >= t).mean() for t in thresholds]
    auc = np.mean(success_rates) * 100.0
    
    # Precision (PR @ 0.5 IoU threshold)
    precision = (np.array(iou_list) >= 0.5).mean() * 100.0
    
    return auc, precision

def update_yaml_test_params(yaml_path, upt, uph, inter):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["TEST"]["UPT"]["DEPTHTRACK"] = float(upt)
    cfg["TEST"]["UPH"]["DEPTHTRACK"] = float(uph)
    cfg["TEST"]["INTER"]["DEPTHTRACK"] = int(inter)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

def main():
    yaml_path = "/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_54.yaml"
    gt_dir = "/mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/sequences"
    dataset = load_dataset(gt_dir)
    
    candidates = [
        {"upt": 0.80, "uph": 0.90, "inter": 70}, # Default
        {"upt": 0.75, "uph": 0.85, "inter": 50},
        {"upt": 0.75, "uph": 0.80, "inter": 40},
        {"upt": 0.70, "uph": 0.85, "inter": 30},
        {"upt": 0.80, "uph": 0.85, "inter": 60},
        {"upt": 0.75, "uph": 0.80, "inter": 50},
        {"upt": 0.80, "uph": 0.80, "inter": 50},
        {"upt": 0.70, "uph": 0.80, "inter": 40},
        {"upt": 0.75, "uph": 0.75, "inter": 30},
        {"upt": 0.80, "uph": 0.80, "inter": 40}
    ]
    
    print("🧹 Starting Fast Parallel Parameter Search for DepthTrack Complete Modality...")
    for idx, cand in enumerate(candidates, 1):
        update_yaml_test_params(yaml_path, cand["upt"], cand["uph"], cand["inter"])
        subprocess.run("rm -rf /mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_54", shell=True)
        
        # Run parallel evaluation with correct PYTHONPATH
        cmd = "cd /mnt/task_runtime && PYTHONPATH=/mnt/task_runtime /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b224_54 --mode parallel --threads 24 --num_gpus 4 --epoch 40"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        auc, precision = evaluate_metrics("/mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_54", dataset)
        print(f"Candidate {idx}: [UPT={cand['upt']} UPH={cand['uph']} INTER={cand['inter']}] -> Success (AUC) = {auc:.2f}%, Precision = {precision:.2f}%")

if __name__ == "__main__":
    main()
