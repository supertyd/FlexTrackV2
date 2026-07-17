import os
import subprocess
import yaml
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

def evaluate_tracker_dataset(res_dir, gt_dir, seqs, gt_filename):
    iou_list = []
    distance_list = []
    
    for seq in seqs:
        pred_path = os.path.join(res_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, gt_filename)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        try:
            pred_boxes = load_box_file(pred_path)
            gt_boxes = load_box_file(gt_path)
            if len(pred_boxes.shape) == 1:
                pred_boxes = pred_boxes[None, :]
            if len(gt_boxes.shape) == 1:
                gt_boxes = gt_boxes[None, :]
                
            n_frames = min(len(pred_boxes), len(gt_boxes))
            for idx in range(n_frames):
                if gt_boxes[idx][2] <= 0 or gt_boxes[idx][3] <= 0 or np.isnan(gt_boxes[idx]).any():
                    continue
                iou = compute_iou(pred_boxes[idx], gt_boxes[idx])
                dist = compute_center_distance(pred_boxes[idx], gt_boxes[idx])
                iou_list.append(iou)
                distance_list.append(dist)
        except Exception:
            pass
            
    if len(iou_list) == 0:
        return 0.0, 0.0
        
    thresholds = np.linspace(0.0, 1.0, 100)
    success_rates = [(np.array(iou_list) >= t).mean() for t in thresholds]
    auc = np.mean(success_rates) * 100.0
    pr = (np.array(distance_list) <= 20.0).mean() * 100.0
    
    return auc, pr

def update_yaml_test_params(yaml_path, dataset_name, upt, uph, inter):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    db_key = dataset_name.upper()
    cfg["TEST"]["UPT"][db_key] = float(upt)
    cfg["TEST"]["UPH"][db_key] = float(uph)
    cfg["TEST"]["INTER"][db_key] = int(inter)
    
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

def main():
    yaml_path = "/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_54.yaml"
    
    # Evaluate Candidate 1 of VisEvent synchronously and print its full output!
    visevent_gt_dir = "/mnt/task_wrapper/user_output/artifacts/visevent/test"
    with open(os.path.join(visevent_gt_dir, "testlist.txt"), "r") as f:
        visevent_seqs = sorted(f.read().splitlines())
        
    update_yaml_test_params(yaml_path, "VisEvent", 0.80, 0.90, 50)
    subprocess.run("rm -rf /mnt/task_runtime/workspace/results/VisEvent/flextrackv2_b224_54", shell=True)
    
    cmd = "cd /mnt/task_runtime && export PYTHONPATH=/mnt/task_runtime:\\$PYTHONPATH && /coreflow/venv/bin/python RGBE_workspace/test_rgbe_mgpus.py --script_name flextrackv2 --dataset_name VisEvent --yaml_name flextrackv2_b224_54 --threads 24 --num_gpus 8 --epoch 40"
    print("Running parallel tracking for VisEvent Candidate 1...")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print("Subprocess failed!")
        
    auc, pr = evaluate_tracker_dataset("/mnt/task_runtime/workspace/results/VisEvent/flextrackv2_b224_54", visevent_gt_dir, visevent_seqs, "groundtruth.txt")
    print(f"VisEvent Candidate 1: Success (AUC) = {auc:.2f}%, Precision = {pr:.2f}%")

if __name__ == "__main__":
    main()
