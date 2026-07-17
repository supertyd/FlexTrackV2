import os
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
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
        
    try:
        if "," in line:
            data = np.loadtxt(path, delimiter=",")
        else:
            data = np.loadtxt(path)
    except:
        # Handle some edge cases in txt format where first line might be weird
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) > 0 and len(lines[0].strip()) < 5:
            # Skip first line if it's just '1' or similar
            lines = lines[1:]
            
        data = []
        for l in lines:
            if "," in l:
                data.append([float(x) for x in l.strip().split(",")])
            else:
                data.append([float(x) for x in l.strip().split()])
        data = np.array(data)
        
    # Standardize to (N, 4)
    if len(data.shape) == 1 and len(data) == 4:
        data = data[None, :]
    elif len(data.shape) == 1 and len(data) > 4:
        # Flattened array? reshape
        data = data.reshape(-1, 4)
        
    return data

def evaluate_dataset(results_dir, gt_dir, seqs, gt_filename):
    iou_list = []
    distance_list = []
    norm_distance_list = []
    
    for seq in seqs:
        pred_path = os.path.join(results_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, gt_filename)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        try:
            pred_boxes = load_box_file(pred_path)
            gt_boxes = load_box_file(gt_path)
            
            n_frames = min(len(pred_boxes), len(gt_boxes))
            for idx in range(n_frames):
                iou = compute_iou(pred_boxes[idx], gt_boxes[idx])
                dist = compute_center_distance(pred_boxes[idx], gt_boxes[idx])
                iou_list.append(iou)
                distance_list.append(dist)
                
                # Normalize distance by groundtruth diagonal size
                gt_w, gt_h = gt_boxes[idx][2], gt_boxes[idx][3]
                diagonal = np.sqrt(gt_w ** 2 + gt_h ** 2)
                if diagonal > 0:
                    norm_distance_list.append(dist / diagonal)
                else:
                    norm_distance_list.append(dist)
        except Exception as e:
            pass
            
    if len(iou_list) == 0:
        return 0.0, 0.0, 0.0
        
    # Overlap Success Score (OPE Success AUC)
    thresholds = np.linspace(0.0, 1.0, 100)
    success_rates = [(np.array(iou_list) >= t).mean() for t in thresholds]
    auc = np.mean(success_rates) * 100.0
    
    # Distance Precision Score (OPE Precision @ 20px)
    pr = (np.array(distance_list) <= 20.0).mean() * 100.0
    
    # Normalized Precision Score (OPE Norm Precision @ 0.2)
    npr = (np.array(norm_distance_list) <= 0.2).mean() * 100.0
    
    return auc, pr, npr

def print_result(tracker, auc, pr, npr):
    print(f"{tracker:20} | Success (AUC): {auc:5.2f}% | Precision (PR): {pr:5.2f}% | Norm Precision (NPR): {npr:5.2f}%")

def main():
    datasets = {
        "DepthTrack": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/Depthtrack_workspace/sequences",
            "mci_res": "/mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_56",
            "flex_res": "/mnt/task_runtime/data_missing_modality/data_missing_modality/FlexTrack/depthtrack",
            "gt_filename": "groundtruth.txt"
        },
        "DepthTrack_miss": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/Depthtrack_workspace/sequences",
            "mci_res": "/mnt/task_runtime/workspace/results/depthtrack_miss/flextrackv2_b224_56",
            "flex_res": "/mnt/task_runtime/data_missing_modality/data_missing_modality/FlexTrack/depthtrack_miss",
            "gt_filename": "groundtruth.txt"
        }
    }

    print("==========================================================================================")
    print("                 FlexTrackV2 V56 vs FlexTrack (AUC/PR Eval)                 ")
    print("==========================================================================================")

    for name, paths in datasets.items():
        gt_dir = paths["gt_dir"]
        mci_dir = paths["mci_res"]
        flex_dir = paths["flex_res"]
        gt_filename = paths["gt_filename"]
        
        seqs = []
        if os.path.exists(gt_dir):
            seqs = [f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f)) and not f.startswith(".")]
                
        seqs = sorted(list(set(seqs)))
        if len(seqs) == 0:
            print(f"Dataset {name} | No sequences found.")
            continue
            
        print(f"\n[{name}] - Seqs: {len(seqs)}")
        
        mci_auc, mci_pr, mci_npr = evaluate_dataset(mci_dir, gt_dir, seqs, gt_filename)
        print_result("FlexTrackV2 V56", mci_auc, mci_pr, mci_npr)
        
        flex_auc, flex_pr, flex_npr = evaluate_dataset(flex_dir, gt_dir, seqs, gt_filename)
        print_result("FlexTrack", flex_auc, flex_pr, flex_npr)

if __name__ == "__main__":
    main()
