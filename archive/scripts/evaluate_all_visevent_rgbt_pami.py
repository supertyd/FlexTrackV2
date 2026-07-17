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
    if "," in line:
        return np.loadtxt(path, delimiter=",")
    else:
        return np.loadtxt(path)

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
            if len(pred_boxes.shape) == 1:
                pred_boxes = pred_boxes[None, :]
            if len(gt_boxes.shape) == 1:
                gt_boxes = gt_boxes[None, :]
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

def main():
    datasets = {
        "VisEvent": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/visevent/test",
            "results_dir": "/mnt/task_runtime/workspace/results/VisEvent/flextrackv2_b224_52",
            "gt_filename": "groundtruth.txt"
        },
        "VisEvent_miss": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/visevent/test",
            "results_dir": "/mnt/task_runtime/workspace/results/VisEvent_miss/flextrackv2_b224_52",
            "gt_filename": "groundtruth.txt"
        },
        "LasHeR": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/lasher/testingset",
            "results_dir": "/mnt/task_runtime/workspace/results/LasHeR/flextrackv2_b224_52",
            "gt_filename": "init.txt"
        },
        "LasHeR_miss": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/lasher/testingset",
            "results_dir": "/mnt/task_runtime/workspace/results/LasHeR_miss/flextrackv2_b224_52",
            "gt_filename": "init.txt"
        },
        "RGBT234": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/rgbt234",
            "results_dir": "/mnt/task_runtime/workspace/results/RGBT234/flextrackv2_b224_52",
            "gt_filename": "visible.txt"
        },
        "RGBT234_miss": {
            "gt_dir": "/mnt/task_wrapper/user_output/artifacts/rgbt234",
            "results_dir": "/mnt/task_runtime/workspace/results/RGBT234_miss/flextrackv2_b224_52",
            "gt_filename": "visible.txt"
        }
    }

    print("==========================================================================================")
    print("                 FlexTrackV2 V52 (BMR-HMoE) IEEE TPAMI Benchmark Evaluation                 ")
    print("==========================================================================================")

    for name, paths in datasets.items():
        gt_dir = paths["gt_dir"]
        results_dir = paths["results_dir"]
        gt_filename = paths["gt_filename"]
        
        # Load sequences list
        seqs = []
        if "VisEvent" in name:
            testlist_path = os.path.join(gt_dir, "testlist.txt")
            if os.path.exists(testlist_path):
                with open(testlist_path, "r") as f:
                    seqs = f.read().splitlines()
        elif "LasHeR" in name:
            if os.path.exists(gt_dir):
                seqs = [f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))]
        elif "RGBT234" in name:
            if os.path.exists(gt_dir):
                seqs = [f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))]
                
        seqs = sorted(list(set(seqs)))
        if len(seqs) == 0:
            print(f"Dataset {name:15} | Seqs: {len(seqs):3} | No sequences found.")
            continue
            
        auc, pr, npr = evaluate_dataset(results_dir, gt_dir, seqs, gt_filename)
        print(f"Dataset {name:15} | Seqs: {len(seqs):3} | Success (AUC): {auc:5.2f}% | Precision (PR): {pr:5.2f}% | Norm Precision (NPR): {npr:5.2f}%")

    print("==========================================================================================")

if __name__ == "__main__":
    main()
