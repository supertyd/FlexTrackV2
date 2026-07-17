import os
import numpy as np

def compute_iou(box1, box2):
    # box format: [x, y, w, h]
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

def evaluate_tracker(results_dir, gt_dir, seqs):
    iou_list = []
    distance_list = []
    
    for seq in seqs:
        pred_path = os.path.join(results_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, "groundtruth.txt")
        
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
            
        pred_boxes = np.loadtxt(pred_path, delimiter=',')
        gt_boxes = np.loadtxt(gt_path, delimiter=',')
        
        # In case single frame sequence gets loaded as 1D array
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
            
    # Calculate Success Rate (AUC) from IoU list (100 thresholds from 0 to 1)
    thresholds = np.linspace(0.0, 1.0, 100)
    success_rates = [(np.array(iou_list) >= t).mean() for t in thresholds]
    auc = np.mean(success_rates) * 100.0
    
    # Calculate Distance Precision (PR at 20 pixels)
    pr = (np.array(distance_list) <= 20.0).mean() * 100.0
    
    return auc, pr

def main():
    seq_home = '/mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/sequences'
    seqs = sorted([f for f in os.listdir(seq_home) if os.path.isdir(os.path.join(seq_home, f))])
    
    print(f"Loaded {len(seqs)} DepthTrack Sequences.")
    
    print("\n--- Evaluating V20 Model (flextrackv2_b224_20) on DepthTrack ---")
    v20_dir = '/mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_20'
    v20_auc, v20_pr = evaluate_tracker(v20_dir, seq_home, seqs)
    print(f"V20 Success (AUC): {v20_auc:.2f}%")
    print(f"V20 Precision (PR @ 20px): {v20_pr:.2f}%")
    
    print("\n--- Evaluating V22 Model (flextrackv2_b224_22) on DepthTrack ---")
    v22_dir = '/mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_22'
    v22_auc, v22_pr = evaluate_tracker(v22_dir, seq_home, seqs)
    print(f"V22 Success (AUC): {v22_auc:.2f}%")
    print(f"V22 Precision (PR @ 20px): {v22_pr:.2f}%")
    
    print("\n--- V22 Improvement over V20 Baseline ---")
    print(f"Success (AUC) Gain: +{(v22_auc - v20_auc):+.2f}%")
    print(f"Precision (PR @ 20px) Gain: +{(v22_pr - v20_pr):+.2f}%")

if __name__ == '__main__':
    main()
