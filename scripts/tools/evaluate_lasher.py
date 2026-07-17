import os
import numpy as np

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x, y, w, h] format.
    """
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
    """
    Compute Euclidean distance between the centers of two bounding boxes.
    Boxes are in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
    cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

def load_box_file(path):
    """
    Load bounding boxes from a text file. Supports comma, space, or tab delimiters.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            lines = f.read().splitlines()
        boxes = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split by comma, tab, or space
            if "," in line:
                parts = line.split(",")
            elif "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            if len(parts) >= 4:
                boxes.append([float(x) for x in parts[:4]])
        return np.array(boxes)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def evaluate_ope(pred_boxes, gt_boxes):
    """
    Evaluate a single sequence under One-Pass Evaluation (OPE).
    Returns lists of IoUs, center distances, and normalized center distances.
    """
    iou_list = []
    distance_list = []
    norm_distance_list = []
    
    n_frames = min(len(pred_boxes), len(gt_boxes))
    for idx in range(n_frames):
        pred_box = pred_boxes[idx]
        gt_box = gt_boxes[idx]
        
        # Compute IoU
        iou = compute_iou(pred_box, gt_box)
        iou_list.append(iou)
        
        # Compute Center Distance
        dist = compute_center_distance(pred_box, gt_box)
        distance_list.append(dist)
        
        # Compute Normalized Center Distance
        gt_w, gt_h = gt_box[2], gt_box[3]
        diagonal = np.sqrt(gt_w ** 2 + gt_h ** 2)
        if diagonal > 0:
            norm_distance_list.append(dist / diagonal)
        else:
            norm_distance_list.append(dist)
            
    return iou_list, distance_list, norm_distance_list

def calculate_metrics(iou_list, distance_list, norm_distance_list):
    """
    Calculate Success Rate (AUC), Precision Rate (PR @ 20px), and Normalized Precision Rate (NPR @ 0.2).
    """
    if not iou_list:
        return 0.0, 0.0, 0.0
        
    # Success Rate (AUC) over 101 thresholds (0.00 to 1.00 with step 0.01)
    thresholds = np.linspace(0.0, 1.0, 101)
    success_rates = [(np.array(iou_list) >= t).mean() for t in thresholds]
    auc = np.mean(success_rates) * 100.0
    
    # Precision Rate (PR @ 20px)
    pr = (np.array(distance_list) <= 20.0).mean() * 100.0
    
    # Normalized Precision Rate (NPR @ 0.2)
    npr = (np.array(norm_distance_list) <= 0.2).mean() * 100.0
    
    return auc, pr, npr

def evaluate_tracker(results_dir, gt_dir, seqs, gt_filename="init.txt"):
    """
    Evaluate a tracker on a list of sequences.
    """
    all_ious = []
    all_distances = []
    all_norm_distances = []
    
    seq_results = {}
    
    for seq in seqs:
        pred_path = os.path.join(results_dir, f"{seq}.txt")
        gt_path = os.path.join(gt_dir, seq, gt_filename)
        
        if not os.path.exists(pred_path):
            # Try alternative naming convention (e.g., seq_tracker.txt)
            pred_files = [f for f in os.listdir(results_dir) if f.startswith(seq) and f.endswith(".txt")]
            if pred_files:
                pred_path = os.path.join(results_dir, pred_files[0])
                
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
            
        pred_boxes = load_box_file(pred_path)
        gt_boxes = load_box_file(gt_path)
        
        if pred_boxes is None or gt_boxes is None:
            continue
            
        iou_list, distance_list, norm_distance_list = evaluate_ope(pred_boxes, gt_boxes)
        
        all_ious.extend(iou_list)
        all_distances.extend(distance_list)
        all_norm_distances.extend(norm_distance_list)
        
        # Calculate per-sequence metrics
        seq_auc, seq_pr, seq_npr = calculate_metrics(iou_list, distance_list, norm_distance_list)
        seq_results[seq] = {
            "success": seq_auc,
            "precision": seq_pr,
            "norm_precision": seq_npr
        }
        
    # Calculate overall metrics
    overall_auc, overall_pr, overall_npr = calculate_metrics(all_ious, all_distances, all_norm_distances)
    
    return {
        "overall": {
            "success": overall_auc,
            "precision": overall_pr,
            "norm_precision": overall_npr
        },
        "sequences": seq_results
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LasHeR Python Evaluation Toolkit")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to tracker results directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to LasHeR ground truth directory")
    parser.add_argument("--gt_filename", type=str, default="init.txt", help="Ground truth filename (default: init.txt)")
    args = parser.parse_args()
    
    # Get list of sequences
    if os.path.exists(args.gt_dir):
        seqs = sorted([f for f in os.listdir(args.gt_dir) if os.path.isdir(os.path.join(args.gt_dir, f))])
    else:
        print(f"Ground truth directory {args.gt_dir} does not exist.")
        seqs = []
        
    if seqs:
        print(f"Found {len(seqs)} sequences in ground truth directory.")
        results = evaluate_tracker(args.results_dir, args.gt_dir, seqs, args.gt_filename)
        
        print("\n====================================================================")
        print("                     LasHeR Evaluation Results                      ")
        print("====================================================================")
        print(f"Success (AUC):          {results['overall']['success']:.2f}%")
        print(f"Precision (PR @ 20px):  {results['overall']['precision']:.2f}%")
        print(f"Norm Precision (NPR):   {results['overall']['norm_precision']:.2f}%")
        print("====================================================================\n")
    else:
        print("No sequences to evaluate.")
