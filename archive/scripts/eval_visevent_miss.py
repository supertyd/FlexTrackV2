import os
from evaluate_lasher_visevent import evaluate_tracker_dataset

gt_dir = "/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test"
testlist_path = os.path.join(gt_dir, "testlist.txt")
with open(testlist_path, "r") as f:
    seqs = sorted(set(f.read().splitlines()))

mci_dir = "/mnt/task_runtime/workspace/results/VisEvent_miss/flextrackv2_b224_56"
flex_dir = "/mnt/task_runtime/workspace/results/VisEvent_miss/FlexTrack"

print(f"--- Evaluating VisEvent_miss on {len(seqs)} sequences ---")
mci_auc, mci_pr = evaluate_tracker_dataset(mci_dir, gt_dir, seqs, "groundtruth.txt", is_visevent=True)
flex_auc, flex_pr = evaluate_tracker_dataset(flex_dir, gt_dir, seqs, "groundtruth.txt", is_visevent=True)

print(f"flextrackv2_v56_miss | Success (AUC): {mci_auc:.2f}% | Precision (PR): {mci_pr:.2f}%")
print(f"FlexTrack_miss    | Success (AUC): {flex_auc:.2f}% | Precision (PR): {flex_pr:.2f}%")
