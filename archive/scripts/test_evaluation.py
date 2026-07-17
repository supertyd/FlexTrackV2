import numpy as np
from evaluate_lasher import compute_iou, compute_center_distance, calculate_metrics

def test_iou():
    # Test case 1: Perfect overlap
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert abs(compute_iou(box1, box2) - 1.0) < 1e-6
    
    # Test case 2: No overlap
    box1 = [10, 10, 50, 50]
    box2 = [100, 100, 50, 50]
    assert abs(compute_iou(box1, box2) - 0.0) < 1e-6
    
    # Test case 3: Partial overlap
    box1 = [10, 10, 50, 50]
    box2 = [35, 10, 50, 50]
    # Intersection: x from 35 to 60 (width 25), y from 10 to 60 (height 50) -> area = 1250
    # Union: 2500 + 2500 - 1250 = 3750
    # IoU: 1250 / 3750 = 1/3
    assert abs(compute_iou(box1, box2) - 1.0/3.0) < 1e-6
    print("compute_iou tests passed!")

def test_center_distance():
    box1 = [10, 10, 50, 50]  # Center: [35, 35]
    box2 = [40, 50, 50, 50]  # Center: [65, 75]
    # Distance: sqrt((65-35)^2 + (75-35)^2) = sqrt(30^2 + 40^2) = 50
    assert abs(compute_center_distance(box1, box2) - 50.0) < 1e-6
    print("compute_center_distance tests passed!")

def test_metrics():
    # Mock lists
    iou_list = [0.8, 0.5, 0.2, 0.0]
    distance_list = [5.0, 15.0, 25.0, 100.0]
    norm_distance_list = [0.05, 0.15, 0.25, 1.0]
    
    auc, pr, npr = calculate_metrics(iou_list, distance_list, norm_distance_list)
    
    # Precision Rate (PR @ 20px): 2 out of 4 are <= 20.0 -> 50.0%
    assert abs(pr - 50.0) < 1e-6
    
    # Normalized Precision Rate (NPR @ 0.2): 2 out of 4 are <= 0.2 -> 50.0%
    assert abs(npr - 50.0) < 1e-6
    
    print(f"calculate_metrics tests passed! AUC: {auc:.2f}%, PR: {pr:.2f}%, NPR: {npr:.2f}%")

if __name__ == "__main__":
    test_iou()
    test_center_distance()
    test_metrics()
    print("All tests passed successfully!")
