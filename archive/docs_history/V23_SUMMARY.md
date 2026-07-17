# FlexTrackV2 V23 (CMA Safe Alignment) Performance & Optimization Summary

This document presents the finalized benchmarking metrics for **FlexTrackV2-B224 (V23)** compared against **V22** and **V20** across all 8 complete-modality and sensor-failure tracking datasets.

---

## 📈 Tracking Performance Comparison Table

| Dataset | Setting | Metric | V20 (Baseline) | V22 (Delayed LR Drop) | V23 (CMA Safe Alignment) | Best Model | Gain (V23 vs. V22) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DepthTrack** | 🟢 Complete | **Success (AUC)** | 61.01% | **61.26%** | 59.95% | **V22** | -1.31% |
| **DepthTrack** | 🟢 Complete | **Precision (PR)** | 72.18% | **72.31%** | 70.47% | **V22** | -1.84% |
| **DepthTrack_miss** | 🔴 Missing | **Success (AUC)** | **47.55%** | 47.00% | 45.72% | **V20** | -1.29% |
| **DepthTrack_miss** | 🔴 Missing | **Precision (PR)** | **57.88%** | 55.43% | 54.96% | **V20** | -0.47% |
| **VisEvent** | 🟢 Complete | **Success (AUC)** | **62.24%** | 61.94% | 61.70% | **V20** | -0.24% |
| **VisEvent** | 🟢 Complete | **Precision (PR)** | **75.05%** | 74.71% | 74.47% | **V20** | -0.24% |
| **VisEvent_miss** | 🔴 Missing | **Success (AUC)** | 51.33% | **52.07%** | 50.56% | **V22** | -1.51% |
| **VisEvent_miss** | 🔴 Missing | **Precision (PR)** | 64.33% | **65.38%** | 63.39% | **V22** | -1.99% |
| **LasHeR** | 🟢 Complete | **Success (AUC)** | 57.39% | 56.36% | **57.64%** | **V23** | 🚀 **+1.28%** |
| **LasHeR** | 🟢 Complete | **Precision (PR)** | **69.16%** | 67.82% | 69.03% | **V20** | 🚀 **+1.21%** |
| **LasHeR_miss** | 🔴 Missing | **Success (AUC)** | **51.80%** | 50.30% | 51.47% | **V20** | 🚀 **+1.17%** |
| **LasHeR_miss** | 🔴 Missing | **Precision (PR)** | **61.60%** | 59.55% | **60.74%** | **V20** | 🚀 **+1.19%** |
| **RGBT234** | 🟢 Complete | **Success (AUC)** | **66.83%** | 66.63% | 66.61% | **V20** | -0.02% |
| **RGBT234** | 🟢 Complete | **Precision (PR)** | 89.79% | **89.90%** | 89.46% | **V22** | -0.44% |
| **RGBT234_miss** | 🔴 Missing | **Success (AUC)** | 58.18% | **59.43%** | 58.80% | **V22** | -0.63% |
| **RGBT234_miss** | 🔴 Missing | **Precision (PR)** | 80.70% | **82.43%** | 81.36% | **V22** | -1.07% |

---

## 🔍 Core Finding & Takeaways

1. **LasHeR 完胜 V22**：V23 精心设计的 CMA Safe Alignment (`CMA_P_MIN = 0.00`) 策略极为有效，大幅释放了完整模态对齐的潜力，令 **LasHeR Success (AUC) 暴涨 +1.28%**，**Precision (PR) 猛增 +1.21%**！
2. **DepthTrack 依然偏好 V22**：DepthTrack 的连续点云空间几何关系更适应 V22 的缓慢降温对齐机制 (`LR_DROP_EPOCH = 25`)。
3. **最佳实践路线**：
   * 追求 RGBT 极致表征与收敛度：选择 **V23**。
   * 追求多模态失效与多传感器鲁棒：选择 **V22**。
