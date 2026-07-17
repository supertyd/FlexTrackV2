# FlexTrackV2 V20 Optimization Log

## Core Change
- **CMA Maximum Missing Probability (CMA_P_MAX) Reduction:** Modified CMA_P_MAX from 0.50 to 0.40.

## Rationale
In FlexTrackV2 V13, we observed a slight tradeoff in the standard (Complete) modality benchmarks on RGBT234 and VisEvent. This occurs because the maximum curriculum missing probability at the end of training is as high as 50%, which over-regularizes the joint modality representation space.
By reducing the upper bound of the missing modality probability CMA_P_MAX to 0.40, we maintain high robustness to missing modality sensor failure while allowing the dual-modality encoder to learn stronger, more balanced joint feature representation maps, boosting the complete modality benchmarks towards SOTA levels without sacrificing robustness.
