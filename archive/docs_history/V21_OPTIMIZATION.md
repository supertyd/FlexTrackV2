# FlexTrackV2 V21 Optimization Log

## Core Change
- **Knowledge Distillation Weight (DISTILL_WEIGHT) Scaling:** Increased the distillation loss weight `DISTILL_WEIGHT` from `1.5` to `2.0`.

## Rationale
In FlexTrackV2 V13, we introduced Masked Modality Distillation (MMD) with a distillation loss weight of 1.5. This aligns the fusion features of the student stream (with missing modalities) to the complete-modality teacher stream.
By scaling up the `DISTILL_WEIGHT` to `2.0` in V21, we impose a stronger alignment constraint between the complete-modality and missing-modality states. This stronger regularization encourages the network to reconstruct robust modal representations under severe sensor failures, allowing the model to achieve superior performance in missing modality benchmarks while maintaining a highly stable and optimized base tracking accuracy.
