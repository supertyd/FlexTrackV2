# FlexTrackV2 V23 Optimization Log

## Core Change
- **Minimum Curriculum Missing Probability (CMA_P_MIN) Minimization:** Reduced the initial missing modality probability `CMA_P_MIN` from `0.10` to `0.00`.

## Rationale
In FlexTrackV2 V13, we initialized Curriculum Missing Augmentation (CMA) with a starting missing probability of 10%. While relatively low, introducing modality masking in the first epoch can still disturb the very early phases of parameter convergence, where the self-attention matrices are trying to align healthy dual-modality cues.
By reducing `CMA_P_MIN` to `0.00` in V23, we allow the network to train with 100% complete and healthy dual-modality inputs in the first epoch. This clean initialization establishes a much sturdier and more stable baseline for representational learning, which is then gradually exposed to progressively tougher modality dropout as training evolves, unlocking superior joint feature representations and elevating standard and sensor-failure tracking precision.
