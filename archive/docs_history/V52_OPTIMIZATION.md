# FlexTrackV2 V52 Optimization Log

## Core Architectural Improvement (BMR-HMoE for PAMI Extension)
Introduced the **Bilateral Modality-Specific Feature Reconstruction & Hallucination Network with Heterogeneous MoE Gating (BMR-HMoE)** with Orthogonality-Constrained Gating (`BMR_HMoE`).

## Technical Rationale & Implementation Detail
This version builds directly on top of the V13 baseline, integrating your signature heterogeneous capacity experts and extending it with state-of-the-art multimodal representation theory for PAMI:
1. **8 Heterogeneous Capacity Experts**: Deploys exactly 8 experts with exponential capacities `[4, 8, 16, 32, 64, 128, 256, 512]` (completely removing capacity 2 as requested). This maintains your core innovation of adaptive capacity.
2. **Bilateral Modality Hallucination & Reconstruction**: Features two lightweight cross-modal projection modules (`rgb_to_aux_hallucinater` and `aux_to_rgb_hallucinater`). If either sensor modality goes offline, the counterpart hallucinates and reconstructs its representation, ensuring the routing layer and experts always receive intact dual-modality information.
3. **Bilateral Consistency Loss**: Incorporates an MSE-based self-distillation loss between the hallucinated representation and active ground truth representation to strictly align the reconstruction.
4. **Orthogonality Constraint**: Adds a gating router orthogonal regularization term to force specialized, non-redundant experts routing assignments.
