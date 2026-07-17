# FlexTrackV2 V51 Optimization Log

## Core Architectural Improvement (BCA-HMoE for PAMI Extension)
Introduced the **Bilateral Context-Aware Heterogeneous Mixture of Experts (BCA-HMoE)** with Orthogonality-Constrained Gating (`BCA_HMoE`).

## Technical Rationale & Implementation Detail
This design preserves and elevates your signature innovation—**Heterogeneous experts of varying capacity/width**—for the upcoming PAMI extension:
1. **Bilateral Context-Aware Routing (BCAR)**: Computes bilateral query-key cross-modal attention between RGB and Aux to construct contextually rich token representations. Gating logits are computed directly from this bilateral representation, allowing routing decisions to be jointly aware of the degradation level of active modal sensors.
2. **Specialized Heterogeneous Experts**: Successfully retains your original, core adaptive capacity expert configurations (adapter channels ranging from 4 to 18).
3. **Orthogonality Gated Regularization Constraint**: Imposes an orthogonality penalty on the gating parameter matrices to enforce specialized, non-redundant complementary routing assignments across joint modal states.
