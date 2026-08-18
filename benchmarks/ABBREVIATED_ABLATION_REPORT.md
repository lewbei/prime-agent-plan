# Epistemic Planning & Verification Runtime: Ablation Benchmark Report
**Evaluation Date**: 2026-08-18 09:39:32 UTC

## Summary Evaluation Across Arms A–J

| Arm ID | Description | False-PASS Rate | Safety Score | UNKNOWN Resolution | Rollback Recovery | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Arm_A_Blind_LLM` | Blind LLM Baseline (Unverified Direct Execution) | **100.0%** | 0.00 | 0.0% | 0.0% | **0.0000** |
| `Arm_B_Linear_PlanIR` | Linear PlanIR (Boolean State Only) | **100.0%** | 0.00 | 0.0% | 100.0% | **0.0000** |
| `Arm_C_4State_Lattice` | 4-State Lattice (Rejects UNKNOWN Without Probing) | **0.0%** | 1.00 | 0.0% | 100.0% | **0.7000** |
| `Arm_D_Random_Probing` | 4-State Lattice + Random Probing | **0.0%** | 1.00 | 75.0% | 0.0% | **0.9250** |
| `Arm_E_VOI_Probing` | 4-State Lattice + VOI-Guided Probing | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_F_Auth_Certificates` | VOI Probing + HMAC Authorization Certificates | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_G_Sandbox_Containment` | Certificates + Sandbox argv Containment | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_H_Evidence_Ledger` | Sandbox + Hash-Chained Evidence Ledger | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_I_Saga_Dual_Judges` | Saga Recovery + Dual Divergence Judges | **0.0%** | 1.00 | 100.0% | 100.0% | **1.0000** |
| `Arm_J_Full_Epistemic_Runtime` | Full Epistemic Planning & Verification Runtime (All TCB Features) | **0.0%** | 1.00 | 100.0% | 100.0% | **1.0000** |