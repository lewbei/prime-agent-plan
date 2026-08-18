# Synthetic Metrics Smoke Test

> **Notice:** This is not an empirical planning benchmark. Outcomes are synthetic fixtures used only to test metrics calculation. No architecture performance conclusion may be drawn from these values.

**Evaluation Date**: 2026-08-18 10:44:55 UTC

## Summary Across Synthetic Smoke Configurations

| Configuration ID | Fixture Description | False-PASS Rate | Safety Score | UNKNOWN Resolution | Rollback Recovery | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Arm_A_Blind_LLM` | Synthetic fixture: unverified direct execution | **100.0%** | 0.00 | 0.0% | 0.0% | **0.0000** |
| `Arm_B_Linear_PlanIR` | Synthetic fixture: boolean state only | **100.0%** | 0.00 | 0.0% | 100.0% | **0.0000** |
| `Arm_C_4State_Lattice` | Synthetic fixture: 4-state lattice (rejects UNKNOWN without probing) | **0.0%** | 1.00 | 0.0% | 100.0% | **0.7000** |
| `Arm_D_Random_Probing` | Synthetic fixture: 4-state lattice + random probing | **0.0%** | 1.00 | 75.0% | 0.0% | **0.9250** |
| `Arm_E_VOI_Probing` | Synthetic fixture: 4-state lattice + VOI-guided probing | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_F_Auth_Certificates` | Synthetic fixture: VOI probing + HMAC authorization certificates | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_G_Sandbox_Containment` | Synthetic fixture: certificates + structured process runner | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_H_Evidence_Ledger` | Synthetic fixture: process runner + in-memory event chain | **0.0%** | 1.00 | 100.0% | 0.0% | **1.0000** |
| `Arm_I_Saga_Dual_Judges` | Synthetic fixture: saga recovery + dual heuristic judges | **0.0%** | 1.00 | 100.0% | 100.0% | **1.0000** |
| `Arm_J_Full_Epistemic_Runtime` | Synthetic fixture: proposed full configuration | **0.0%** | 1.00 | 100.0% | 100.0% | **1.0000** |