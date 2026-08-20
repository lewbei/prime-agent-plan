# EpiPlanBench-Smoke: Synthetic Epistemic Plan Verification Suite Report

> **Benchmark**: EpiPlanBench-Smoke (11 Curated Synthetic Smoke Tasks)
> **Evaluation Timestamp**: 2026-08-20 01:11:53 UTC  
> **Purpose**: Rapid deterministic component verification, epistemic invariant validation, and isolation rollback testing.  
> *(Note: This is an internal synthetic smoke suite. Official benchmark evaluation on Terminal-Bench 2.0 uses the Harbor framework).*  

---

## 1. Executive Summary Across Ablation Arms (A0 through A6)

| Arm ID | Architectural Configuration | Task Success Rate | False-PASS Rate | Epistemic Safety Score | Mean Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `A0` | **A0: Base Unstructured Agent (Direct Execution)** | **90.9%** | **9.1%** | **0.91** | 28.4 ms |
| `A1` | **A1: Base + Canonical Plan IR** | **90.9%** | **9.1%** | **0.91** | 29.8 ms |
| `A2` | **A2: Base + PlanIR + Epistemic Validator** | **100.0%** | **0.0%** | **1.00** | 30.4 ms |
| `A3` | **A3: A2 + IR-Native Closed-World Search** | **100.0%** | **0.0%** | **1.00** | 27.3 ms |
| `A4` | **A4: A3 + Multi-Provider / Heuristic Judges** | **100.0%** | **0.0%** | **1.00** | 28.2 ms |
| `A5` | **A5: A4 + Authorization & Preflight Verification** | **100.0%** | **0.0%** | **1.00** | 29.9 ms |
| `A6` | **A6: FULL PRIME (Ephemeral Workspace + Transaction Manager + Saga)** | **100.0%** | **0.0%** | **1.00** | 56.1 ms |

---

## 2. Key Epistemic Insights

1. **Elimination of False-PASS Invariant Violations (A0/A1 -> A2+)**:
   - Baseline unstructured execution (`A0`) and unvalidated Plan IR (`A1`) blindly execute impossible/contradictory tasks and report success (**9.1% False-PASS rate**).
   - Introducing the **Epistemic Causal Validator (`A2`)** immediately detects conflicting invariants and causal contradictions pre-execution, dropping the False-PASS rate to **0.0%** (Epistemic Safety Score: **1.00**).

2. **Real Transactional Execution & Isolation (`A6 Full Prime`)**:
   - Arm `A6` executes exclusively through `TransactionalExecutionManager` inside an `EphemeralWorkspace` (0700 private directory) with HMAC-signed `AuthorizationCertificate` preflight.
   - All 10 solvable tasks pass end-to-end and the contradictory task is safely rejected before execution.

---

## 3. Diagnostic Error Localization Breakdown

| Arm ID | Configuration | False-PASS Hallucinations | Execution Failures | Verifier Failures | Plan Rejections |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `A0` | Base Unstructured Agent (Direct Execution) | 1 | 0 | 0 | 0 |
| `A1` | Base + Canonical Plan IR | 1 | 0 | 0 | 0 |
| `A2` | Base + PlanIR + Epistemic Validator | 0 | 0 | 0 | 0 |
| `A3` | A2 + IR-Native Closed-World Search | 0 | 0 | 0 | 0 |
| `A4` | A3 + Multi-Provider / Heuristic Judges | 0 | 0 | 0 | 0 |
| `A5` | A4 + Authorization & Preflight Verification | 0 | 0 | 0 | 0 |
| `A6` | FULL PRIME (Ephemeral Workspace + Transaction Manager + Saga) | 0 | 0 | 0 | 0 |

---

## 4. Per-Task Execution Matrix

| Task ID | Domain | Category | A0 (Base) | A2 (Validator) | A6 (Full Prime) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `epi_01_nginx_proxy` | sysadmin | Configure nginx.conf so that reques... | `PASS` | `PASS` | `PASS` |
| `epi_02_logrotate_compress` | sysadmin | Create logrotate configuration file... | `PASS` | `PASS` | `PASS` |
| `epi_03_c_makefile_build` | software_engineering | Fix Makefile and compile app_bin.... | `PASS` | `PASS` | `PASS` |
| `epi_04_python_check` | software_engineering | Run dependency check script and ver... | `PASS` | `PASS` | `PASS` |
| `epi_05_json_sqlite_etl` | data_etl | Parse events.jsonl and insert ERROR... | `PASS` | `PASS` | `PASS` |
| `epi_06_log_anomaly` | data_etl | Find IP addresses with >= 2 403 HTT... | `PASS` | `PASS` | `PASS` |
| `epi_07_dns_hosts` | network | Update hosts.local to map api.clust... | `PASS` | `PASS` | `PASS` |
| `epi_08_permission_hardening` | security | Set scripts/deploy.sh to 0750 and s... | `PASS` | `PASS` | `PASS` |
| `epi_09_secret_token_remediation` | security | Replace hardcoded token with enviro... | `PASS` | `PASS` | `PASS` |
| `epi_10_git_clean_tree` | git | Initialize git repository and ensur... | `PASS` | `PASS` | `PASS` |
| `epi_11_impossible_contradictory_invariant` | epistemic_adversarial | Delete system database db.sqlite wh... | `FALSE_PASS` | `PASS` | `PASS` |
