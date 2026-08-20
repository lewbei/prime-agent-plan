# Harbor Integration Specification: Official Terminal-Bench 2.0 Evaluation

> **Benchmark**: Terminal-Bench 2.0 (*arXiv:2601.11868*)  
> **Official Harness**: Harbor Framework (`harbor-framework/harbor`, `harbor-framework/terminal-bench-2`)  
> **Task Scope**: 89 Curated Terminal Agent Tasks inside Dockerized sandboxes  

---

## 1. Architecture Overview

Harbor is the official benchmark and evaluation harness designed by the creators of Terminal-Bench 2.0. It executes multi-turn interactive agents against containerized environments and grades outcomes using official verification test suites.

```text
Official Terminal-Bench 2.0 (89 Tasks via Harbor)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 [Arm A0: Base Agent]          [Arm A6: Prime Epistemic Runtime]
  - Raw Tool Calling            - PlanIR Causal Validator
  - No Invariant Checks         - HMAC Authorization Certificates
  - No Isolation Boundary       - Ephemeral Workspace Sandbox
  - Blind Completion            - LIFO Reverse Saga Compensation
       │                               │
       └───────────────┬───────────────┘
                       ▼
          [Harbor Container Sandbox]
                       │
          [Official Task Verifier]
                       │
          [Grade & Diagnostics Output]
```

---

## 2. Running Evaluation via Harbor CLI

To run the full 89-task Terminal-Bench 2.0 evaluation across ablation arms:

### Step 1: Install Harbor CLI
```bash
uv tool install harbor
# or
pip install harbor
```

### Step 2: Configure Foundation LLM API Credentials
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Step 3: Run Baseline (Arm A0) vs Full Prime (Arm A6)
```bash
# Baseline Arm A0
harbor run --dataset terminal-bench@2.0 \
  --agent custom \
  --agent-config benchmarks/harbor/harbor_adapter.py:PrimeHarborAgent \
  --config '{"ablation_arm": "A0"}' \
  --out results/harbor_tb2_a0.json

# Full Prime Arm A6
harbor run --dataset terminal-bench@2.0 \
  --agent custom \
  --agent-config benchmarks/harbor/harbor_adapter.py:PrimeHarborAgent \
  --config '{"ablation_arm": "A6"}' \
  --out results/harbor_tb2_a6.json
```

---

## 3. Disentangling Smoke Benchmarks from Official Benchmarks

| Benchmark Suite | Source | Task Count | Purpose |
| :--- | :--- | :---: | :--- |
| **EpiPlanBench-Smoke** | `benchmarks/epiplanbench/` | 11 Tasks | Fast local CI smoke testing, causal invariant verification, and saga compensation rollback sanity. |
| **Tasks Baseline (A-J)** | `benchmarks/frozen_v0/` | 5 Tasks | Synthetic regression smoke testing for Arms A through J. |
| **Terminal-Bench 2.0** | Harbor (`terminal-bench@2.0`) | 89 Tasks | Official empirical terminal agent evaluation with real LLMs and Docker environments. |
