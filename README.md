# Prime Agent Plan Mode (`/plan`)

> **Make the plan better before the agent starts doing the work.**

Plan Mode is a formal planning, causal validation, and epistemic verification engine for Prime Agent. Instead of executing the first draft produced by an LLM, `/plan` converts the draft into a structured **Canonical Plan IR**, runs **STRIPS causal validation**, searches the plan space using **evolutionary AST mutations**, tests feasibility via **sandboxed execution contracts**, and enforces **transactional commit with Saga rollback**.

---

## 1. System Architecture

```mermaid
flowchart TD
    %% Styling & Class Definitions
    classDef startNode fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc;
    classDef epistemic fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef search fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef runtime fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef gate fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef fail fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#f8fafc;

    subgraph PlanningPhase ["1. Orchestration & Iterative Planning"]
        A(["Objective & Draft Request"]):::startNode --> B["plan.start(objective)"]:::startNode
        B --> C["LLM Planner Drafts Markdown Plan"]
        C --> D["Canonical Plan IR Parser"]:::epistemic
    end

    subgraph EpistemicVerification ["2. Epistemic Verification Engine"]
        D --> E{"Epistemic Causal Validator"}:::gate
        E -->|Check Causal Links| E1["STRIPS Dependency & Threat Analysis
(Clobber Detection)"]:::epistemic
        E -->|Check Grounding| E2["Typed Fact Identity & Disk Grounding
(4-State Truth: Observed vs Projected)"]:::epistemic
        E -->|Check Simulation| E3["Forward State Simulation
(Executable State Transition)"]:::epistemic
    end

    subgraph SearchOptimization ["3. Evolutionary AST Search & Verification"]
        E -->|Flaws / Unmet Preconditions| F["Evolutionary AST Engine"]:::search
        F --> F1["Flaw-Directed Mutations"]:::search
        F --> F2["AST Subgraph Crossover"]:::search
        F --> F3["Same-Model Same-Thinking
Probabilistic Best-of-N Selector"]:::search
        F3 -->|Refined Candidate| D
    end

    subgraph ExecutionPhase ["4. Sandboxed Contract Execution & Saga Recovery"]
        E -->|Hard Gates PASS| G["Execution Contract Verification"]:::gate
        G --> G1["Spike / Feasibility Probe"]:::runtime
        G --> G2["AST Symbol Declaration Audit"]:::runtime
        G --> G3["Sandboxed Verification Commands"]:::runtime
        G1 & G2 & G3 --> H{"Attested Execution Ledger"}:::gate
        H -->|Verified Evidence & Exit Criteria Met| I(["Transactional Commit & Release"]):::runtime
        H -->|Runtime Drift / Failure| J["Saga Recovery
(Reverse LIFO Compensation)"]:::fail
        J -->|Compensation Verified| K(["Clean Rollback"]):::fail
        J -->|Compensation Blocked| L(["CONTAINMENT_FAILED"]):::fail
    end
```

---

## 2. Transactional Sandbox Execution & Saga Recovery

```mermaid
sequenceDiagram
    autonumber
    actor Orchestrator as Agent Orchestrator
    participant TxMgr as Transactional Execution Manager
    participant Sandbox as Execution Sandbox
    participant Ledger as Integrity-Linked Ledger
    participant Verifier as Epistemic Verifier
    participant Recovery as Saga Recovery Manager

    Orchestrator->>TxMgr: execute_plan(Plan IR, Contract)
    TxMgr->>Sandbox: Initialize Ephemeral Workspace & Scrub Env
    loop For Each Step in Plan Order
        TxMgr->>Sandbox: Execute Structured Action Argv
        Sandbox-->>TxMgr: Process Result (stdout, stderr, exit code)
        TxMgr->>Ledger: Append Action Execution Event
        TxMgr->>Verifier: Run Witness Verifier & Verify Postconditions
        alt Step Verification PASS
            Verifier-->>TxMgr: Witness Attestation OK
            TxMgr->>Ledger: Record Verified State
        else Step Verification FAIL / Drift
            Verifier-->>TxMgr: Verification Failed
            TxMgr->>Recovery: Trigger Saga Recovery
            loop Reverse LIFO Order of Executed Steps
                Recovery->>Sandbox: Execute Registered Compensation Action
                Sandbox-->>Recovery: Compensation Result
                Recovery->>Verifier: Witness Post-Compensation State
            end
            alt All Compensations Verified
                Recovery-->>TxMgr: Rollback Clean
                TxMgr-->>Orchestrator: Abort with Clean Rollback
            else Any Compensation Fails or Blocked
                Recovery-->>TxMgr: Containment Failure
                TxMgr-->>Orchestrator: FAIL (CONTAINMENT_FAILED)
            end
        end
    end
    TxMgr->>Verifier: Final Exit Criteria & Symbol Parity Audit
    Verifier-->>TxMgr: All Criteria Attested
    TxMgr->>Ledger: Commit Transaction
    TxMgr-->>Orchestrator: Success (Attested Evidence Released)
```

---

## 3. Core Engine Pillars

| Subsystem | Key Components | Purpose & Guarantees |
| :--- | :--- | :--- |
| **Canonical Plan IR & Epistemic Truth** | `plan_mode.ir`<br>`plan_mode.epistemic_validator`<br>`plan_mode.fact_identity` | Separates *projected causal truth* (for plan reasoning) from *empirical observed truth* across 4 truth states (`OBSERVED_TRUE`, `OBSERVED_FALSE`, `ASSUMED_TRUE`, `UNKNOWN`). Validates STRIPS causal links $\langle a_i, p, a_j \rangle$ and detects clobber threats. |
| **Evolutionary AST Search & Verification** | `plan_mode.ir_search`<br>`plan_mode.self_verification`<br>`plan_mode.judges` | Refines plans through flaw-directed AST mutations, subgraph crossover, and same-model same-thinking Best-of-N probabilistic candidate selection. Multi-provider and blind judge ensembles evaluate semantic alignment. |
| **Execution Contracts & Symbol Parity** | `plan_mode.execution_contract`<br>`plan_mode.probing` | Binds plan claims to verifiable real-world evidence: executable feasibility probes, AST symbol audits (detecting missing or undeclared functions/variables), file size/line budgets, and parity checks. |
| **Sandboxed Execution & Integrity Ledger** | `plan_mode.runtime.sandbox`<br>`plan_mode.runtime.ledger`<br>`plan_mode.runtime.executor` | Executes structured `argv` commands with process isolation, environment scrubbing, strict timeouts, and witness attestation. All state changes are recorded in an integrity-linked event ledger. |
| **Saga Compensation & Recovery** | `plan_mode.runtime.transaction`<br>`plan_mode.recovery` | Dispatches registered compensation actions in reverse LIFO order upon failure or runtime drift. Fails closed with `CONTAINMENT_FAILED` if compensation is blocked, unobservable, or unverified. |
| **Cordis Spatiotemporal Composability** | `plan_mode.cordis` | Provides unified context $\Gamma_\infty$, Fiber calculus, dynamic Provider Stacks, and Twisted Monoid $(f_1, g_1) \circ (f_2, g_2) = (f_1 \circ f_2, g_2 \circ g_1)$ LIFO inverse rollback. |

---

## 4. Quick Start & Python API

### Basic Planning Loop
```python
import plan

# 1. Initialize or resume planning session
session = plan.start("Refactor auth pipeline to use JWT with Redis cache", max_rounds=6)

# 2. Assess a candidate plan draft
result = plan.assess(
    session,
    plan_text,
    require_execution_contract=True,
    run_probe=True,
)

# 3. Check convergence status and critiques
if result["status"] == "converged":
    print("Plan is clean, causally valid, and grounded!")
elif result["status"] == "plateaued":
    print("Hard gate failure:", result.get("convergence_checks"))
else:
    print(f"Open critiques ({len(result['critiques'])}):", result["critiques"])
```

### Inherited Best-of-N Self-Verification
```python
# Evaluate multiple candidate drafts using the active runtime model and thinking profile
best_candidate = plan.assess_candidates(
    session,
    drafts=[draft_a, draft_b, draft_c],
    n_evaluations=3,
)
```

### Transactional Plan Execution with Saga Rollback
```python
from plan_mode.runtime import TransactionalExecutionManager, EphemeralWorkspace

workspace = EphemeralWorkspace(base_dir="./workspaces")
manager = TransactionalExecutionManager(workspace=workspace)

# Execute plan with integrity ledger and reverse compensation on failure
outcome = manager.execute(
    plan_ir=canonical_plan_ir,
    capability_registry=registry,
)

if outcome.committed:
    print("Transaction committed with attested evidence:", outcome.evidence)
else:
    print("Transaction aborted and rolled back. Status:", outcome.status)
```

---

## 5. Testing & Verification

```bash
# Run full unit and adversarial test suite
python -m pytest -q

# Run internal engine self-check
python -c "import plan; import plan_mode; res = plan_mode.selfcheck(run_pytest=False); assert res['ok']"

# Run EpiPlanBench benchmark suite
python benchmarks/run_epiplanbench.py
```
