# Adversarial Code Audit & Stress Test Report

**Target Repository:** `/home/lewbei/.prime/agent/skills/plan` (`prime-agent-plan`)  
**Engine Version:** v0.15.0  
**Audit Scope:** Execution Contracts, AgentRewind Checkpoints, ACID-Style Commits, RoT Experience Tree, Replanning Ladder Tier 4 Drift Recovery, Numerical Version Parsing, and Concurrency/Flock Mechanics.  
**Auditor:** Independent Adversarial Code Auditor & Stress Tester  
**Date:** 2026-08-17  

---

## 1. Executive Summary & Verdict

### Final Verdict: **FAIL (REQUIRES DEFENSIVE HARDENING BEFORE PRODUCTION USE)**

While the unit test suite passes cleanly (119/119 tests passing) and multi-threaded/multi-process locking primitives demonstrate strong reentrancy and isolation, adversarial probing revealed **several severe uncaught exceptions and logic flaws** that can crash active agent sessions or cause silent data misbehavior when handling untrusted/LLM-generated plan texts.

### Key Risk Summary
1. **Critical Crash on Contract Parsing:** `parse_execution_contract` crashes with unhandled `TypeError` / `AttributeError` when JSON contract keys contain `null` or unexpected primitive types (common in LLM generation).
2. **Critical Crash on Probe Execution:** `probe_contract` / `run_command` only catches `subprocess.TimeoutExpired`, allowing `FileNotFoundError` and `PermissionError` to crash `assess(run_probe=True)`.
3. **False-Positive Storm in Symbol Audit:** `_symbols_from_source` traverses into inner function bodies and class methods, capturing local variables and methods as module-level symbols, which causes valid Python code to fail `symbol_audit`.
4. **Silent State Hijacking on Rewind:** `rewind(checkpoint_id=...)` silently falls back to the latest checkpoint when an invalid or typoed checkpoint ID is requested, rather than raising an error.
5. **Tuple Comparison Anomaly in Version Parsing:** Two-part versions like `"0.14"` evaluate to `(0, 14)` which Python compares as strictly smaller than `(0, 14, 0)`, falsely classifying them as historical/outdated.
6. **Index Out-of-Bounds in State Accessors:** `committed()` and `best()` do not validate version boundaries, causing `IndexError` on out-of-bounds versions and index `-1` wraparound on version 0.

---

## 2. Detailed Adversarial Audit Findings

### Subsystem 1: Execution Contracts & Feasibility Probes (`src/plan_mode/execution_contract.py`)

#### Finding 1.1: Uncaught `TypeError` and `AttributeError` on `null` or Primitive Contract Fields
- **Severity:** High (Crash Hazard)
- **Location:** `src/plan_mode/execution_contract.py:84-92`
- **Mechanism:**
  ```python
  contract = ExecutionContract(
      probe=data.get("probe", {}) if isinstance(data.get("probe"), dict) else {},
      verification_commands=[cmd for cmd in data.get("verification_commands", []) if isinstance(cmd, list)],
      expected_artifacts={str(k): (v if isinstance(v, dict) else {}) for k, v in data.get("expected_artifacts", {}).items()},
      workspace_invariants=[str(x) for x in data.get("workspace_invariants", [])],
      parity_checks=[p for p in data.get("parity_checks", []) if isinstance(p, dict)],
      symbols={str(k): (v if isinstance(v, dict) else {}) for k, v in data.get("symbols", {}).items()},
      raw=data,
  )
  ```
  If `data.get("expected_artifacts")` is `None` or a string, calling `.items()` raises `AttributeError: 'NoneType' object has no attribute 'items'`. Similarly, if `verification_commands`, `workspace_invariants`, or `parity_checks` are `None` or integers, list comprehensions raise `TypeError: 'NoneType' object is not iterable`.
- **Impact:** Any plan evaluated via `assess()` or `release()` containing `{"symbols": null}` or `{"verification_commands": null}` crashes the entire process.

#### Finding 1.2: Indented Markdown Code Fences Missed by Regex Matcher
- **Severity:** Medium (Parser Blindspot)
- **Location:** `src/plan_mode/execution_contract.py:59-67`
- **Mechanism:** The regex patterns `r"##\s*Execution\s+Contract.*?
```json\s*
(.*?)
```"` and `r"```json\s*
(.*?)
```"` require the opening and closing triple backticks to follow a newline immediately without leading whitespace.
- **Impact:** Contracts indented inside numbered lists, blockquotes, or formatted markdown sections are ignored and reported as `execution contract missing`.

#### Finding 1.3: Uncaught `ValueError` on Malformed Artifact Budgets
- **Severity:** Medium (Crash Hazard)
- **Location:** `src/plan_mode/execution_contract.py:126, 133`
- **Mechanism:** `int(budget["min_bytes"])` and `int(budget["min_lines"])` are invoked directly without defensive validation. If a plan provides `"min_bytes": "100KB"` or `"min_bytes": "invalid"`, a raw `ValueError` is raised instead of returning a validation error.

#### Finding 1.4: Uncaught `FileNotFoundError` in `run_command` and `probe_contract`
- **Severity:** High (Crash Hazard)
- **Location:** `src/plan_mode/execution_contract.py:246-258`
- **Mechanism:** `run_command` only handles `subprocess.TimeoutExpired`. If the spike binary specified in `probe["command"]` does not exist on the system (e.g. `["missing-tool", "--check"]`), `subprocess.run` raises `FileNotFoundError`.
- **Impact:** Invoking `assess(..., run_probe=True)` on candidate plans with missing CLI dependencies crashes the agent rather than recording a failed probe critique.

#### Finding 1.5: Symbol Audit False-Positive Storm on Local Variables & Class Methods
- **Severity:** High (Usability / Verification Loop Blocker)
- **Location:** `src/plan_mode/execution_contract.py:139-178` (`_symbols_from_source`) and `symbol_audit`
- **Mechanism:** `_symbols_from_source` traverses the full AST recursively, calling `self.generic_visit` inside `visit_FunctionDef` and `visit_ClassDef`. As a result:
  1. Internal local variables assigned inside functions (e.g., `temp = 1`, `for x in ...`) are added to `actual_vars`.
  2. Methods defined inside classes (e.g., `__init__`, `execute`) are added to `actual_funcs`.
  3. `symbol_audit` then asserts that every actual function and variable must be explicitly listed in `contract.symbols`.
- **Impact:** Every realistic Python file fails `symbol_audit` with `undeclared variables not listed in contract` or `undeclared functions not listed in contract` unless internal local variables and class methods are enumerated in the plan JSON.

#### Finding 1.6: Non-Python Files in `scan_symbols` Falsely Reported as Missing
- **Severity:** Low (False Positive)
- **Location:** `src/plan_mode/execution_contract.py:186-188`
- **Mechanism:** `if not p.exists() or p.suffix != ".py": out[raw] = {..., "missing": True}`. If an artifact in `symbols` is `config.json` or `schema.yaml`, `scan_symbols` marks it as `missing: True` regardless of whether the file exists on disk.

---

### Subsystem 2: AgentRewind Checkpoints & Session Restores (`src/plan_mode/__init__.py`)

#### Finding 2.1: Silent Fallback to Latest Checkpoint on Invalid `checkpoint_id`
- **Severity:** Medium (State Integrity / Silent Failure)
- **Location:** `src/plan_mode/__init__.py:934`
- **Mechanism:**
  ```python
  cp = next((c for c in reversed(cps) if c.get("id") == checkpoint_id), cps[-1]) if checkpoint_id else cps[-1]
  ```
  If `checkpoint_id` is supplied as `"cp-non-existent-123"`, the generator exhausts and `next()` falls back to `cps[-1]` (the latest checkpoint).
- **Impact:** The session is silently rewound to an unintended state without warning the user or agent that the requested checkpoint ID was not found.

#### Finding 2.2: Deep Copy Coverage in `_CHECKPOINT_FIELDS`
- **Status:** **PASS**
- **Detail:** All relevant session fields (`rounds`, `best_version`, `best_score`, `search_tree`, `release_gate`, `committed_*`, etc.) are deep-copied on checkpoint and successfully restored on rewind without leaking mutations.

#### Finding 2.3: `search(checkpoint_before=True)` Hook
- **Status:** **PASS**
- **Detail:** Verified that `plan.search(..., checkpoint_before=True)` captures a snapshot tagged `search:<mode>:pre-expansion` prior to running plan tree expansions.

---

### Subsystem 3: ACID Commit vs Best Separation (`src/plan_mode/__init__.py`)

#### Finding 3.1: Commit Gate Separation & Hash Immutability
- **Status:** **PASS**
- **Detail:** `plan.committed()` strictly requires a successful `release()` gate passing all mechanical, verify, feasibility, simulation, contract, and judge checks before promoting `best_version` to `committed_version`. Failed releases leave `committed_version` untouched.

#### Finding 3.2: Boundary Hazard in `committed()` and `best()`
- **Severity:** Medium (Crash Hazard)
- **Location:** `src/plan_mode/__init__.py:853, 868`
- **Mechanism:** `r = s["rounds"][ver - 1]` does not verify `1 <= ver <= len(s["rounds"])`. If `ver` is manually mutated, out of sync, or set to `0`, it causes `IndexError` or indexes the last item via `-1`.

---

### Subsystem 4: RoT Experience Tree & Environmental Outcome Feedback (`src/plan_mode/memory_distiller.py`)

#### Finding 4.1: Perspective Namespacing & Isolation
- **Status:** **PASS**
- **Detail:** `RoTRuleBase.distill_from_flaws` correctly keys rules with `perspective:<flaw_type>:<hash>` and `check_plan_violations(perspective=...)` isolates rules by analytical perspective.

#### Finding 4.2: Outcome Feedback Calibration & Confidence Tracking
- **Status:** **PASS**
- **Detail:** `record_outcome(rule_id, success=..., evidence=...)` correctly increments success/failure counters and adjusts confidence dynamically. `tree_report()` accurately aggregates hierarchical perspectives.

---

### Subsystem 5: Replanning Ladder Tier 4 Drift Recovery (`src/plan_mode/memory_distiller.py`)

#### Finding 5.1: Drift Signal Escalation Priority
- **Status:** **PASS**
- **Detail:** `ReplanningLadder.determine_replan_tier` detects drift keywords (`"drift"`, `"silent failure"`, `"schema violation"`, `"divergence"`, `"unexpected side effect"`, `"behavioral drift"`, `"irreversible"`) and elevates directly to Tier 4 runtime drift recovery before evaluating retry counts.

#### Finding 5.2: Inconsistent Return Key Naming
- **Severity:** Low (API Inconsistency)
- **Location:** `src/plan_mode/memory_distiller.py:210, 226, 233`
- **Detail:** Tier 1 and Tier 4 return `"task_id"`, Tier 2 returns `"failed_task_id"`, and Tier 3 returns neither. Consumers inspecting `tier_dict.get("task_id")` receive `None` on Tier 2.

#### Finding 5.3: Last-Task Boundary Condition Skipping Tier 2
- **Severity:** Low (Behavioral Anomaly)
- **Location:** `src/plan_mode/memory_distiller.py:223`
- **Mechanism:** `if retry_count <= 2 and failed_task_id < total_tasks:`. If the failure occurs on the final task (`failed_task_id == total_tasks`), the condition evaluates to `False` and jumps immediately to Tier 3 global redraft rather than attempting a Tier 2 subgraph repair.

---

### Subsystem 6: Numerical Version Parsing in `selfcheck` & `fold_history`

#### Finding 6.1: Tuple Length Comparison Anomaly in `_version_tuple`
- **Severity:** Medium (Version Logic Flaw)
- **Location:** `src/plan_mode/__init__.py:79-88`
- **Mechanism:**
  ```python
  def _version_tuple(v: str | None) -> tuple[int, ...]:
      if not v:
          return (0, 0, 0)
      try:
          return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])
      except Exception:
          return (0, 0, 0)
  ```
  For `"0.14"`, `re.findall` yields `["0", "14"]`, producing `(0, 14)`. In Python, `(0, 14) < (0, 14, 0)` evaluates to `True`. Consequently, two-part versions are incorrectly classified as historical/pre-0.14.
- **Remediation:** Pad tuple slices to length 3 with zeroes: `(vt + (0, 0, 0))[:3]`.

---

### Subsystem 7: Concurrency, Flock Locking & Reentrancy Stress Tests

#### Finding 7.1: Multi-Threaded & Multi-Process Stress Results
- **Status:** **PASS**
- **Stress Test Conditions:**
  - 8 concurrent threads simultaneously acquiring session locks, creating checkpoints, and evaluating plans.
  - 6 independent spawned processes executing concurrent assessments on the same session directory with `fcntl.flock`.
- **Results:**
  - 0 lock collisions or deadlocks observed.
  - Session state journal remained fully consistent across 16 sequential operations.
  - Reentrancy within identical thread/process contexts resolved cleanly.

---

## 3. Recommended Remediation Patches

### Patch 1: Robust Contract Parsing and Probe Execution (`src/plan_mode/execution_contract.py`)
```python
def parse_execution_contract(plan_text: str) -> tuple[Optional[ExecutionContract], list[str]]:
    # ... regex improvements to support indented markdown code fences:
    # r"(?m)^\s*##\s*Execution\s+Contract.*?
\s*```json\s*
(.*?)
\s*```"
    # ...
    # Defensive type extraction:
    raw_probe = data.get("probe")
    probe = raw_probe if isinstance(raw_probe, dict) else {}

    raw_cmds = data.get("verification_commands")
    verification_commands = [cmd for cmd in raw_cmds if isinstance(cmd, list)] if isinstance(raw_cmds, list) else []

    raw_artifacts = data.get("expected_artifacts")
    expected_artifacts = {str(k): (v if isinstance(v, dict) else {}) for k, v in raw_artifacts.items()} if isinstance(raw_artifacts, dict) else {}

    raw_invariants = data.get("workspace_invariants")
    workspace_invariants = [str(x) for x in raw_invariants] if isinstance(raw_invariants, list) else []

    raw_parity = data.get("parity_checks")
    parity_checks = [p for p in raw_parity if isinstance(p, dict)] if isinstance(raw_parity, list) else []

    raw_symbols = data.get("symbols")
    symbols = {str(k): (v if isinstance(v, dict) else {}) for k, v in raw_symbols.items()} if isinstance(raw_symbols, dict) else {}
```

### Patch 2: Top-Level Only Symbol Scanning in `_symbols_from_source`
```python
def _symbols_from_source(source: str) -> dict[str, set[str]]:
    tree = ast.parse(source)
    funcs: set[str] = set()
    classes: set[str] = set()
    vars_: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    vars_.add(target.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                vars_.add(node.target.id)

    return {"functions": funcs, "classes": classes, "variables": vars_}
```

### Patch 3: Zero-Padded Version Parsing
```python
def _version_tuple(v: str | None) -> tuple[int, int, int]:
    if not v:
        return (0, 0, 0)
    try:
        nums = [int(x) for x in re.findall(r"\d+", str(v))]
        padded = (nums + [0, 0, 0])[:3]
        return (padded[0], padded[1], padded[2])
    except Exception:
        return (0, 0, 0)
```

### Patch 4: Explicit Error on Unknown Checkpoint ID
```python
if checkpoint_id:
    cp = next((c for c in reversed(cps) if c.get("id") == checkpoint_id), None)
    if cp is None:
        raise ValueError(f"checkpoint '{checkpoint_id}' not found in session")
else:
    cp = cps[-1]
```

---

## 4. Summary Table of Audit Probes

| Subsystem / Feature | Probe Target | Status | Criticality |
| :--- | :--- | :--- | :--- |
| **Execution Contract** | Malformed JSON & `null` dictionary values | **FAIL (Crashed)** | High |
| **Execution Contract** | Indented markdown code fences | **FAIL (Missed)** | Medium |
| **Execution Contract** | Non-existent probe executable | **FAIL (Crashed)** | High |
| **Symbol Audit** | Local variables inside functions | **FAIL (False positives)** | High |
| **Symbol Audit** | Class methods & nested functions | **FAIL (False positives)** | High |
| **Symbol Audit** | Non-`.py` artifacts in symbol contract | **FAIL (False positives)** | Low |
| **AgentRewind** | Snapshot field deep-copy fidelity | **PASS** | Low |
| **AgentRewind** | Rewind to invalid checkpoint ID | **FAIL (Silent fallback)**| Medium |
| **AgentRewind** | `search(checkpoint_before=True)` | **PASS** | Low |
| **ACID Commits** | Best vs Committed separation | **PASS** | Low |
| **ACID Commits** | Out-of-bounds version indexing | **FAIL (IndexError)** | Medium |
| **RoT Experience Tree**| Perspective namespacing & isolation | **PASS** | Low |
| **RoT Experience Tree**| Environmental outcome calibration | **PASS** | Low |
| **Replanning Ladder** | Tier 4 runtime drift detection | **PASS** | Low |
| **Replanning Ladder** | Dictionary return key consistency | **FAIL (Key mismatch)** | Low |
| **Version Parsing** | Two-part versions (`"0.14"` vs `"0.14.0"`) | **FAIL (Tuple length)** | Medium |
| **Concurrency / Flock**| Multi-threaded & multi-process locking | **PASS** | Low |

---
