# AGENTS.md — Rigorous Engineering & Verification Contract

## 1. Project Overview & Architecture Intent (WHAT & WHY)
- **Repository**: `prime-agent-plan` (`plan` / `plan_mode`)
- **Purpose**: Canonical Plan IR, epistemic verification runtime, STRIPS causal simulation, AST search, and transactional execution recovery.
- **Core Principle**: Epistemic humility. Never claim an assumption is verified without an explicit, witnessed effect from an external tool or test.

---

## 2. Mandatory Non-Negotiable Operational Rules

### Rule 1: Zero Unverified Claims (The Grounding Mandate)
- Never state that code works, tests pass, or a task is complete without running the command in this turn and observing exit code 0.
- Never write tautological or self-approving tests (e.g. `assert True`, mocking the tested logic, or asserting hardcoded return values).
- When reporting results, always provide the exact command line, duration, and raw stdout summary.

### Rule 2: Strict Test-First (TDD) Contract
For all bug fixes, refactors, and feature additions:
1. **RED Phase**: Write an adversarial unit test that fails on current code. Run it and demonstrate the failure.
2. **GREEN Phase**: Implement only the minimal code necessary to make the test pass. Run tests and verify exit code 0.
3. **REFACTOR / AUDIT Phase**: Inspect `git diff` to remove all temporary logs, debug statements, and unused code.

### Rule 3: Single-Phase Scope Quarantine
- Work strictly on one gated subtask at a time.
- Never generate speculative scaffolding, empty mock classes, or fake implementations for future phases.
- If a user prompt spans multiple phases, execute only the active phase and stop at the phase gate.

### Rule 4: Repository-Wide Search Verification
- Never assume documentation or code comments match reality.
- Before declaring a phase done, use `grep_search` across all `.py` and `.md` files to audit for overclaim keywords (`guarantee`, `trusted runtime`, `isolated sandbox`, `mock`, `pass`).

---

## 3. Build, Verification & CI Commands (HOW)

```bash
# 1. Clean editable install with test dependencies
python -m pip install -e ".[test]"

# 2. Run full unit test suite
python -m pytest -q

# 3. Verify test portability (without optional planning corpus)
PLANNING_CORPUS=/nonexistent python -m pytest -q

# 4. Engine internal selfcheck
python -c "import plan; import plan_mode; res = plan_mode.selfcheck(run_pytest=False); assert res['ok']"

# 5. Synthetic metrics smoke validation
python benchmarks/run_ablations.py

# 6. CI Matrix inspection (via GitHub CLI)
gh run list --limit 3
gh run watch <run_id>
```

---

## 4. Phase Gate Protocol
Every phase must conclude with:
1. **Exact Files Modified** (from `git status`)
2. **Raw Test Output** (command + passing count)
3. **CI Matrix Status** (across Python 3.10–3.14)
4. **Explicit Gate Verdict**: `PHASE_GATE: PASS` (or `FAIL` with root cause).
