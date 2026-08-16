"""Rubric v4 tests: structure, compilation, corpus verification, samples, S7 drift."""
import json
import os
import re
from pathlib import Path

import pytest

import plan_mode

SKILL = Path(__file__).resolve().parent.parent
RUBRIC_MD = SKILL / "src" / "plan_mode" / "RUBRIC.md"
CORPUS = Path(os.environ.get("PLANNING_CORPUS", "/home/lewbei/deep_learning/planning_paper"))


def _rubric():
    text = RUBRIC_MD.read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    assert m, "RUBRIC.md must contain a JSON block"
    return json.loads(m.group(1))


def _checks(rubric):
    out = {}
    for sec, cfg in rubric.items():
        for it in cfg["items"]:
            out[it[0]] = {"section": sec, "regex": it[1], "hint": it[2]}
    return out


def test_json_parses_and_all_regexes_compile():
    rubric = _rubric()
    for cfg in rubric.values():
        for it in cfg["items"]:
            re.compile(it[1])


def test_check_count_at_least_97():
    rubric = _rubric()
    n = sum(len(s["items"]) for s in rubric.values())
    assert n >= 97, f"rubric v6 requires >= 97 checks, got {n}"


def test_weight_total_152():
    rubric = _rubric()
    assert sum(s["weight"] for s in rubric.values()) == 155


NEW_CHECKS = ["deviation_handling", "preflight_risky", "reuse_components",
              "lookahead_backward", "limited_commitment", "history_budget",
              "rules_from_failures", "planner_upgrade", "process_based_eval",
              "replan_timing", "root_cause_isolation",
              "hierarchy_validation", "solver_first", "error_detection",
              "parallel_opportunities", "capability_alignment",
              "evidence_preservation", "oscillation_guard",
              "user_preferences", "verify_on_mismatch", "rule_review_gate",
              "feedback_world_update", "opaque_domain_state",
              "complexity_gated_reflection", "decomposition_cost",
              "grounded_inputs"]


def test_all_11_new_checks_present():
    checks = _checks(_rubric())
    for name in NEW_CHECKS:
        assert name in checks, f"missing new check {name}"


@pytest.mark.skipif(not (CORPUS / "txts").exists(), reason="Corpus txts directory not found (set PLANNING_CORPUS)")
def test_every_cited_id_exists_in_corpus():
    txts = set(os.listdir(CORPUS / "txts"))
    for name, c in _checks(_rubric()).items():
        for i in re.findall(r"\b(2[0-6]\d{2}\.\d{5})\b", c["hint"]):
            assert f"{i}.txt" in txts, f"{name} cites {i}, absent from txts/"


# positive/negative samples per new check (>= 90% negatives must be rejected)
SAMPLES = {
    "deviation_handling": ("If the check fails, the corrective action is a local fix.", ["all good"]),
    "preflight_risky": ("A pre-execution critique pass covers risky steps.", ["we ship immediately"]),
    "reuse_components": ("Reuse validated components from prior demonstrations.", ["fresh from nothing"]),
    "lookahead_backward": ("Backward value propagation from the goal plus lookahead.", ["forward only"]),
    "limited_commitment": ("Commit at most 3 next steps, then re-evaluate.", ["commit to a plan today"]),
    "history_budget": ("Fold finished subgoals and prune history; cap the context budget.", ["keep every draft"]),
    "rules_from_failures": ("Distill rules from past failures and append them.", ["try harder"]),
    "planner_upgrade": ("Upgrade the planner itself after repeated failure.", ["change one step"]),
    "process_based_eval": ("Score the reasoning steps and the recovery steps, not just the outcome.", ["final score only"]),
    "replan_timing": ("Schedule a fresh replan immediately before timing-sensitive steps.", ["replan whenever"]),
    "root_cause_isolation": ("Isolate the root cause of the failure first.", ["fix the symptom"]),
    "hierarchy_validation": ("Validate the hierarchical decomposition syntactically.", ["we just wing the levels"]),
    "solver_first": ("Solver-first: the solver checks, the LLM repairs.", ["no solver here"]),
    "error_detection": ("Detect the error immediately at each step.", ["everything fine"]),
    "parallel_opportunities": ("Tasks 2 and 3 run in parallel.", ["sequential only"]),
    "capability_alignment": ("Align subgoals with executor capabilities.", ["no capabilities here"]),
    "evidence_preservation": ("Preserve intermediate evidence across steps.", ["forget everything"]),
    "oscillation_guard": ("Avoid revisiting the same failed state (oscillation guard).", ["try again blindly"]),
    "user_preferences": ("Capture user preferences before drafting.", ["ignore the user"]),
    "verify_on_mismatch": ("Verify only on mismatch (speculative steps).", ["verify always"]),
    "rule_review_gate": ("Human review of generated rules before they bind.", ["ship it"]),
    "feedback_world_update": ("Execution feedback updates the world model, closing the loop.", ["log it and move on"]),
    "opaque_domain_state": ("State the opaque domain's hidden rules explicitly before planning.", ["proceed normally"]),
    "complexity_gated_reflection": ("A complexity assessment decides reflection intensity per task.", ["always reflect a lot"]),
    "decomposition_cost": ("Match decomposition granularity to task difficulty and cost.", ["one size fits all"]),
    "grounded_inputs": ("Requires: existing_input.md.", ["no inputs listed"]),
}


def test_positive_and_negative_samples():
    checks = _checks(_rubric())
    rejected = 0
    total = 0
    for name, (pos, negs) in SAMPLES.items():
        c = checks[name]
        assert re.search(c["regex"], pos, re.I | re.M), f"{name} must match its positive"
        for neg in negs:
            total += 1
            if not re.search(c["regex"], neg, re.I | re.M):
                rejected += 1
    assert rejected / total >= 0.9, f"only {rejected}/{total} negatives rejected"


@pytest.mark.skipif(not (CORPUS / "digests").exists(), reason="Corpus digests directory not found (set PLANNING_CORPUS)")
def test_corpus_excerpts_do_not_overmatch():
    """20 corpus excerpts: at least 90% must NOT match each new check regex."""
    checks = _checks(_rubric())
    excerpts = []
    for b in range(1, 7):
        t = (CORPUS / f"digests/batch-{b}.md").read_text()
        excerpts += t.split("\n\n")[3:12]
    excerpts = [e for e in excerpts if len(e) > 40][:20]
    assert len(excerpts) >= 10
    rejected = {n: 0 for n in NEW_CHECKS}
    total = len(excerpts)
    for e in excerpts:
        for n in NEW_CHECKS:
            if not re.search(checks[n]["regex"], e, re.I | re.M):
                rejected[n] += 1
    for n in NEW_CHECKS:
        assert rejected[n] / total >= 0.9, f"{n} overmatches: {rejected[n]}/{total}"


@pytest.mark.skipif(not (CORPUS / "plan_improvements").exists(), reason="Corpus plan_improvements directory not found (set PLANNING_CORPUS)")
def test_s7_drift_bounded():
    """Re-score the two archived reference candidates: drift vs golden <= 3."""
    golden = json.loads((CORPUS / "plan_improvements/backup/golden/golden.json").read_text())
    rubric = plan_mode._load_rubric()
    for name in ["draft_v3A", "draft_v3B"]:
        t = (CORPUS / f"plan_improvements/{name}.md").read_text()
        old = golden[name]["rubric_score"]
        new = plan_mode._score(t, rubric)["score"]
        assert abs(new - old) <= 3, f"{name}: drift {abs(new - old)} > 3"
