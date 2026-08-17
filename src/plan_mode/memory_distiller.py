"""Rule of Thought (RoT) Memory Distillation & Context Management.

Automates learning from planning failures, negative constraint distillation,
token-aware history folding, and 3-tier hierarchical replanning.

Literature grounding:
- RoT (2404.05449): Distill explicit reusable rules from failures to prevent oscillation.
- PPA-Plan (2601.11908): Pre-emptive negative constraints and distractor rejection.
- HIPIF (2606.10507): Token-aware context budgeting and history compression.
- Tiered Replanning Ladder (2605.25851) & RePLan (2401.04157): Multi-tier hierarchical replanning.
- SERP (2603.02772): Self-evolution and strategy upgrade loop.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .causal_validator import CausalFlaw, PlanAST


# ---------------------------------------------------------------------------
# 1. Rule of Thought (RoT) Distillation Engine (Structural Matching)
# ---------------------------------------------------------------------------

@dataclass
class RoTRule:
    """A distilled reusable rule learned from planning or execution failures."""
    rule_id: str
    trigger_condition: str
    forbidden_pattern: str
    remedy: str
    source_flaw_type: str
    predicate: str = ""
    affected_resource: str = ""
    perspective: str = "planning"
    hit_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


class RoTRuleBase:
    """Persistent rule repository that learns from failures and guards future plans."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.rules: dict[str, RoTRule] = {}
        if storage_path and storage_path.exists():
            self._load()

    def distill_from_flaws(self, flaws: list[dict[str, Any]], context_tag: str = "general",
                           perspective: str = "planning") -> list[RoTRule]:
        """Automatically distill negative rules from detected structural/causal flaws.

        `perspective` is the Tree-of-Experience (2608.09044) analytical path
        that produced the flaw. Rules are namespaced by perspective so that
        retrieval can follow the current plan's reasoning path instead of a
        single global pool.
        """
        new_rules = []
        for flaw in flaws:
            flaw_type = flaw.get("type", "flaw")
            detail = flaw.get("detail", "")
            remedy = flaw.get("remedy", "")

            pred_m = re.search(r"['\"]([\w:-]+(?:\(.*?\))?)['\"]", detail)
            predicate = pred_m.group(1) if pred_m else ""
            res_m = re.search(r"([\w./-]+\.\w{1,8})", detail)
            resource = res_m.group(1) if res_m else ""

            rule_key = f"{perspective}:{flaw_type}:{predicate or resource or detail[:20]}"
            rule_id = f"rot:{flaw_type}:{hashlib.sha256(rule_key.encode('utf-8')).hexdigest()[:8]}"
            if rule_id in self.rules:
                self.rules[rule_id].hit_count += 1
                self.rules[rule_id].confidence = min(1.0, self.rules[rule_id].confidence + 0.1)
                continue

            rule = RoTRule(
                rule_id=rule_id,
                trigger_condition=f"Context: {context_tag}",
                forbidden_pattern=detail,
                remedy=remedy,
                source_flaw_type=flaw_type,
                predicate=predicate,
                affected_resource=resource,
                perspective=perspective,
                hit_count=1,
                failure_count=1,
                confidence=1.0
            )
            self.rules[rule_id] = rule
            new_rules.append(rule)

        if self.storage_path:
            self._save()
        return new_rules

    def check_plan_violations(self, plan_text: str, ast: Optional[PlanAST] = None,
                             perspective: Optional[str] = None) -> list[dict[str, Any]]:
        """Check if a candidate plan violates any active distilled rules (structural matching).

        When `perspective` is provided, only rules distilled from that
        Tree-of-Experience perspective are consulted.
        """
        violations = []
        for rule_id, rule in self.rules.items():
            if perspective and rule.perspective != perspective:
                continue
            violation_found = False

            # 1. Structural check via AST if available
            if ast and rule.predicate:
                pred_clean = rule.predicate.lower().strip("'\"")
                for action in ast.actions:
                    if rule.source_flaw_type == "clobber_threat":
                        if any(d.positive_key == pred_clean for d in action.del_effects):
                            violation_found = True
                            break
                    elif rule.source_flaw_type == "unsatisfied_precondition":
                        if any(p.positive_key == pred_clean for p in action.preconditions):
                            if not any(a.id < action.id and any(eff.positive_key == pred_clean for eff in a.add_effects) for a in ast.actions):
                                violation_found = True
                                break

            # 2. Structural resource check
            if ast and rule.affected_resource and not violation_found:
                res_clean = rule.affected_resource.lower().strip("'\"")
                for action in ast.actions:
                    if res_clean in [inp.lower() for inp in action.inputs]:
                        if not any(a.id < action.id and res_clean in [out.lower() for out in a.outputs] for a in ast.actions):
                            violation_found = True
                            break

            # 3. Fallback pattern match
            if not violation_found and rule.forbidden_pattern and rule.forbidden_pattern.lower() in plan_text.lower():
                violation_found = True

            if violation_found:
                violations.append({
                    "rule_id": rule.rule_id,
                    "flaw_type": rule.source_flaw_type,
                    "violation": f"Structural rule violation [{rule.rule_id}]: {rule.source_flaw_type} on '{rule.predicate or rule.affected_resource or rule.forbidden_pattern[:30]}'",
                    "remedy": rule.remedy,
                    "confidence": rule.confidence
                })
        return violations

    def record_outcome(self, rule_id: str, *, success: bool, evidence: str = "") -> dict[str, Any]:
        """Update an experience-tree rule with environmental outcome feedback."""
        rule = self.rules.get(rule_id)
        if not rule:
            return {"error": f"unknown rule {rule_id}"}
        rule.hit_count += 1
        if success:
            rule.success_count += 1
        else:
            rule.failure_count += 1
        total = rule.success_count + rule.failure_count
        rule.confidence = round(rule.success_count / total, 3) if total else rule.confidence
        if self.storage_path:
            self._save()
        return {"rule_id": rule_id, "confidence": rule.confidence,
                "success_count": rule.success_count, "failure_count": rule.failure_count,
                "evidence": evidence[:200]}

    def tree_report(self) -> dict[str, Any]:
        """Summarize the experience tree grouped by perspective and flaw type."""
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for rule in self.rules.values():
            groups.setdefault(rule.perspective, {}).setdefault(rule.source_flaw_type, []).append({
                "rule_id": rule.rule_id,
                "predicate": rule.predicate,
                "resource": rule.affected_resource,
                "confidence": rule.confidence,
                "successes": rule.success_count,
                "failures": rule.failure_count,
            })
        return {"perspectives": len(groups), "rules": len(self.rules), "tree": groups}

    def _save(self):
        if not self.storage_path:
            return
        data = {
            r_id: {
                "rule_id": r.rule_id,
                "trigger_condition": r.trigger_condition,
                "forbidden_pattern": r.forbidden_pattern,
                "remedy": r.remedy,
                "source_flaw_type": r.source_flaw_type,
                "predicate": r.predicate,
                "affected_resource": r.affected_resource,
                "perspective": r.perspective,
                "hit_count": r.hit_count,
                "success_count": r.success_count,
                "failure_count": r.failure_count,
                "confidence": r.confidence,
                "created_at": r.created_at
            }
            for r_id, r in self.rules.items()
        }
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self):
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for r_id, d in data.items():
                self.rules[r_id] = RoTRule(**d)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Token-Aware Context Management (HIPIF)
# ---------------------------------------------------------------------------

class ContextBudgeter:
    """Budgets token usage and compresses superseded history to strictly honor max_context_tokens."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation (~4 chars per token)."""
        return max(1, len(text) // 4)

    @classmethod
    def session_token_count(cls, session: dict[str, Any]) -> int:
        """Estimate total tokens across all rounds in a session."""
        total = 0
        for r in session.get("rounds", []):
            total += cls.estimate_tokens(r.get("plan_text", ""))
            total += cls.estimate_tokens(str(r.get("critiques", [])))
        return total

    @classmethod
    def compress_history(cls, session: dict[str, Any], max_context_tokens: int = 4000, keep_last: int = 2) -> dict[str, Any]:
        """Compress old superseded planning rounds into concise semantic diffs until tokens <= max_context_tokens."""
        rounds = session.get("rounds", [])
        if len(rounds) <= 2:
            return session

        best_ver = session.get("best_version")
        last_ver = rounds[-1].get("version")

        # 1. Primary pass: fold all superseded rounds older than keep_last
        for i, r in enumerate(rounds):
            v = r.get("version", i + 1)
            is_best = (v == best_ver)
            is_recent = (i >= len(rounds) - keep_last)
            if is_best or is_recent or r.get("folded"):
                continue

            crit_ids = ", ".join([(c.get("id", str(c)) if isinstance(c, dict) else str(c)) for c in r.get("critiques", [])][:3]) or "none"
            summary_text = f"[folded: version {v}, score {r.get('score')}, delta {r.get('delta')}, critiques {crit_ids}]"
            r["folded"] = True
            r["summary"] = summary_text
            r["plan_text"] = summary_text

        # 2. Budget pass: if total tokens still exceed max_context_tokens, fold intermediate rounds (except best and latest)
        for i, r in enumerate(rounds):
            v = r.get("version", i + 1)
            if v == best_ver or v == last_ver or r.get("folded"):
                continue
            if cls.session_token_count(session) > max_context_tokens:
                crit_ids = ", ".join([(c.get("id", str(c)) if isinstance(c, dict) else str(c)) for c in r.get("critiques", [])][:3]) or "none"
                summary_text = f"[folded: version {v}, score {r.get('score')}, delta {r.get('delta')}, critiques {crit_ids}]"
                r["folded"] = True
                r["summary"] = summary_text
                r["plan_text"] = summary_text

        return session


# ---------------------------------------------------------------------------
# 3. 3-Tier Hierarchical Replanning Ladder
# ---------------------------------------------------------------------------

class ReplanningLadder:
    """Escalates replanning across four hierarchical tiers upon execution or validation failure.

    Tiers: L1 local repair, L2 subgraph replan, L3 global redraft, and
    L4 runtime drift recovery (2608.14109) for silent/behavioral drift.
    """

    DRIFT_SIGNALS = ("drift", "silent failure", "schema violation", "divergence",
                     "unexpected side effect", "behavioral drift", "irreversible")

    @staticmethod
    def determine_replan_tier(failed_task_id: int, error_message: str,
                              total_tasks: int, retry_count: int) -> dict[str, Any]:
        """Classify failure and return optimal replan scope (Tier 1 -> Tier 4)."""
        msg = error_message.lower()
        if any(sig in msg for sig in ReplanningLadder.DRIFT_SIGNALS):
            return {
                "tier": 4,
                "scope": "runtime_drift_recovery",
                "task_id": failed_task_id,
                "action": "classify_drift_and_restore_checkpoint",
                "description": (f"Tier 4 Drift Recovery: Classify drift near task {failed_task_id}, "
                                "restore the latest checkpoint if needed, and add a negative constraint")
            }
        if retry_count == 0 and ("timeout" in error_message.lower() or "retry" in error_message.lower() or "rate_limit" in error_message.lower()):
            return {
                "tier": 1,
                "scope": "local_task",
                "task_id": failed_task_id,
                "action": "adjust_parameters_and_retry",
                "description": f"Tier 1 Replan: Adjust task {failed_task_id} parameters without modifying DAG"
            }

        if retry_count <= 2 and failed_task_id <= total_tasks:
            return {
                "tier": 2,
                "scope": "subgraph_replan",
                "task_id": failed_task_id,
                "failed_task_id": failed_task_id,
                "action": "replan_subgraph_from_failure",
                "description": f"Tier 2 Replan: Preserve validated prefix (Tasks 1..{failed_task_id-1}), replan from Task {failed_task_id} onwards"
            }

        return {
            "tier": 3,
            "scope": "global_strategy_replan",
            "action": "upgrade_decomposition_strategy",
            "description": "Tier 3 Replan: Full strategy re-decomposition with reinforced negative constraints"
        }
