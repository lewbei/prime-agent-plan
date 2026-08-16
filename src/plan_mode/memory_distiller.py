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
# 1. Rule of Thought (RoT) Distillation Engine
# ---------------------------------------------------------------------------

@dataclass
class RoTRule:
    """A distilled reusable rule learned from planning or execution failures."""
    rule_id: str
    trigger_condition: str
    forbidden_pattern: str
    remedy: str
    source_flaw_type: str
    hit_count: int = 0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


class RoTRuleBase:
    """Persistent rule repository that learns from failures and guards future plans."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.rules: dict[str, RoTRule] = {}
        if storage_path and storage_path.exists():
            self._load()

    def distill_from_flaws(self, flaws: list[dict[str, Any]], context_tag: str = "general") -> list[RoTRule]:
        """Automatically distill negative rules from detected structural/causal flaws."""
        new_rules = []
        for flaw in flaws:
            flaw_type = flaw.get("type", "flaw")
            detail = flaw.get("detail", "")
            remedy = flaw.get("remedy", "")

            rule_id = f"rot:{flaw_type}:{hashlib.sha256(detail.encode('utf-8')).hexdigest()[:8]}"
            if rule_id in self.rules:
                self.rules[rule_id].hit_count += 1
                continue

            rule = RoTRule(
                rule_id=rule_id,
                trigger_condition=f"Context: {context_tag}",
                forbidden_pattern=detail,
                remedy=remedy,
                source_flaw_type=flaw_type,
                hit_count=1
            )
            self.rules[rule_id] = rule
            new_rules.append(rule)

        if self.storage_path:
            self._save()
        return new_rules

    def check_plan_violations(self, plan_text: str, ast: Optional[PlanAST] = None) -> list[dict[str, str]]:
        """Check if a candidate plan violates any active distilled rules."""
        violations = []
        for rule_id, rule in self.rules.items():
            if rule.forbidden_pattern and rule.forbidden_pattern.lower() in plan_text.lower():
                violations.append({
                    "rule_id": rule.rule_id,
                    "flaw_type": rule.source_flaw_type,
                    "violation": f"Violates distilled rule {rule.rule_id}: {rule.forbidden_pattern}",
                    "remedy": rule.remedy
                })
        return violations

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
                "hit_count": r.hit_count,
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
    """Escalates replanning across three hierarchical tiers upon execution or validation failure."""

    @staticmethod
    def determine_replan_tier(failed_task_id: int, error_message: str,
                              total_tasks: int, retry_count: int) -> dict[str, Any]:
        """Classify failure and return optimal replan scope (Tier 1 -> Tier 2 -> Tier 3)."""
        if retry_count == 0 and ("timeout" in error_message.lower() or "retry" in error_message.lower() or "rate_limit" in error_message.lower()):
            return {
                "tier": 1,
                "scope": "local_task",
                "task_id": failed_task_id,
                "action": "adjust_parameters_and_retry",
                "description": f"Tier 1 Replan: Adjust task {failed_task_id} parameters without modifying DAG"
            }

        if retry_count <= 2 and failed_task_id < total_tasks:
            return {
                "tier": 2,
                "scope": "subgraph_replan",
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
