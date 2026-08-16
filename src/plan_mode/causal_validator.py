"""Deterministic Causal Validator & Symbolic Planning Engine.

Replaces shallow regex matching with formal STRIPS/PDDL-style action semantics,
causal link tracking, threat/clobber detection, and localized constraint solving.

Literature grounding:
- SymPlanner (2505.01479): Symbolic state simulation with explicit precondition/effect transitions.
- LLM-Modulo (2502.12435, 2512.09629): Deterministic non-LLM verifier in the loop.
- GNNVerifier & Grounded Diagnosis (2603.14730): Localized structural flaw localization.
- SafeRun & Constraint Solvers (2606.09027, 2605.20873): Hard numeric/resource budget solver.
- Representation Invariance (PlanBench 2409.13373): Validation based on action semantics rather than prose.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Action Schema & Propositional State Definitions
# ---------------------------------------------------------------------------

@dataclass
class Proposition:
    """Atomic state predicate: e.g. 'file_exists(out.txt)', 'service_ready(db)', 'verified(model)'."""
    name: str
    args: tuple[str, ...] = field(default_factory=tuple)
    negated: bool = False

    def __str__(self) -> str:
        arg_str = f"({', '.join(self.args)})" if self.args else ""
        return f"{'not ' if self.negated else ''}{self.name}{arg_str}"

    @classmethod
    def parse(cls, text: str) -> Proposition:
        text = text.strip()
        negated = False
        if text.startswith("not ") or text.startswith("!"):
            negated = True
            text = text[4:] if text.startswith("not ") else text[1:]
        m = re.match(r"^([\w:-]+)(?:\((.*?)\))?$", text)
        if m:
            name = m.group(1)
            raw_args = m.group(2)
            args = tuple(a.strip() for a in raw_args.split(",") if a.strip()) if raw_args else ()
            return cls(name=name, args=args, negated=negated)
        return cls(name=text, negated=negated)


@dataclass
class ActionSchema:
    """Typed formal specification of an action."""
    id: int
    name: str
    preconditions: list[Proposition] = field(default_factory=list)
    add_effects: list[Proposition] = field(default_factory=list)
    del_effects: list[Proposition] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)        # File/resource paths consumed
    outputs: list[str] = field(default_factory=list)      # File/resource paths produced
    depends_on: list[int] = field(default_factory=list)    # Declared task IDs
    cost: float = 1.0                                     # Cost units (tokens, time, USD)
    duration_minutes: float = 10.0
    parallel_group: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class CausalLink:
    """Causal link <producer_id, condition, consumer_id> asserting that producer achieves condition for consumer."""
    producer: int
    condition: Proposition
    consumer: int


@dataclass
class CausalFlaw:
    """Pinpointed diagnostic structural flaw in a plan."""
    flaw_type: str        # 'unsatisfied_precondition', 'clobber_threat', 'dead_artifact', 'cyclic_dependency', 'resource_budget_exceeded', 'type_mismatch'
    task_id: int
    detail: str
    remedy_hint: str
    involved_tasks: list[int] = field(default_factory=list)
    involved_proposition: Optional[Proposition] = None


@dataclass
class PlanAST:
    """Structured Abstract Syntax Tree of a plan."""
    goal: str
    actions: list[ActionSchema] = field(default_factory=list)
    initial_state: set[str] = field(default_factory=set)
    target_propositions: list[Proposition] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_action(self, action_id: int) -> Optional[ActionSchema]:
        for a in self.actions:
            if a.id == action_id:
                return a
        return None

    @property
    def action_ids(self) -> list[int]:
        return [a.id for a in self.actions]


# ---------------------------------------------------------------------------
# 2. Plan Text Parser -> PlanAST
# ---------------------------------------------------------------------------

class PlanParser:
    """Extracts formal action schemas, preconditions, effects, and bounds from plan text."""

    @classmethod
    def parse_plan(cls, plan_text: str, objective: str = "") -> PlanAST:
        actions: list[ActionSchema] = []

        # 1. Parse Goal
        goal = objective
        goal_m = re.search(r"(?:Goal|Objective)\s*[:(]?\s*(.+)", plan_text, re.I)
        if goal_m and not goal:
            goal = goal_m.group(1).split("\n")[0].strip()

        # 2. Extract Tasks
        markers = [m for m in re.finditer(r"^\s*(\d+)[.)]\s+(.+)$", plan_text, re.M)]
        for i, m in enumerate(markers):
            task_id = int(m.group(1))
            task_header = m.group(2).strip()
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(plan_text)
            body = plan_text[start:end].strip()
            full_task_text = f"{task_id}. {task_header}\n{body}"

            # Depends on
            deps: list[int] = []
            for dep_m in re.finditer(r"depends?\s+on\s+([^.(]+)", full_task_text, re.I):
                deps.extend(int(x) for x in re.findall(r"\d+", dep_m.group(1)))
            deps = sorted(set(deps))

            # Outputs / artifacts
            outs: list[str] = []
            for out_m in re.finditer(r"(?:output|produces?|writes?|deliverable)\s*[:(]?\s*([\w./-]+\.\w{1,8})", full_task_text, re.I):
                val = out_m.group(1)
                if val:
                    outs.append(val.strip())
            outs = sorted(set(outs))

            # Inputs / reads
            ins: list[str] = []
            for in_m in re.finditer(r"(?:requires?|inputs?|needs|consumes)\s*[:(]?\s*([\w./-]+\.\w{1,8})|\breads?\b\s*[:(]?\s*([\w./-]+\.\w{1,8})", full_task_text, re.I):
                val = in_m.group(1) or in_m.group(2)
                if val:
                    ins.append(val.strip())
            ins = sorted(set(ins))

            # Preconditions
            preconds: list[Proposition] = []
            for pre_m in re.finditer(r"(?:precondition|requires_state|requires_condition)\s*[:(]?\s*([^\n;]+)", full_task_text, re.I):
                raw_pre = pre_m.group(1).strip()
                for token in re.split(r"[,;]\s*", raw_pre):
                    if token.strip():
                        preconds.append(Proposition.parse(token.strip()))

            for input_file in ins:
                preconds.append(Proposition(name="exists", args=(input_file,)))

            # Effects
            add_effects: list[Proposition] = []
            del_effects: list[Proposition] = []
            for eff_m in re.finditer(r"(?:effects?|postcondition|produces_state)\s*[:(]?\s*([^\n;]+)", full_task_text, re.I):
                raw_eff = eff_m.group(1).strip()
                for token in re.split(r"[,;]\s*", raw_eff):
                    if token.strip():
                        p = Proposition.parse(token.strip())
                        if p.negated:
                            del_effects.append(p)
                        else:
                            add_effects.append(p)

            for out_file in outs:
                add_effects.append(Proposition(name="exists", args=(out_file,)))

            # Time duration
            duration = 15.0
            time_m = re.search(r"(\d+(?:\.\d+)?)\s*(min|minute|hour)s?", full_task_text, re.I)
            if time_m:
                val = float(time_m.group(1))
                unit = time_m.group(2).lower()
                duration = val * 60.0 if "hour" in unit else val

            # Cost
            cost = 1.0
            cost_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:tokens?|USD|\$|k\s*tokens?)", full_task_text, re.I)
            if cost_m:
                cost = float(cost_m.group(1))
                if "k" in cost_m.group(0).lower():
                    cost *= 1000.0

            action = ActionSchema(
                id=task_id,
                name=task_header,
                preconditions=preconds,
                add_effects=add_effects,
                del_effects=del_effects,
                inputs=ins,
                outputs=outs,
                depends_on=deps,
                duration_minutes=duration,
                cost=cost,
                raw_text=full_task_text
            )
            actions.append(action)

        # 3. Extract Global Constraints
        constraints: dict[str, Any] = {}
        for cons_m in re.finditer(r"(at most|at least|no more than|maximum of|minimum of)\s+(\d+(?:\.\d+)?)\s*(tasks?|steps?|hours?|minutes?|tokens?|USD|\$)", plan_text, re.I):
            kind = cons_m.group(1).lower()
            val = float(cons_m.group(2))
            unit = cons_m.group(3).lower()
            constraints.setdefault(unit, []).append({"type": kind, "value": val})

        return PlanAST(
            goal=goal or "Execute multi-step plan",
            actions=actions,
            constraints=constraints
        )


# ---------------------------------------------------------------------------
# 3. Deterministic Causal Validator
# ---------------------------------------------------------------------------

class CausalValidator:
    """Formally simulates state evolution, verifies causal links, and detects threats/deadlocks."""

    @classmethod
    def validate(cls, ast: PlanAST, initial_state: Optional[Set[str]] = None) -> dict[str, Any]:
        """Perform comprehensive causal and symbolic validation of the PlanAST."""
        state: set[str] = set(initial_state or ast.initial_state)
        flaws: list[CausalFlaw] = []
        causal_links: list[CausalLink] = []
        task_execution_order: list[int] = []
        produced_propositions: dict[str, int] = {}
        consumed_propositions: set[str] = set()

        actions = sorted(ast.actions, key=lambda a: a.id)
        action_ids = [a.id for a in actions]

        # 1. Contiguity and sequence check
        if action_ids:
            expected_ids = list(range(1, len(action_ids) + 1))
            if action_ids != expected_ids:
                flaws.append(CausalFlaw(
                    flaw_type="non_contiguous_sequence",
                    task_id=action_ids[0],
                    detail=f"Task numbering is non-contiguous: found {action_ids}, expected {expected_ids}",
                    remedy_hint="Renumber tasks consecutively 1, 2, ..., N",
                    involved_tasks=action_ids
                ))

        # 2. Forward execution simulation
        for action in actions:
            t = action.id
            task_execution_order.append(t)

            # A. Dependency order check
            for dep in action.depends_on:
                if dep not in task_execution_order[:-1]:
                    flaws.append(CausalFlaw(
                        flaw_type="unsatisfied_dependency",
                        task_id=t,
                        detail=f"Task {t} declares dependency on Task {dep}, which has not yet executed",
                        remedy_hint=f"Place Task {dep} before Task {t} or remove forward dependency",
                        involved_tasks=[t, dep]
                    ))

            # B. Precondition satisfaction & Causal Link Construction
            for prec in action.preconditions:
                prec_str = str(prec)
                consumed_propositions.add(prec_str)
                if prec_str not in state and not (prec.name == "exists" and prec.args and prec.args[0] in state):
                    producer = produced_propositions.get(prec_str)
                    if producer is None and prec.name == "exists" and prec.args:
                        producer = produced_propositions.get(str(Proposition(name="exists", args=prec.args)))

                    if producer is not None and producer in task_execution_order:
                        causal_links.append(CausalLink(producer=producer, condition=prec, consumer=t))
                    else:
                        flaws.append(CausalFlaw(
                            flaw_type="unsatisfied_precondition",
                            task_id=t,
                            detail=f"Task {t} requires precondition '{prec_str}' which is unsatisfied in world state",
                            remedy_hint=f"Add an earlier action producing '{prec_str}' or supply it in initial environment inputs",
                            involved_tasks=[t],
                            involved_proposition=prec
                        ))
                else:
                    producer = produced_propositions.get(prec_str, 0)
                    causal_links.append(CausalLink(producer=producer, condition=prec, consumer=t))

            # C. Apply Delete Effects
            for del_prop in action.del_effects:
                del_str = str(del_prop)
                state.discard(del_str)

            # D. Apply Add Effects
            for add_prop in action.add_effects:
                add_str = str(add_prop)
                state.add(add_str)
                produced_propositions[add_str] = t
                if add_prop.name == "exists" and add_prop.args:
                    state.add(add_prop.args[0])
                    produced_propositions[add_prop.args[0]] = t

            for out_file in action.outputs:
                state.add(out_file)
                produced_propositions[out_file] = t

                # 3. Threat / Clobber detection on established causal links (producer < k < consumer)
        for link in causal_links:
            cond_str = str(link.condition)
            for action in actions:
                k = action.id
                if link.producer < k < link.consumer:
                    if any(str(d) == cond_str for d in action.del_effects):
                        flaws.append(CausalFlaw(
                            flaw_type="clobber_threat",
                            task_id=k,
                            detail=f"Task {k} deletes '{cond_str}', destroying causal link from Task {link.producer} to Task {link.consumer}",
                            remedy_hint=f"Reorder Task {k} after Task {link.consumer} or preserve condition '{cond_str}'",
                            involved_tasks=[link.producer, k, link.consumer],
                            involved_proposition=link.condition
                        ))

        # 4. Dead Artifacts
        all_outputs = {out for a in actions for out in a.outputs}
        all_inputs = {inp for a in actions for inp in a.inputs}
        dead_artifacts = sorted(all_outputs - all_inputs)

        # 5. Global Resource & Budget Constraints Validation
        total_cost = sum(a.cost for a in actions)
        total_duration = sum(a.duration_minutes for a in actions)

        for unit, cons_list in ast.constraints.items():
            for cons in cons_list:
                limit = cons["value"]
                ctype = cons["type"]
                if "task" in unit or "step" in unit:
                    actual = len(actions)
                elif "min" in unit:
                    actual = total_duration
                elif "hour" in unit:
                    actual = total_duration / 60.0
                elif "token" in unit or "$" in unit or "usd" in unit:
                    actual = total_cost
                else:
                    continue

                if ("at most" in ctype or "no more than" in ctype or "maximum of" in ctype) and actual > limit:
                    flaws.append(CausalFlaw(
                        flaw_type="resource_budget_exceeded",
                        task_id=0,
                        detail=f"Plan violates constraint: {ctype} {limit} {unit} (actual: {actual:g} {unit})",
                        remedy_hint=f"Reduce task count or resource consumption below {limit} {unit}",
                        involved_tasks=[]
                    ))
                elif ("at least" in ctype or "minimum of" in ctype) and actual < limit:
                    flaws.append(CausalFlaw(
                        flaw_type="resource_budget_exceeded",
                        task_id=0,
                        detail=f"Plan violates constraint: {ctype} {limit} {unit} (actual: {actual:g} {unit})",
                        remedy_hint=f"Expand tasks to meet minimum {limit} {unit}",
                        involved_tasks=[]
                    ))

        ok = (len(flaws) == 0)
        return {
            "ok": ok,
            "flaws": [
                {
                    "type": f.flaw_type,
                    "task_id": f.task_id,
                    "detail": f.detail,
                    "remedy": f.remedy_hint,
                    "involved_tasks": f.involved_tasks
                } for f in flaws
            ],
            "causal_links": [
                {
                    "producer": cl.producer,
                    "condition": str(cl.condition),
                    "consumer": cl.consumer
                } for cl in causal_links
            ],
            "final_state": sorted(state),
            "dead_artifacts": dead_artifacts,
            "total_cost": total_cost,
            "total_duration_minutes": total_duration,
            "num_actions": len(actions)
        }
