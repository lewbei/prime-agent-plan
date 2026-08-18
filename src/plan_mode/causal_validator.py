"""Deterministic Causal Validator & Symbolic Planning Engine.

Formal STRIPS/PDDL-style action semantics, causal link construction,
closed-world negation, threat/clobber detection, and localized flaw diagnosis.

Literature grounding:
- SymPlanner (2505.01479): Symbolic state simulation with explicit precondition/effect transitions.
- LLM-Modulo (2502.12435, 2512.09629): Deterministic non-LLM verifier in the loop.
- GNNVerifier & Grounded Diagnosis (2603.14730): Localized structural flaw localization.
- SafeRun & Constraint Solvers (2606.09027, 2605.20873): Hard numeric/resource budget solver.
- Representation Invariance (PlanBench 2409.13373): Validation based on action semantics.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Action Schema & Propositional State Definitions
# ---------------------------------------------------------------------------

@dataclass
class Proposition:
    """Atomic state predicate: e.g. 'file_exists(out.txt)', 'service_ready(db)', 'dirty(cache)'."""
    name: str
    args: tuple[str, ...] = field(default_factory=tuple)
    negated: bool = False

    @property
    def positive_key(self) -> str:
        """Normalized positive representation: name(arg1, arg2)."""
        arg_str = f"({', '.join(self.args)})" if self.args else ""
        return f"{self.name}{arg_str}".lower()

    def __str__(self) -> str:
        arg_str = f"({', '.join(self.args)})" if self.args else ""
        return f"{'not ' if self.negated else ''}{self.name}{arg_str}"

    @classmethod
    def parse(cls, text: str) -> Proposition:
        text = text.strip().rstrip(".,;")
        negated = False
        if text.startswith("not ") or text.startswith("!"):
            negated = True
            text = text[4:] if text.startswith("not ") else text[1:]
        m = re.match(r"^([\w:-]+)(?:\((.*?)\))?$", text)
        if m:
            name = m.group(1).strip()
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
    flaw_type: str        # 'unsatisfied_precondition', 'clobber_threat', 'dead_artifact', 'cyclic_dependency', 'resource_budget_exceeded', 'type_mismatch', 'unreachable_goal'
    task_id: int
    detail: str
    remedy_hint: str
    involved_tasks: list[int] = field(default_factory=list)
    involved_proposition: Optional[Proposition] = None


@dataclass
class PredicateSignature:
    """SymPlanner-style typed predicate signature (2505.01479).

    The paper models a planning problem as P = (F, A, I, G) where F is a
    set of ground atoms over typed predicates. This class makes that type
    information explicit for validation.
    """
    name: str
    arity: int
    arg_types: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, text: str) -> PredicateSignature:
        text = text.strip()
        m = re.match(r"^([\w:-]+)\((.*?)\)$", text)
        if not m:
            return cls(name=text, arity=0, arg_types=())
        raw_args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        arg_types: list[str] = []
        for raw in raw_args:
            if ":" in raw:
                arg_types.append(raw.rsplit(":", 1)[1].strip())
            else:
                arg_types.append("object")
        return cls(name=m.group(1), arity=len(raw_args), arg_types=tuple(arg_types))


@dataclass
class PlanAST:
    """Structured Abstract Syntax Tree of a plan."""
    goal: str
    actions: list[ActionSchema] = field(default_factory=list)
    initial_state: set[str] = field(default_factory=set)
    target_propositions: list[Proposition] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    predicate_signatures: list[PredicateSignature] = field(default_factory=list)
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
    def _clean_token(cls, token: str) -> str:
        t = token.strip()
        t = re.split(r"(?:\.\s+|\n|(?:\s+(?:output|inputs?|effects?|depends|time|cost)\s*:))", t, flags=re.I)[0]
        return t.strip().rstrip(".,;")

    @classmethod
    def parse_plan(cls, plan_text: str, objective: str = "") -> PlanAST:
        actions: list[ActionSchema] = []

        # 1. Parse Goal & Explicit Desired State
        goal = objective
        goal_m = re.search(r"(?:Goal|Objective)\s*[:(]?\s*([^\n]+)", plan_text, re.I)
        if goal_m and not goal:
            goal = goal_m.group(1).strip()

        target_props: list[Proposition] = []
        targets_m = re.search(r"(?:Desired State|Target Propositions?|Final State)\s*[:(]?\s*([^\n]+)", plan_text, re.I)
        if targets_m:
            for tok in re.split(r"[,;]\s*", targets_m.group(1)):
                clean_tok = cls._clean_token(tok)
                if clean_tok:
                    target_props.append(Proposition.parse(clean_tok))

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
            for dep_m in re.finditer(r"depends?\s+on\s+([^.(;\n]+)", full_task_text, re.I):
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

            # Preconditions (Preconditions: or Precondition:)
            preconds: list[Proposition] = []
            for pre_m in re.finditer(r"(?:preconditions?|requires_state|requires_condition)\s*[:(]?\s*([^\n]+)", full_task_text, re.I):
                raw_pre = pre_m.group(1).strip()
                for token in re.split(r"[,;]\s*", raw_pre):
                    clean = cls._clean_token(token)
                    if clean:
                        preconds.append(Proposition.parse(clean))

            for input_file in ins:
                preconds.append(Proposition(name="exists", args=(input_file,)))

            # Effects (Effects: or Effect:)
            add_effects: list[Proposition] = []
            del_effects: list[Proposition] = []
            for eff_m in re.finditer(r"(?:effects?|postconditions?|produces_state)\s*[:(]?\s*([^\n]+)", full_task_text, re.I):
                raw_eff = eff_m.group(1).strip()
                for token in re.split(r"[,;]\s*", raw_eff):
                    clean = cls._clean_token(token)
                    if clean:
                        p = Proposition.parse(clean)
                        if p.negated:
                            del_effects.append(p)
                        else:
                            add_effects.append(p)

            for del_m in re.finditer(r"(?:deletes?|removes?)\s*[:(]?\s*([^\n]+)", full_task_text, re.I):
                raw_del = del_m.group(1).strip()
                for token in re.split(r"[,;]\s*", raw_del):
                    clean = cls._clean_token(token)
                    if clean:
                        p = Proposition.parse(clean)
                        del_effects.append(p)

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

        # 2b. Parse SymPlanner-style typed predicate signatures
        predicate_signatures: list[PredicateSignature] = []
        for sig_m in re.finditer(r"^\s*(?:#{1,6}\s*)?(?:Predicate\s+Signature|Signature|Predicate)\s*:\s*(\w[\w:-]*\([^\n]*\))", plan_text, re.I | re.M):
            try:
                predicate_signatures.append(PredicateSignature.parse(sig_m.group(1)))
            except Exception:
                pass

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
            target_propositions=target_props,
            constraints=constraints,
            predicate_signatures=predicate_signatures
        )


def validate_typed_atom(prop: Proposition, signature: PredicateSignature) -> list[str]:
    """Validate one ground atom against a typed predicate signature."""
    errors: list[str] = []
    if prop.name.lower() != signature.name.lower():
        return errors  # signature does not govern this proposition
    if len(prop.args) != signature.arity:
        errors.append(f"predicate {signature.name}: expected arity {signature.arity}, got {len(prop.args)} for {prop}")
    for i, arg in enumerate(prop.args):
        declared = signature.arg_types[i] if i < len(signature.arg_types) else "object"
        if ":" in arg:
            actual = arg.rsplit(":", 1)[1].strip()
            if actual.lower() != declared.lower():
                errors.append(f"predicate {signature.name}: argument {arg!r} has type {actual!r}, expected {declared!r}")
    return errors


# ---------------------------------------------------------------------------
# 3. Deterministic Causal Validator (Closed-World Semantics)
# ---------------------------------------------------------------------------

class CausalValidator:
    """Formally simulates state evolution, verifies causal links, and detects threats/deadlocks."""

    @classmethod
    def validate(cls, ast: PlanAST, initial_state: Optional[Set[str]] = None) -> dict[str, Any]:
        """Perform comprehensive causal and symbolic validation of the PlanAST with closed-world negation."""
        state: set[str] = set()
        raw_init = set(initial_state or ast.initial_state)
        for s in raw_init:
            state.add(s.strip().lower())
            p = Proposition.parse(s)
            state.add(p.positive_key)
            if p.name == "exists" and p.args:
                state.add(p.args[0].lower())
            elif "." in s:
                state.add(f"exists({s.lower()})")

        flaws: list[CausalFlaw] = []
        causal_links: list[CausalLink] = []
        task_execution_order: list[int] = []
        produced_propositions: dict[str, int] = {k: 0 for k in state}

        actions = sorted(ast.actions, key=lambda a: a.id)
        action_ids = [a.id for a in actions]

        # 0. Typed predicate validation (SymPlanner 2505.01479)
        sigs = {sig.name.lower(): sig for sig in ast.predicate_signatures}
        if sigs:
            for action in actions:
                for prop in list(action.preconditions) + list(action.add_effects) + list(action.del_effects):
                    sig = sigs.get(prop.name.lower())
                    if sig is None:
                        continue
                    for err in validate_typed_atom(prop, sig):
                        flaws.append(CausalFlaw(
                            flaw_type="type_mismatch", task_id=action.id,
                            detail=f"Task {action.id}: {err}",
                            remedy_hint=f"Fix argument count or type to match {sig.name}/{sig.arity}",
                            involved_tasks=[action.id], involved_proposition=prop
                        ))

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

        # 2. Cycle Detection
        adj: dict[int, list[int]] = {a.id: a.depends_on for a in actions}
        visited: set[int] = set()
        rec_stack: set[int] = set()

        def _dfs_cycle(u: int, path: list[int]) -> bool:
            visited.add(u)
            rec_stack.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    if _dfs_cycle(v, path + [v]):
                        return True
                elif v in rec_stack:
                    flaws.append(CausalFlaw(
                        flaw_type="cyclic_dependency",
                        task_id=u,
                        detail=f"Dependency cycle detected involving task chain: {path + [v]}",
                        remedy_hint="Eliminate cyclic dependency edges in task DAG",
                        involved_tasks=path + [v]
                    ))
                    return True
            rec_stack.discard(u)
            return False

        for a in actions:
            if a.id not in visited:
                _dfs_cycle(a.id, [a.id])

        # 3. Forward execution simulation
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

            # B. Precondition satisfaction & Causal Links
            for prec in action.preconditions:
                pkey = prec.positive_key
                raw_target = prec.args[0].lower() if (prec.name == "exists" and prec.args) else pkey

                producer = produced_propositions.get(pkey, produced_propositions.get(raw_target))

                if producer is not None and producer in task_execution_order:
                    causal_links.append(CausalLink(producer=producer, condition=prec, consumer=t))

                if prec.negated:
                    if pkey in state or raw_target in state:
                        flaws.append(CausalFlaw(
                            flaw_type="unsatisfied_precondition",
                            task_id=t,
                            detail=f"Task {t} requires '{prec}', but '{pkey}' currently holds in world state",
                            remedy_hint=f"Add a preceding action to delete/clean '{pkey}' before Task {t}",
                            involved_tasks=[t],
                            involved_proposition=prec
                        ))
                else:
                    # Positive precondition: check for clobber threats first
                    if producer is not None:
                        clobberers = [a.id for a in actions if producer < a.id < t and any(d.positive_key == pkey for d in a.del_effects)]
                        if clobberers:
                            flaws.append(CausalFlaw(
                                flaw_type="clobber_threat",
                                task_id=clobberers[0],
                                detail=f"Task {clobberers[0]} deletes '{prec}', destroying causal link from Task {producer} to Task {t}",
                                remedy_hint=f"Reorder Task {clobberers[0]} after Task {t} or preserve condition '{prec}'",
                                involved_tasks=[producer, clobberers[0], t],
                                involved_proposition=prec
                            ))
                            continue

                    satisfied = (pkey in state or raw_target in state or str(prec).lower() in state)
                    if not satisfied:
                        flaws.append(CausalFlaw(
                            flaw_type="unsatisfied_precondition",
                            task_id=t,
                            detail=f"Task {t} requires precondition '{prec}' which is unsatisfied in world state",
                            remedy_hint=f"Add an earlier action producing '{prec}' or supply it in initial environment inputs",
                            involved_tasks=[t],
                            involved_proposition=prec
                        ))

            # C. Apply Delete Effects
            for del_prop in action.del_effects:
                dkey = del_prop.positive_key
                state.discard(dkey)
                if del_prop.name == "exists" and del_prop.args:
                    state.discard(del_prop.args[0].lower())
                state.discard(str(del_prop).lower())

            # D. Apply Add Effects
            for add_prop in action.add_effects:
                akey = add_prop.positive_key
                state.add(akey)
                produced_propositions[akey] = t
                if add_prop.name == "exists" and add_prop.args:
                    arg_val = add_prop.args[0].lower()
                    state.add(arg_val)
                    produced_propositions[arg_val] = t
                state.add(str(add_prop).lower())

            for out_file in action.outputs:
                out_low = out_file.lower()
                state.add(out_low)
                state.add(f"exists({out_low})")
                produced_propositions[out_low] = t
                produced_propositions[f"exists({out_low})"] = t

        # 4. Type Mismatch Diagnosis
        by_stem: dict[str, list[tuple[str, int]]] = {}
        for a in actions:
            for out in a.outputs:
                stem = Path(out).stem.lower()
                ext = Path(out).suffix.lower()
                by_stem.setdefault(stem, []).append((ext, a.id))

        for a in actions:
            for inp in a.inputs:
                stem = Path(inp).stem.lower()
                ext = Path(inp).suffix.lower()
                for prod_ext, prod_id in by_stem.get(stem, []):
                    if prod_ext and ext and prod_ext != ext:
                        flaws.append(CausalFlaw(
                            flaw_type="type_mismatch",
                            task_id=a.id,
                            detail=f"Type mismatch: Task {a.id} consumes '{inp}' ({ext}) but Task {prod_id} produces '{stem}{prod_ext}'",
                            remedy_hint=f"Align file extensions: change '{inp}' to '{stem}{prod_ext}'",
                            involved_tasks=[prod_id, a.id]
                        ))

        # 5. Explicit Target Proposition Reachability
        for target in ast.target_propositions:
            tkey = target.positive_key
            if target.negated:
                if tkey in state:
                    flaws.append(CausalFlaw(
                        flaw_type="unreachable_goal",
                        task_id=0,
                        detail=f"Target goal '{target}' unmet: '{tkey}' remains in final world state",
                        remedy_hint=f"Add a concluding action to delete '{tkey}'",
                        involved_tasks=[]
                    ))
            else:
                if tkey not in state and str(target).lower() not in state:
                    flaws.append(CausalFlaw(
                        flaw_type="unreachable_goal",
                        task_id=0,
                        detail=f"Target goal '{target}' unmet: not achieved in final world state",
                        remedy_hint=f"Add an action to produce '{target}'",
                        involved_tasks=[]
                    ))

        # 6. Dead Artifacts
        all_outputs = {out for a in actions for out in a.outputs}
        all_inputs = {inp for a in actions for inp in a.inputs}
        dead_artifacts = sorted(all_outputs - all_inputs)

        # 7. Global Resource Constraints
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
