"""Structural AST-Level Evolutionary & MCTS Search Engine.

Implements structural plan crossover with causal dependency remapping,
flaw-directed mutation operators, population diversity maintenance, and state-graph transposition tables.

Literature grounding:
- Mind Evolution (2501.09891): Evaluator-driven recombination of structured plan subgraphs.
- Diversity Maintenance (2509.22613): Explicit population diversity tracking and Pareto fitness.
- LATS (2310.04406) & SYMPHONY (2601.22623): MCTS over state graphs with UCB1 candidate selection.
- GATS (2607.08894): Layered world model and state transposition tables.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .causal_validator import ActionSchema, CausalFlaw, CausalValidator, PlanAST, PlanParser, Proposition


# ---------------------------------------------------------------------------
# 1. Structural Crossover & Flaw-Directed Mutations
# ---------------------------------------------------------------------------

def ast_distance(ast_a: PlanAST, ast_b: PlanAST) -> float:
    """Compute Jaccard graph-edit distance between two PlanASTs: D(A, B) in [0, 1]."""
    actions_a = {a.name.lower() for a in ast_a.actions}
    actions_b = {a.name.lower() for a in ast_b.actions}

    deps_a = {(a.id, dep) for a in ast_a.actions for dep in a.depends_on}
    deps_b = {(b.id, dep) for b in ast_b.actions for dep in b.depends_on}

    union_actions = len(actions_a | actions_b) or 1
    inter_actions = len(actions_a & actions_b)
    action_dist = 1.0 - (inter_actions / union_actions)

    union_deps = len(deps_a | deps_b) or 1
    inter_deps = len(deps_a & deps_b)
    dep_dist = 1.0 - (inter_deps / union_deps)

    return 0.6 * action_dist + 0.4 * dep_dist


def crossover_ast(parent_a: PlanAST, parent_b: PlanAST) -> PlanAST:
    """Crossover two plan ASTs with exact semantic dependency remapping."""
    if not parent_a.actions or not parent_b.actions:
        return copy.deepcopy(parent_a if parent_a.actions else parent_b)

    cut_a = max(1, len(parent_a.actions) // 2)
    prefix = [copy.deepcopy(a) for a in parent_a.actions[:cut_a]]

    prefix_names = {a.name.lower() for a in prefix}
    b_id_map: dict[int, int] = {}
    suffix: list[ActionSchema] = []

    current_new_id = len(prefix) + 1
    for b_action in parent_b.actions:
        if b_action.name.lower() not in prefix_names:
            copied = copy.deepcopy(b_action)
            b_id_map[b_action.id] = current_new_id
            copied.id = current_new_id
            suffix.append(copied)
            current_new_id += 1

    for action in suffix:
        remapped_deps: list[int] = []
        for old_dep in action.depends_on:
            if old_dep in b_id_map:
                remapped_deps.append(b_id_map[old_dep])
            else:
                remapped_deps.append(len(prefix))
        action.depends_on = sorted(set(d for d in remapped_deps if d < action.id))

    combined_actions = prefix + suffix
    for i, a in enumerate(combined_actions, 1):
        a.id = i
        a.depends_on = sorted(set(d for d in a.depends_on if d < i))

    child = PlanAST(
        goal=parent_a.goal,
        actions=combined_actions,
        initial_state=parent_a.initial_state | parent_b.initial_state,
        constraints=copy.deepcopy(parent_a.constraints),
        metadata=copy.deepcopy(parent_a.metadata)
    )
    return child


def mutate_flaw_directed(ast: PlanAST, validation_result: dict[str, Any]) -> PlanAST:
    """Targeted mutation: directly repairs flaws identified by the CausalValidator."""
    mutated = copy.deepcopy(ast)
    flaws = validation_result.get("flaws", [])
    if not flaws:
        return mutate_exploratory(mutated)

    flaw = flaws[0]
    flaw_type = flaw.get("type")
    task_id = flaw.get("task_id")

    if flaw_type == "unsatisfied_dependency":
        action = mutated.get_action(task_id)
        if action and action.depends_on:
            dep_id = action.depends_on[0]
            dep_action = mutated.get_action(dep_id)
            if dep_action and mutated.actions.index(action) < mutated.actions.index(dep_action):
                mutated.actions.remove(dep_action)
                idx = mutated.actions.index(action)
                mutated.actions.insert(idx, dep_action)

    elif flaw_type == "unsatisfied_precondition":
        action = mutated.get_action(task_id)
        if action:
            idx = mutated.actions.index(action)
            producer = ActionSchema(
                id=999,
                name=f"Prepare environment for {action.name}",
                add_effects=copy.deepcopy(action.preconditions),
                outputs=[f"prep_{task_id}.json"]
            )
            mutated.actions.insert(idx, producer)

    elif flaw_type == "clobber_threat":
        action = mutated.get_action(task_id)
        if action:
            mutated.actions.remove(action)
            mutated.actions.append(action)

    elif flaw_type == "type_mismatch":
        action = mutated.get_action(task_id)
        if action and action.inputs:
            action.inputs = [re.sub(r"\.\w+$", ".json", inp) for inp in action.inputs]

    for i, a in enumerate(mutated.actions, 1):
        a.id = i
        a.depends_on = sorted(set(d for d in a.depends_on if d < i))

    return mutated


def mutate_exploratory(ast: PlanAST) -> PlanAST:
    """Exploratory mutation: adds checkpointing, dependency tightening, or sub-action refinement."""
    mutated = copy.deepcopy(ast)
    if not mutated.actions:
        return mutated

    mutation_choice = random.choice(["tighten_deps", "add_milestone"])

    if mutation_choice == "tighten_deps" and len(mutated.actions) > 1:
        target_idx = random.randint(1, len(mutated.actions) - 1)
        prev_id = mutated.actions[target_idx - 1].id
        if prev_id not in mutated.actions[target_idx].depends_on:
            mutated.actions[target_idx].depends_on.append(prev_id)
            mutated.actions[target_idx].depends_on.sort()

    elif mutation_choice == "add_milestone":
        target_idx = random.randint(0, len(mutated.actions) - 1)
        action = mutated.actions[target_idx]
        if not action.outputs:
            action.outputs.append(f"task_{action.id}_result.json")
            action.add_effects.append(Proposition(name="verified", args=(f"step_{action.id}",)))

    return mutated


# ---------------------------------------------------------------------------
# 2. Evolutionary Population Manager & Transposition Table
# ---------------------------------------------------------------------------

@dataclass
class PopulationMember:
    ast: PlanAST
    plan_text: str
    validation: dict[str, Any]
    score: float
    effective_fitness: float
    diversity_score: float = 0.0


class ASTSearchEngine:
    """Evolutionary MCTS & Genetic Search Engine over Plan ASTs."""

    def __init__(self, objective: str, initial_state: Optional[Set[str]] = None, source_plan_text: str = ""):
        self.objective = objective
        self.initial_state = initial_state or set()
        self.source_plan_text = source_plan_text
        self.population: list[PopulationMember] = []
        self.transposition_table: dict[str, dict[str, Any]] = {}

    def _state_hash(self, ast: PlanAST) -> str:
        """Compute state hash for transposition table."""
        action_names = tuple(a.name.lower() for a in ast.actions)
        deps = tuple((a.id, tuple(a.depends_on)) for a in ast.actions)
        return hashlib.sha256(json.dumps([action_names, deps]).encode("utf-8")).hexdigest()

    def evaluate_ast(self, ast: PlanAST, rubric_score_fn: Optional[Callable[[str], float]] = None, source_plan_text: str = "") -> PopulationMember:
        """Evaluate a PlanAST: validates causal consistency, computes rubric score and effective fitness."""
        src_text = source_plan_text or self.source_plan_text
        if src_text and not ast.metadata.get("header_section"):
            m_first = re.search(r"^\s*1[.)]\s+", source_plan_text, re.M)
            if m_first:
                ast.metadata["header_section"] = source_plan_text[:m_first.start()].strip()
            m_footer = re.search(r"^##\s+(?:Risks|Milestones|Rollback|Constraints|Verification)", source_plan_text, re.M)
            if m_footer:
                ast.metadata["footer_section"] = source_plan_text[m_footer.start():].strip()

        val = CausalValidator.validate(ast, self.initial_state)
        shash = self._state_hash(ast)

        plan_text = self.render_plan(ast)
        raw_score = rubric_score_fn(plan_text) if rubric_score_fn else (90.0 if val["ok"] else 50.0)

        num_flaws = len(val.get("flaws", []))
        effective_fitness = max(0.0, raw_score - (25.0 * num_flaws))

        member = PopulationMember(
            ast=ast,
            plan_text=plan_text,
            validation=val,
            score=raw_score,
            effective_fitness=effective_fitness
        )
        self.transposition_table[shash] = {"member": member, "fitness": effective_fitness}
        return member

    def render_plan(self, ast: PlanAST) -> str:
        """Render a PlanAST into formatted markdown text preserving parent rubric sections."""
        header = ast.metadata.get("header_section")
        footer = ast.metadata.get("footer_section")

        task_lines = ["## Tasks"]
        for a in ast.actions:
            task_lines.append(f"{a.id}. {a.name}")
            if a.depends_on:
                task_lines.append(f"   Depends on: {', '.join(str(d) for d in a.depends_on)}")
            if a.inputs:
                task_lines.append(f"   Inputs: {', '.join(a.inputs)}")
            if a.outputs:
                task_lines.append(f"   Output: {', '.join(a.outputs)}")
            if a.preconditions:
                task_lines.append(f"   Preconditions: {', '.join(str(p) for p in a.preconditions)}")
            if a.add_effects:
                task_lines.append(f"   Effects: {', '.join(str(p) for p in a.add_effects)}")
            task_lines.append(f"   Time: {a.duration_minutes:.0f} min | Cost: {a.cost:.0f} tokens")
            task_lines.append("")
        task_block = "\n".join(task_lines)

        if header:
            full_plan = header.strip() + "\n\n" + task_block
            if footer:
                full_plan += "\n\n" + footer.strip()
            return full_plan

        lines = [f"# Plan: {ast.goal}", "", "## Objective & Goal", f"Goal: {ast.goal}", "", task_block]
        return "\n".join(lines)

    def evolve_step(self, population_size: int = 4, rubric_score_fn: Optional[Callable[[str], float]] = None) -> list[PopulationMember]:
        """Perform one evolutionary search step: Crossover + Flaw-Directed Mutations + Diversity Pruning."""
        if not self.population:
            return []

        new_candidates: list[PlanAST] = []

        for member in self.population[:population_size]:
            mutated = mutate_flaw_directed(member.ast, member.validation)
            new_candidates.append(mutated)

        if len(self.population) >= 2:
            p1 = random.choice(self.population[:population_size])
            p2 = random.choice(self.population[:population_size])
            if p1 != p2:
                child = crossover_ast(p1.ast, p2.ast)
                new_candidates.append(child)

        for cand in new_candidates:
            evaluated = self.evaluate_ast(cand, rubric_score_fn)
            self.population.append(evaluated)

        for i, m1 in enumerate(self.population):
            div = sum(ast_distance(m1.ast, m2.ast) for j, m2 in enumerate(self.population) if i != j)
            m1.diversity_score = div / max(1, len(self.population) - 1)

        self.population.sort(key=lambda m: -(m.effective_fitness + 5.0 * m.diversity_score))
        self.population = self.population[:population_size * 2]
        return self.population
