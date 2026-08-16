"""Tests for AST-Level Evolutionary Search Engine."""
from __future__ import annotations

import pytest
from plan_mode.ast_search import ASTSearchEngine, ast_distance, crossover_ast, mutate_flaw_directed
from plan_mode.causal_validator import ActionSchema, PlanAST, PlanParser, Proposition


def test_ast_distance_computation():
    """Verify Jaccard graph distance between different PlanASTs."""
    p1 = PlanParser.parse_plan("1. Task A\nOutput: a.txt\n2. Task B\nDepends on 1\nOutput: b.txt")
    p2 = PlanParser.parse_plan("1. Task A\nOutput: a.txt\n2. Task B\nDepends on 1\nOutput: b.txt")
    p3 = PlanParser.parse_plan("1. Task X\nOutput: x.txt\n2. Task Y\nDepends on 1\nOutput: y.txt")

    assert ast_distance(p1, p2) == 0.0
    assert ast_distance(p1, p3) > 0.5


def test_crossover_ast():
    """Verify structural crossover of subgraphs from two parent plans."""
    p1 = PlanParser.parse_plan("1. Step A\nOutput: a.txt\n2. Step B\nDepends on 1\nOutput: b.txt")
    p2 = PlanParser.parse_plan("1. Step C\nOutput: c.txt\n2. Step D\nDepends on 1\nOutput: d.txt")

    child = crossover_ast(p1, p2)
    assert len(child.actions) >= 2
    # Verify continuous renumbering
    assert child.action_ids == list(range(1, len(child.actions) + 1))


def test_flaw_directed_mutation_repairs_unsatisfied_dependency():
    """Verify that flaw-directed mutation repairs an unsatisfied dependency flaw."""
    # Plan with backward/broken dependency
    broken_plan = PlanParser.parse_plan("1. Task Two\nDepends on 2\nOutput: b.txt\n2. Task One\nOutput: a.txt")
    flaw = {"type": "unsatisfied_dependency", "task_id": 1, "detail": "dependency on task 2 not executed"}

    repaired = mutate_flaw_directed(broken_plan, {"flaws": [flaw]})
    assert len(repaired.actions) == 2
    assert repaired.action_ids == [1, 2]


def test_ast_search_engine_evolution():
    """Verify evolutionary population search step and diversity tracking."""
    engine = ASTSearchEngine(objective="Deploy inference cluster")
    p1 = PlanParser.parse_plan("1. Step A\nOutput: a.txt\n2. Step B\nDepends on 1\nOutput: b.txt")
    p2 = PlanParser.parse_plan("1. Step X\nOutput: x.txt\n2. Step Y\nDepends on 1\nOutput: y.txt")

    engine.evaluate_ast(p1)
    engine.evaluate_ast(p2)
    engine.population = [engine.evaluate_ast(p1), engine.evaluate_ast(p2)]

    evolved = engine.evolve_step(population_size=2)
    assert len(evolved) >= 2
    assert all(m.diversity_score >= 0.0 for m in evolved)
