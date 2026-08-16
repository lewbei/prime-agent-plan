"""Tests for AST-Level Evolutionary Search Engine."""
from __future__ import annotations

import pytest
from plan_mode.ast_search import ASTSearchEngine, ast_distance, crossover_ast, mutate_flaw_directed
from plan_mode.causal_validator import ActionSchema, CausalValidator, PlanAST, PlanParser, Proposition


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


def test_ast_crossover_semantic_dependency_remapping():
    """Verify that crossover correctly remaps dependencies from parent B to new child indices."""
    p1 = PlanParser.parse_plan("1. Step A1\nOutput: a1.txt\n2. Step A2\nDepends on 1\nOutput: a2.txt")
    p2 = PlanParser.parse_plan("1. Step B1\nOutput: b1.txt\n2. Step B2\nDepends on 1\nOutput: b2.txt\n3. Step B3\nDepends on 2\nOutput: b3.txt")

    child = crossover_ast(p1, p2)
    # Child must have actions with valid causal dependencies pointing to strictly earlier indices
    for action in child.actions:
        assert all(d < action.id for d in action.depends_on)
    # Suffix action (B3) should depend on new renumbered B2 index
    val = CausalValidator.validate(child)
    assert not any(f["type"] == "unsatisfied_dependency" for f in val["flaws"])


def test_ast_search_preserves_parent_rubric_sections():
    """Verify that evaluate_ast and render_plan preserve parent markdown headers and success criteria."""
    parent_plan = """# Goal
Goal: Build a high-throughput stream ingestion pipeline.

## Success Criteria
- S1: 10k events/sec throughput
- S2: zero data loss

## Tasks
1. Setup Kafka
   Output: kafka.yaml
2. Deploy Pipeline
   Depends on 1
   Inputs: kafka.yaml
   Output: pipeline.log

## Risks
- Risk 1: network partition -> auto-reconnect
"""
    engine = ASTSearchEngine(objective="Kafka stream pipeline", source_plan_text=parent_plan)
    ast = PlanParser.parse_plan(parent_plan)
    member = engine.evaluate_ast(ast, source_plan_text=parent_plan)

    # Rendered text must contain parent sections
    rendered = member.plan_text
    assert "## Success Criteria" in rendered
    assert "10k events/sec throughput" in rendered
    assert "## Risks" in rendered
    assert "Setup Kafka" in rendered


def test_ast_render_avoids_duplicate_task_headers():
    """Verify render_plan does not duplicate ## Tasks header."""
    plan_text = """# Objective
Goal: Test clean rendering

## Tasks
1. Clean Task. Output: a.txt.
"""
    engine = ASTSearchEngine(objective="Test clean rendering", source_plan_text=plan_text)
    ast = PlanParser.parse_plan(plan_text)
    member = engine.evaluate_ast(ast)

    # Must not contain duplicate ## Tasks
    assert member.plan_text.count("## Tasks") == 1
    # Must have clean task action name
    assert "1. Clean Task" in member.plan_text


def test_ast_search_with_pre_task_sections_no_duplication():
    """Verify that plans with pre-task sections (e.g. ## Risks before ## Tasks) render without duplication."""
    plan_with_pre_risks = """# Objective
Goal: Test pre-task sections

## Risks
- Risk 1: High memory usage

## Tasks
1. Task Alpha
   Output: alpha.json
2. Task Beta
   Depends on 1
   Inputs: alpha.json
   Output: beta.json

## Post Verification
- Step V1: Verify beta.json
"""
    engine = ASTSearchEngine(objective="Test pre-task sections", source_plan_text=plan_with_pre_risks)
    ast = PlanParser.parse_plan(plan_with_pre_risks)
    member = engine.evaluate_ast(ast, source_plan_text=plan_with_pre_risks)

    rendered = member.plan_text
    # ## Risks should appear exactly once (in header)
    assert rendered.count("## Risks") == 1
    # ## Tasks should appear exactly once
    assert rendered.count("## Tasks") == 1
    # ## Post Verification should appear in footer
    assert "## Post Verification" in rendered
    assert "Task Alpha" in rendered
    assert "Task Beta" in rendered
